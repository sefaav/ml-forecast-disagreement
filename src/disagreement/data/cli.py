"""Command line: ``python -m disagreement.data {crsp-extract,build}``.

Commands print aggregate information only. A failed data check prints its
message, which holds counts and column names only.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

from .config import DEFAULT_CONFIG, DEFAULT_DATA_DIR, DataPaths, load_data_config
from .validation import DataValidationError


def cmd_crsp_extract(args, cfg, paths: DataPaths) -> None:
    from . import crsp

    conn = crsp.connect(args.wrds_username or os.environ.get("WRDS_USERNAME"))
    try:
        print(f"Extracting {cfg.crsp_library}.{cfg.crsp_table} (row counts per year):")
        manifest = crsp.extract_monthly(conn, cfg.crsp_library, cfg.crsp_table, paths.crsp(cfg.crsp_table))
    finally:
        conn.close()
    n_rows = sum(manifest["rows_by_year"].values())
    print(f"  {manifest['table_first_date']} to {manifest['table_last_date']}: {n_rows:,} rows")


def cmd_build(args, cfg, paths: DataPaths) -> None:
    from .pipeline import build

    summary = build(paths, cfg, args.osap_file)
    print("Build summary:")
    for key, value in summary.items():
        if isinstance(value, float):
            value = f"{value:.2%}" if key.endswith("rate") else f"{value:,.4g}"
        elif isinstance(value, int) and not key.endswith("month"):
            value = f"{value:,}"
        print(f"  {key}: {value}")


def _parser() -> argparse.ArgumentParser:
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    common.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser = argparse.ArgumentParser(prog="python -m disagreement.data")
    sub = parser.add_subparsers(dest="command", required=True)

    extract = sub.add_parser("crsp-extract", parents=[common], help="download the CRSP CIZ fields from WRDS")
    extract.add_argument("--wrds-username", help="defaults to $WRDS_USERNAME, else wrds prompts")
    extract.set_defaults(handler=cmd_crsp_extract)

    build = sub.add_parser("build", parents=[common], help="build and validate the universe and panel")
    build.add_argument("--osap-file", type=Path, help="default: data/raw/osap/<file from config>")
    build.set_defaults(handler=cmd_build)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        args.handler(args, load_data_config(args.config), DataPaths(args.data_dir))
    except DataValidationError as exc:
        print(f"Data validation failed: {exc}", file=sys.stderr)
        return 2
    return 0
