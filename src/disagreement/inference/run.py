"""Command line of the frozen inference: ``python -m disagreement.inference.run``.

Without ``--execute`` nothing is loaded or estimated: the command only states
the calendar it would use and where it would publish. With ``--execute`` it
runs the frozen baseline (calendar from the configuration, frozen estimators,
HAC lags 6, 0 and 12), records the SHA-256 of every input file, and publishes
the result set (``disagreement.inference.output``). An existing result set is
kept unless ``--replace`` is given, and only a verifiable published inference
result set is ever replaced; a destination that is a link, a protected research
path or inside a pipeline data directory is refused before anything is loaded.
Output is administrative only: no coefficient, return or verdict is printed.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from disagreement.data.config import DEFAULT_CONFIG, DEFAULT_DATA_DIR, DataPaths
from disagreement.data.validation import DataValidationError
from disagreement.inference.orchestrate import baseline_calendar
from disagreement.inference.output import PublicationError, default_output_dir, execute_inference


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m disagreement.inference.run",
                                     description="Run and publish the frozen H1-H3 inference.")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--output-dir", type=Path, default=None, help="default: <data-dir>/diagnostics/inference")
    parser.add_argument("--execute", action="store_true", help="load the outcomes, estimate and publish")
    parser.add_argument("--replace", action="store_true", help="supersede an existing, verifiable published result set")
    args = parser.parse_args(argv)
    paths = DataPaths(args.data_dir)
    destination = args.output_dir or default_output_dir(paths)
    try:
        grid, years = baseline_calendar(args.config)
    except (OSError, ValueError, KeyError) as exc:
        print(f"Invalid configuration: {exc}", file=sys.stderr)
        return 2
    if not args.execute:
        print(f"Dry run: {len(grid)} months ({grid[0]}-{grid[-1]}, {len(years)} OOS years) would be estimated "
              f"and published to {destination}. Nothing was loaded; pass --execute to run.")
        return 0
    try:
        summary = execute_inference(paths, args.config, destination, replace=args.replace)
    except (DataValidationError, PublicationError, ValueError, OSError) as exc:
        print(f"Inference not published: {exc}", file=sys.stderr)
        return 2
    print(f"Published {summary['grid']['n_months']} months to {destination} "
          f"(manifest {summary['manifest_sha256'][:12]}).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
