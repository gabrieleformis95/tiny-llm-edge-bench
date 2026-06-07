"""Generate report artifacts from aggregated results.

Usage:
  python scripts/generate_report.py
  make report
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parents[1]))

from src.analysis.aggregate import aggregate_results
from src.analysis.report import generate_report
from src.config import settings


def main() -> None:
    df = aggregate_results(settings.results_dir)
    generate_report(df, settings.reports_dir)
    print(f"Report generated in {settings.reports_dir}/")


if __name__ == "__main__":
    main()
