"""Command-line entry point."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="fatanalyze",
        description="CT ectopic-fat analysis (liver, pancreas, psoas at L3).",
    )
    parser.add_argument("dicom_dir", type=Path, help="Path to DICOM series folder")
    parser.add_argument(
        "--cache-dir", type=Path, default=Path(".cache/totalseg_runs"),
        help="Disk cache for segmentation results",
    )
    parser.add_argument(
        "--out-dir", type=Path, default=Path("fatanalyze-out"),
        help="Output directory for figures and metrics",
    )
    parser.add_argument(
        "--device", choices=["cpu", "cuda", "mps"], default="cpu",
    )
    parser.add_argument(
        "--targets", nargs="+", default=None,
        help="Subset of targets to analyze (default: all configured)",
    )
    args = parser.parse_args(argv)

    from fatanalyze.io.dicom_loader import load_ct_series
    from fatanalyze.segment.totalseg import segment
    from fatanalyze.config import load_default_config

    cfg = load_default_config()
    targets = args.targets or list(cfg["targets"].keys())

    image, qc = load_ct_series(args.dicom_dir)
    print(f"[QC] {qc.summary()}")

    masks = segment(
        image, roi_names=targets, cache_dir=args.cache_dir, device=args.device,
    )

    args.out_dir.mkdir(parents=True, exist_ok=True)
    summary_path = args.out_dir / "summary.json"
    summary_path.write_text(json.dumps(
        {"qc": qc.to_dict(), "masks": list(masks.keys())}, indent=2,
    ))
    print(f"Wrote {summary_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
