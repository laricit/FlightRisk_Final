from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parent
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from flightrisk.data import load_dataset, prepare_modeling_frame, time_split_dataset
from flightrisk.modeling import train_bundle


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train the Flight Risk prototype model.")
    parser.add_argument("--data-path", required=True, help="Path to CSV or parquet input data.")
    parser.add_argument(
        "--output-dir",
        default="artifacts",
        help="Directory where the trained model bundle and metrics will be stored.",
    )
    parser.add_argument(
        "--sample-size",
        type=int,
        default=None,
        help="Optional row sample size for faster iteration.",
    )
    parser.add_argument(
        "--min-route-support",
        type=int,
        default=15,
        help="Minimum historical route option support to expose in the demo app.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    df = load_dataset(args.data_path, sample_size=args.sample_size)
    modeling_df = prepare_modeling_frame(df)
    split = time_split_dataset(modeling_df)

    bundle, metrics = train_bundle(split, min_route_support=args.min_route_support)

    bundle_path = output_dir / "flight_risk_bundle.joblib"
    metrics_path = output_dir / "metrics.json"
    bundle.save(bundle_path)
    metrics_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    print(f"Saved model bundle to: {bundle_path}")
    print(f"Saved metrics to: {metrics_path}")
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
