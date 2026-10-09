"""Create a small synthetic MUSDB-like dataset to try the pipeline without real data.

    python scripts/make_toy_dataset.py --out data/raw/toy --num-songs 8 --duration 15
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.utils.synthetic import make_toy_dataset  # noqa: E402


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--out", type=Path, required=True, help="Output directory for the toy dataset")
    p.add_argument("--num-songs", type=int, default=8)
    p.add_argument("--duration", type=float, default=15.0, help="Seconds per stem")
    p.add_argument("--sample-rate", type=int, default=44100)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()
    files = make_toy_dataset(args.out, num_songs=args.num_songs, duration=args.duration,
                             sample_rate=args.sample_rate, seed=args.seed)
    print(f"Wrote {len(files)} files to {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
