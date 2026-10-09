"""Generate the Phase 1 dataset. Same as ``python -m src.data.generate_dataset``."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.cli.generate import main  # noqa: E402

if __name__ == "__main__":
    sys.exit(main())
