"""Entry point: ``python -m src.data.generate_dataset --help``."""

import sys

from src.cli.generate import main

if __name__ == "__main__":
    sys.exit(main())
