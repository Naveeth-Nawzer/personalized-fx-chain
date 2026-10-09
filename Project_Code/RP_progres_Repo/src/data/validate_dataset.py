"""Entry point: ``python -m src.data.validate_dataset --help``."""

import sys

from src.cli.validate import main

if __name__ == "__main__":
    sys.exit(main())
