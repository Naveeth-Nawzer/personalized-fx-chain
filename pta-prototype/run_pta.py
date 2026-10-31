#!/usr/bin/env python3
"""
Root entry point for the Personalized Audio PTA + XAI Interactive Terminal.

Allows live interactive participation by an examiner or user to:
1. Create new persistent profiles with neutral starting vectors.
2. Resume existing user profiles with baseline comparison.
3. Test real preference updates with before/after diff tables.
4. Inspect evidence-grounded XAI explanation cards.
5. Inspect the derived 16-D conditioning vector (z_taste).
6. Finalize session with selective dimension preservation.

Usage:
    python run_pta.py
    python run_pta.py --data-dir data
"""
import sys
from pathlib import Path

repo_root = Path(__file__).resolve().parent
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

from pta.cli.app import main

if __name__ == "__main__":
    sys.exit(main())
