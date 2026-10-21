"""Main CLI application coordinator for PTA + XAI terminal application."""
import sys
import argparse
from typing import Callable, List, Optional
import numpy as np

# Safe console encoding for Windows terminals
if sys.platform == "win32" and hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

from pta.storage.json_store import JSONPreferenceStore
from pta.core.session_manager import SessionManager
from pta.models.schemas import VECTOR_DIM, VECTOR_LABELS
from pta.cli.session_runner import InteractiveSession
from pta.cli.display import (
    banner,
    format_single_vector_table,
    CYAN,
    GREEN,
    YELLOW,
    RED,
    BRIGHT,
    DIM,
    RESET,
)


class PTAInteractiveApp:
    """Top-level terminal application for PTA + Explainable AI."""

    def __init__(
        self,
        base_dir: str = "data",
        input_fn: Callable[[str], str] = input,
        print_fn: Callable[..., None] = print,
    ):
        self.base_dir = base_dir
        self.store = JSONPreferenceStore(base_dir=base_dir)
        self.input_fn = input_fn
        self.print_fn = print_fn

    def run(self) -> None:
        """Main interaction menu loop."""
        while True:
            self.print_fn(
                banner(
                    "PERSONALIZED AUDIO PTA + XAI SYSTEM",
                    "Live Preference Tracking & Grounded Explanation Suite",
                )
            )
            self.print_fn(f"{BRIGHT}MAIN MENU:{RESET}")
            self.print_fn(f"  {BRIGHT}1.{RESET} Create new user & start session")
            self.print_fn(f"  {BRIGHT}2.{RESET} Continue as existing user")
            self.print_fn(f"  {BRIGHT}3.{RESET} List all saved user profiles")
            self.print_fn(f"  {BRIGHT}4.{RESET} System & XAI grounding specifications")
            self.print_fn(f"  {RED}5.{RESET} Exit")
            self.print_fn("-" * 62)

            choice = self.input_fn(f"{BRIGHT}Select an option (1-5): {RESET}").strip()

            if choice == "1":
                self.handle_new_user()
            elif choice == "2":
                self.handle_existing_user()
            elif choice == "3":
                self.handle_list_users()
            elif choice == "4":
                self.handle_system_status()
            elif choice in ("5", "q", "exit", "quit"):
                self.print_fn(f"\n{CYAN}Thank you for using the PTA + XAI Personalization System.{RESET}\n")
                break
            else:
                self.print_fn(f"{RED}Invalid option '{choice}'. Please select a number between 1 and 5.{RESET}\n")

    def _next_user_id(self) -> str:
        """Generate next available sequential user ID."""
        users = self.store.list_users()
        existing_ids = {u["user_id"] for u in users if u.get("user_id")}
        idx = 1
        while f"USER-{idx:03d}" in existing_ids:
            idx += 1
        return f"USER-{idx:03d}"

    def handle_new_user(self) -> Optional[dict]:
        """Workflow: Create a brand new user profile and start a session."""
        self.print_fn(f"\n{BRIGHT}--- CREATE NEW USER PROFILE ---{RESET}")
        default_uid = self._next_user_id()
        uid_input = self.input_fn(f"Enter User ID [default: {default_uid}]: ").strip()
        user_id = uid_input if uid_input else default_uid

        inst_input = self.input_fn("Enter Target Instrument [default: electric_guitar]: ").strip()
        instrument = inst_input if inst_input else "electric_guitar"

        genre_input = self.input_fn("Enter Target Genre [default: rock]: ").strip()
        genre = genre_input if genre_input else "rock"

        # Initialize profile with neutral preferences
        self.store.save_profile(user_id, [0.0] * VECTOR_DIM, instrument=instrument, genre=genre)

        self.print_fn(f"\n{GREEN}[OK] Created persistent profile for '{user_id}' ({instrument} / {genre}){RESET}")

        session_mgr = SessionManager(
            user_id=user_id,
            instrument=instrument,
            genre=genre,
            store=self.store,
            initial_vector=np.zeros(VECTOR_DIM),
        )

        runner = InteractiveSession(
            session_mgr=session_mgr,
            prev_profile=None,
            input_fn=self.input_fn,
            print_fn=self.print_fn,
        )
        return runner.run()

    def handle_existing_user(self) -> Optional[dict]:
        """Workflow: Load an existing user profile and start a new session."""
        self.print_fn(f"\n{BRIGHT}--- LOAD EXISTING USER PROFILE ---{RESET}")
        users = self.store.list_users()

        if not users:
            self.print_fn(f"{YELLOW}No saved user profiles found in storage directory '{self.base_dir}/users'.{RESET}")
            create_choice = self.input_fn("Would you like to create a new user now? (y/N): ").strip().lower()
            if create_choice == "y":
                return self.handle_new_user()
            return None

        self.print_fn("-" * 72)
        self.print_fn(f"{'#':<3} {'User ID':<12} {'Instrument':<16} {'Genre':<10} {'Taste Highlights':<26}")
        self.print_fn("-" * 72)

        for i, u in enumerate(users, 1):
            vec = u.get("vector", [0.0] * VECTOR_DIM)
            highlights = []
            for dim_name, val in zip(VECTOR_LABELS, vec):
                if abs(val) >= 0.05:
                    highlights.append(f"{dim_name[:3]}:{val:+.2f}")
            hl_str = ", ".join(highlights) if highlights else "Neutral (all 0.0)"
            self.print_fn(
                f"{i:<3} {u['user_id']:<12} {u['instrument']:<16} {u['genre']:<10} {hl_str:<26}"
            )
        self.print_fn("-" * 72)

        sel = self.input_fn(f"{BRIGHT}Select user number (1-{len(users)}) or 'B' to go back: {RESET}").strip()
        if sel.lower() in ("b", "back", "q", "exit"):
            return None

        try:
            sel_idx = int(sel) - 1
            if not (0 <= sel_idx < len(users)):
                raise ValueError()
        except ValueError:
            self.print_fn(f"{RED}Invalid selection '{sel}'. Returning to main menu.{RESET}\n")
            return None

        selected = users[sel_idx]
        user_id = selected["user_id"]
        instrument = selected["instrument"]
        genre = selected["genre"]

        saved_profile = self.store.load_profile(user_id, instrument, genre)
        self.print_fn(f"\n{GREEN}[OK] Loaded saved profile for '{user_id}'. Starting session...{RESET}")

        # In accordance with multi-session design:
        # Every session starts with an independent all-zeros preference vector.
        # The saved profile is passed as prior reference and updated only upon session completion.
        session_mgr = SessionManager(
            user_id=user_id,
            instrument=instrument,
            genre=genre,
            store=self.store,
            initial_vector=np.zeros(VECTOR_DIM),
        )

        runner = InteractiveSession(
            session_mgr=session_mgr,
            prev_profile=saved_profile.copy(),
            input_fn=self.input_fn,
            print_fn=self.print_fn,
        )
        return runner.run()

    def handle_list_users(self) -> None:
        """Display detailed overview of all saved profiles."""
        self.print_fn(f"\n{BRIGHT}SAVED USER PROFILES REPOSITORY{RESET}")
        users = self.store.list_users()
        if not users:
            self.print_fn(f"  {DIM}(No profiles found in {self.base_dir}/users){RESET}\n")
            return

        for idx, u in enumerate(users, 1):
            self.print_fn(f"\n[{idx}] User: {BRIGHT}{u['user_id']}{RESET} | Track: {u['instrument']} ({u['genre']})")
            vec = u.get("vector", [0.0] * VECTOR_DIM)
            self.print_fn(format_single_vector_table("Taste Vector", vec))
        self.print_fn("")

    def handle_system_status(self) -> None:
        """Display architecture specifications and grounding integrity declarations."""
        self.print_fn(
            banner(
                "SYSTEM SPECIFICATIONS & XAI GROUNDING INTEGRITY",
                "Architectural Invariants & Downstream Interface Audit",
            )
        )
        specs = [
            ("Taste Vector Dimensionality", "7 Dimensions (brightness, compression, reverb, delay, saturation, width, warmth)"),
            ("Conditioning Representation", "16 Dimensions (z_taste: 7 linear + 6 interaction terms + 3 energy norms)"),
            ("Supported DSP Processors", "7 Effects: EQ, Compressor, Delay, Reverb, Distortion, Chorus, Flanger"),
            ("Hardware Constraint", "MONO RENDERING (channels=1). Spatial widening is preserved in profile but unrendered."),
            ("Session Scope", "Every new session starts with all zeros [0.0]*7 independently"),
            ("Update Rule", "P_(t+1) = (1 - alpha) * P_t + alpha * F (alpha=0.30, exponential smoothing)"),
            ("Profile Merge Rule", "Selective dimension preservation: untouched session dimensions NEVER decay"),
            ("Storage Layer", f"JSON Preference Store at '{self.base_dir}' (users/, sessions/, feedback/)"),
            ("Truthfulness Guard", "SATISFIED means terminal session state; never implies audio quality improvement"),
            ("Standalone Mode", "Audio rendering status: NOT EXECUTED / SIMULATED; Quality: NOT EVALUATED"),
        ]
        for title, val in specs:
            self.print_fn(f"  {CYAN}{title:<30}:{RESET} {val}")
        self.print_fn("-" * 72)
        self.input_fn("\nPress Enter to return to main menu...")


def main() -> int:
    """CLI entrypoint."""
    parser = argparse.ArgumentParser(description="Personalized Audio PTA + XAI Interactive Terminal")
    parser.add_argument("--data-dir", default="data", help="Directory where user profiles and sessions are persisted")
    args = parser.parse_args()

    app = PTAInteractiveApp(base_dir=args.data_dir)
    try:
        app.run()
    except (KeyboardInterrupt, EOFError):
        print("\n\nOperation interrupted. Exiting cleanly.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
