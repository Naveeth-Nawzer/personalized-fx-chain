"""Interactive session runner for PTA + XAI terminal application."""
from typing import Callable, Dict, List, Optional
import numpy as np

from pta.core.session_manager import SessionManager
from pta.core.guidance_builder import project_taste_vector
from pta.models.schemas import VECTOR_LABELS
from pta.cli.display import (
    banner,
    format_single_vector_table,
    format_vector_diff_table,
    format_z_taste_table,
    format_xai_card,
    format_session_summary_report,
    CYAN,
    GREEN,
    YELLOW,
    RED,
    BRIGHT,
    DIM,
    RESET,
)

MENU_OPTIONS = [
    ("1", "More Reverb", "MORE_REVERB"),
    ("2", "Less Reverb", "LESS_REVERB"),
    ("3", "More Compression", "MORE_COMPRESSION"),
    ("4", "Less Compression", "LESS_COMPRESSION"),
    ("5", "More Delay", "MORE_DELAY"),
    ("6", "Less Delay", "LESS_DELAY"),
    ("7", "Brighter (EQ High Shelf)", "BRIGHTER"),
    ("8", "Warmer (EQ Low-Mid Boost)", "WARMER"),
    ("9", "More Saturation (Harmonic Drive)", "MORE_SATURATION"),
    ("10", "Wider (Stereo Image — Mono Pipeline Note)", "WIDER"),
    ("11", "View Explanation History", "ACTION_HISTORY"),
    ("12", "Inspect 16-D Conditioning Vector (z_taste)", "ACTION_Z_TASTE"),
    ("13", "Satisfied / Finish Session", "SATISFIED"),
    ("0", "Cancel / Exit Session Without Merging", "ACTION_CANCEL"),
]


class InteractiveSession:
    """Manages the interactive feedback loop within an active session."""

    def __init__(
        self,
        session_mgr: SessionManager,
        prev_profile: Optional[List[float]] = None,
        input_fn: Callable[[str], str] = input,
        print_fn: Callable[..., None] = print,
    ):
        self.session_mgr = session_mgr
        self.prev_profile = prev_profile
        self.input_fn = input_fn
        self.print_fn = print_fn
        self.user_id = session_mgr.session.user_id
        self.session_id = session_mgr.session.session_id
        self.start_vector = session_mgr.initial_session_vector.tolist()

    def print_menu(self) -> None:
        """Display feedback menu."""
        self.print_fn(f"\n{BRIGHT}AVAILABLE FEEDBACK ACTIONS & TOOLS:{RESET}")
        self.print_fn("-" * 62)
        for num, desc, _ in MENU_OPTIONS:
            if num in ("11", "12"):
                self.print_fn(f"  {CYAN}{num:>2}.{RESET} {desc}")
            elif num == "13":
                self.print_fn(f"  {GREEN}{num:>2}.{RESET} {BRIGHT}{desc}{RESET}")
            elif num == "0":
                self.print_fn(f"  {RED}{num:>2}.{RESET} {desc}")
            else:
                self.print_fn(f"  {BRIGHT}{num:>2}.{RESET} {desc}")
        self.print_fn("-" * 62)

    def run(self) -> Optional[dict]:
        """Run the interactive feedback loop until satisfaction or cancellation."""
        self.print_fn(
            banner(
                f"ACTIVE SESSION: {self.session_id}",
                f"User ID: {self.user_id} | Target: {self.session_mgr.session.instrument} ({self.session_mgr.session.genre})",
            )
        )

        if self.prev_profile is not None:
            self.print_fn(f"\n{CYAN}--- Existing User Profile Loaded ---{RESET}")
            self.print_fn(
                format_single_vector_table("1. Saved Long-Term Profile (Prior Learning Across Sessions)", self.prev_profile)
            )
            self.print_fn(
                format_single_vector_table("2. New Session Preference Vector (Starts Neutral at 0.00)", self.start_vector)
            )
            self.print_fn(f"\n{DIM}Note: Feedback will update the session vector locally. Long-term profile updates upon session completion.{RESET}\n")
        else:
            self.print_fn(
                format_single_vector_table("Initial Session Preference Vector (Starts Neutral at 0.00)", self.start_vector)
            )

        while True:
            self.print_menu()
            choice = self.input_fn(f"{BRIGHT}Select an option (0-13): {RESET}").strip()

            matching = [opt for opt in MENU_OPTIONS if opt[0] == choice]
            if not matching:
                self.print_fn(f"{RED}Invalid selection '{choice}'. Please select a number from 0 to 13.{RESET}")
                continue

            _, desc, command = matching[0]

            if command == "ACTION_CANCEL":
                confirm = self.input_fn("Abort session without saving changes to profile? (y/N): ").strip().lower()
                if confirm == "y":
                    self.print_fn(f"{YELLOW}Session cancelled. No profile changes were merged.{RESET}")
                    return None
                continue

            if command == "ACTION_HISTORY":
                self._show_history()
                continue

            if command == "ACTION_Z_TASTE":
                self._show_z_taste()
                continue

            if command == "SATISFIED":
                return self._finalize_satisfied()

            # Normal feedback command
            self._apply_feedback(command)

    def _apply_feedback(self, command: str) -> None:
        """Apply a feedback command and show updated vector diff and XAI card."""
        self.print_fn(f"\n{BRIGHT}Applying feedback: {CYAN}{command}{RESET}...")
        try:
            res = self.session_mgr.submit_feedback(command)
        except Exception as e:
            self.print_fn(f"{RED}Error submitting feedback: {e}{RESET}")
            return

        active_dim = res.get("active_dimension")
        before_vec = res["preference_before"]
        after_vec = res["preference_after"]
        explanation = res["explanation"]

        before_val = None
        after_val = None
        if active_dim and active_dim in VECTOR_LABELS:
            idx = VECTOR_LABELS.index(active_dim)
            before_val = before_vec[idx]
            after_val = after_vec[idx]

        # 1. Print vector update table
        self.print_fn(
            format_vector_diff_table(
                before=before_vec,
                after=after_vec,
                prev_long_term=self.prev_profile,
                active_dim=active_dim,
            )
        )

        # 2. Print evidence-grounded XAI explanation card
        self.print_fn(
            format_xai_card(
                command=command,
                active_dimension=active_dim,
                before_val=before_val,
                after_val=after_val,
                explanation_text=explanation,
                alpha=self.session_mgr.alpha,
            )
        )

    def _show_history(self) -> None:
        """Display all recorded events in this session."""
        events = self.session_mgr.session.feedback_events
        self.print_fn(f"\n{BRIGHT}EXPLANATION & EVENT HISTORY ({len(events)} events){RESET}")
        self.print_fn("=" * 72)
        if not events:
            self.print_fn("  No feedback events recorded yet.")
            return

        for idx, ev in enumerate(events, 1):
            cmd = ev.get("command", "")
            dim = ev.get("active_dimension", "")
            b_val = ev.get("before", [])
            a_val = ev.get("after", [])
            delta = ev.get("delta", [])
            ts = ev.get("timestamp", "")
            
            d_str = ""
            if dim in VECTOR_LABELS:
                d_idx = VECTOR_LABELS.index(dim)
                d_val = delta[d_idx] if d_idx < len(delta) else 0.0
                d_str = f"| Delta: {d_val:+.2f}"

            self.print_fn(f"Event #{idx:02d} [{ts}]")
            self.print_fn(f"  Command : {CYAN}{cmd}{RESET}")
            self.print_fn(f"  Target  : {dim.capitalize() if dim else 'N/A'} {d_str}")
            self.print_fn("-" * 72)

    def _show_z_taste(self) -> None:
        """Display the derived 16-D conditioning vector."""
        curr_vec = self.session_mgr.get_session_vector()
        z_taste = project_taste_vector(curr_vec)
        self.print_fn(format_z_taste_table(z_taste))

    def _finalize_satisfied(self) -> dict:
        """Handle SATISFIED termination, merge profile, and print final report."""
        self.print_fn(f"\n{GREEN}User marked session as SATISFIED. Finalizing session...{RESET}")
        self.session_mgr.submit_feedback("SATISFIED")

        # Selective merge into long-term profile
        merged_profile = self.session_mgr.merge_into_long_term()

        final_vec = self.session_mgr.get_session_vector().tolist()
        events = self.session_mgr.session.feedback_events

        profile_path = None
        session_path = None
        if self.session_mgr.store:
            profile_path = str(
                self.session_mgr.store._profile_path(
                    self.user_id,
                    self.session_mgr.session.instrument,
                    self.session_mgr.session.genre,
                )
            )
            session_path = str(
                self.session_mgr.store.base_dir
                / "sessions"
                / f"{self.user_id}__{self.session_id}.json"
            )

        report = format_session_summary_report(
            user_id=self.user_id,
            session_id=self.session_id,
            instrument=self.session_mgr.session.instrument,
            genre=self.session_mgr.session.genre,
            start_vector=self.start_vector,
            final_vector=final_vec,
            prev_profile=self.prev_profile,
            events=events,
            merged_long_term=merged_profile,
            termination_reason="SATISFIED",
            saved_profile_path=profile_path,
            saved_session_path=session_path,
        )
        self.print_fn(report)

        return {
            "user_id": self.user_id,
            "session_id": self.session_id,
            "start_vector": self.start_vector,
            "final_vector": final_vec,
            "merged_profile": merged_profile,
            "iterations": self.session_mgr.session.iterations,
            "events_count": len(events),
            "satisfied": True,
        }
