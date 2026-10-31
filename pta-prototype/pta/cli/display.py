"""Terminal formatting and presentation utilities for PTA + XAI CLI."""
from typing import Dict, List, Optional
from pta.models.schemas import VECTOR_DIM, VECTOR_LABELS, DEFAULT_ALPHA
from pta.core.preference_mapping import FEEDBACK_INTENT_CATALOG

try:
    import colorama
    colorama.init(autoreset=True)
    CYAN = colorama.Fore.CYAN
    GREEN = colorama.Fore.GREEN
    YELLOW = colorama.Fore.YELLOW
    RED = colorama.Fore.RED
    MAGENTA = colorama.Fore.MAGENTA
    WHITE = colorama.Fore.WHITE
    BRIGHT = colorama.Style.BRIGHT
    DIM = colorama.Style.DIM
    RESET = colorama.Style.RESET_ALL
except Exception:
    CYAN = GREEN = YELLOW = RED = MAGENTA = WHITE = BRIGHT = DIM = RESET = ""


def banner(title: str, subtitle: Optional[str] = None) -> str:
    """Format a prominent section header."""
    width = 72
    lines = [
        "=" * width,
        f"{BRIGHT}{title.center(width)}{RESET}",
    ]
    if subtitle:
        lines.append(f"{DIM}{subtitle.center(width)}{RESET}")
    lines.append("=" * width)
    return "\n".join(lines)


def make_bar(val: float, bar_width: int = 16) -> str:
    """
    Render a horizontal bidirectional bar for values in [-1.0, 1.0].
    Center represents 0.0. Negative goes left, positive goes right.
    """
    half = bar_width // 2
    clamped = max(-1.0, min(1.0, float(val)))
    
    if abs(clamped) < 0.02:
        # Zero center
        return f"[{' ' * half}|{' ' * half}]"
    
    if clamped > 0:
        fill_len = int(round(clamped * half))
        fill_len = max(1, min(half, fill_len))
        right_fill = "=" * (fill_len - 1) + ">"
        pad = " " * (half - fill_len)
        return f"[{' ' * half}|{GREEN}{right_fill}{RESET}{pad}]"
    else:
        fill_len = int(round(abs(clamped) * half))
        fill_len = max(1, min(half, fill_len))
        left_fill = "<" + "=" * (fill_len - 1)
        pad = " " * (half - fill_len)
        return f"[{pad}{RED}{left_fill}{RESET}|{' ' * half}]"


def format_single_vector_table(title: str, vector: List[float]) -> str:
    """Display a single 7-D preference vector with bars."""
    lines = [
        f"\n{BRIGHT}{title}{RESET}",
        "-" * 62,
        f"{'Dimension':<16} {'Value':>8}   {'Visual Scale [-1.0 ... +1.0]':<26}",
        "-" * 62,
    ]
    for label, val in zip(VECTOR_LABELS, vector):
        bar = make_bar(val)
        val_str = f"{val:+.2f}"
        if abs(val) < 0.001:
            val_str = " 0.00"
        lines.append(f"{label.capitalize():<16} {val_str:>8}   {bar}")
    lines.append("-" * 62)
    return "\n".join(lines)


def format_vector_diff_table(
    before: List[float],
    after: List[float],
    prev_long_term: Optional[List[float]] = None,
    active_dim: Optional[str] = None,
) -> str:
    """
    Display session vector update comparison table with delta indicators.
    If prev_long_term is provided, displays the persistent long-term profile column.
    """
    if prev_long_term is not None:
        header = f"{'Dimension':<14} {'Long-Term':>10} {'Sess Before':>12} {'Sess After':>12} {'Change':>10}   {'Visual':<18}"
        sep = "-" * 86
    else:
        header = f"{'Dimension':<14} {'Sess Before':>12} {'Sess After':>12} {'Change':>10}   {'Visual':<18}"
        sep = "-" * 72

    lines = [
        f"\n{BRIGHT}SESSION PREFERENCE VECTOR UPDATE{RESET}",
        f"{DIM}Session-local vector tracks feedback from neutral 0.00 baseline{RESET}",
        sep,
        header,
        sep,
    ]

    for i, label in enumerate(VECTOR_LABELS):
        b_val = before[i]
        a_val = after[i]
        delta = a_val - b_val
        bar = make_bar(a_val, bar_width=14)

        if abs(delta) < 0.001:
            delta_str = f"{DIM} 0.00{RESET}"
        elif delta > 0:
            delta_str = f"{GREEN}+{delta:.2f} (+){RESET}"
        else:
            delta_str = f"{RED}{delta:.2f} (-){RESET}"

        highlight = BRIGHT if label == active_dim else ""
        dim_name = f"{highlight}{label.capitalize()}{RESET}"

        if prev_long_term is not None:
            p_val = prev_long_term[i]
            p_str = f"{p_val:+.2f}" if abs(p_val) >= 0.001 else " 0.00"
            lines.append(
                f"{dim_name:<24} {p_str:>10} {b_val:>12.2f} {a_val:>12.2f} {delta_str:>19}   {bar}"
            )
        else:
            lines.append(
                f"{dim_name:<24} {b_val:>12.2f} {a_val:>12.2f} {delta_str:>19}   {bar}"
            )

    lines.append(sep)
    return "\n".join(lines)


def format_z_taste_table(z_taste: List[float]) -> str:
    """Format the 16-D conditioning vector into grouped sections."""
    lines = [
        f"\n{BRIGHT}16-DIMENSIONAL CONDITIONING VECTOR (M3 z_taste Contract){RESET}",
        f"{DIM}Derived deterministically from 7-D taste vector for downstream RL policy conditioning{RESET}",
        "-" * 72,
        f"{'Slot':<6} {'Representation':<28} {'Value':>8}   {'Scale':<18}",
        "-" * 72,
        f"{CYAN}--- Group 1: Direct Taste Dimensions (Slots 0..6) ---{RESET}",
    ]
    direct_names = [f"z[{i}] {dim.capitalize()}" for i, dim in enumerate(VECTOR_LABELS)]
    for i in range(7):
        val = z_taste[i]
        lines.append(f"{i:<6} {direct_names[i]:<28} {val:>+8.2f}   {make_bar(val, 12)}")

    lines.append(f"{CYAN}--- Group 2: Pairwise Interaction Terms (Slots 7..12) ---{RESET}")
    interactions = [
        "z[7]  Comp x Reverb",
        "z[8]  Reverb x Delay",
        "z[9]  Bright x Warmth",
        "z[10] Saturation x Warmth",
        "z[11] Comp x Saturation",
        "z[12] Width x Reverb",
    ]
    for i, name in enumerate(interactions, start=7):
        val = z_taste[i]
        lines.append(f"{i:<6} {name:<28} {val:>+8.2f}   {make_bar(val, 12)}")

    lines.append(f"{CYAN}--- Group 3: Aggregate Energy Norms (Slots 13..15) ---{RESET}")
    norms = [
        "z[13] Mean Absolute (L1)",
        "z[14] RMS Energy (L2)",
        "z[15] Peak Absolute (L_inf)",
    ]
    for i, name in enumerate(norms, start=13):
        val = z_taste[i]
        lines.append(f"{i:<6} {name:<28} {val:>+8.2f}   {make_bar(val, 12)}")

    lines.append("-" * 72)
    return "\n".join(lines)


def format_xai_card(
    command: str,
    active_dimension: Optional[str],
    before_val: Optional[float],
    after_val: Optional[float],
    explanation_text: str,
    alpha: float = DEFAULT_ALPHA,
) -> str:
    """Format an evidence-grounded XAI explanation card."""
    intent = FEEDBACK_INTENT_CATALOG.get(command)
    target_fx = intent.target_effect if intent else "None"
    target_params = ", ".join(intent.target_parameters) if intent and intent.target_parameters else "None"

    card_border = "+======================================================================+"
    lines = [
        f"\n{BRIGHT}{card_border}{RESET}",
        f"{BRIGHT}|                     XAI DECISION & UPDATE EXPLANATION                |{RESET}",
        f"{BRIGHT}{card_border}{RESET}",
        f"  {CYAN}Trigger Command    :{RESET} {BRIGHT}{command}{RESET}",
        f"  {CYAN}Active Dimension   :{RESET} {active_dimension.capitalize() if active_dimension else 'None'}",
    ]

    if active_dimension and before_val is not None and after_val is not None:
        delta = after_val - before_val
        d_sym = f"+{delta:.2f}" if delta >= 0 else f"{delta:.2f}"
        lines.extend([
            f"  {CYAN}Session Delta      :{RESET} {before_val:.2f} -> {after_val:.2f} ({d_sym})",
            f"  {CYAN}Update Mechanism   :{RESET} Exponential smoothing: P_(t+1) = (1-a)*P_t + a*d (alpha={alpha:.2f})",
            f"  {CYAN}Scope Context      :{RESET} Session-local preference (starts neutral at 0.00; blended into long-term profile upon session satisfaction)",
        ])

    lines.extend([
        f"  {CYAN}Downstream Target  :{RESET} Effect [{target_fx}] | Parameters: [{target_params}]",
        f"  {CYAN}Grounded Rationale :{RESET}",
        f"    \"{explanation_text}\"",
    ])

    # Truthfulness guards
    lines.append(f"  {YELLOW}Truthfulness & Evaluation Bounds:{RESET}")
    if command == "WIDER":
        lines.append(
            f"    {RED}[!] HARDWARE CONSTRAINT: Active pipeline is MONO (channels=1). "
            f"Width preference recorded in taste profile but cannot be rendered by mono DAPN.{RESET}"
        )
    lines.extend([
        f"    * Audio Rendering Status   : {DIM}NOT EXECUTED (PTA Preference Mode){RESET}",
        f"    * Audio Quality Assessment : {DIM}NOT EVALUATED (No perceptual improvement claimed){RESET}",
        f"{BRIGHT}{card_border}{RESET}",
    ])
    return "\n".join(lines)


def format_session_summary_report(
    user_id: str,
    session_id: str,
    instrument: str,
    genre: str,
    start_vector: List[float],
    final_vector: List[float],
    prev_profile: Optional[List[float]],
    events: List[dict],
    merged_long_term: Optional[List[float]] = None,
    termination_reason: str = "SATISFIED",
    saved_profile_path: Optional[str] = None,
    saved_session_path: Optional[str] = None,
) -> str:
    """Format final session completion report distinguishing session vector from persistent profile."""
    lines = [
        banner("SESSION COMPLETED — FINAL XAI SUMMARY", f"Session: {session_id} | User: {user_id}"),
        f"  {CYAN}User ID            :{RESET} {user_id}",
        f"  {CYAN}Session ID         :{RESET} {session_id}",
        f"  {CYAN}Instrument / Genre :{RESET} {instrument} / {genre}",
        f"  {CYAN}Feedback Events    :{RESET} {len(events)} accepted interactions",
        f"  {CYAN}Termination Status :{RESET} {GREEN}{termination_reason}{RESET}",
        "",
        f"{BRIGHT}1. SESSION PREFERENCE TRAJECTORY SUMMARY (Session-Local){RESET}",
        f"{DIM}Session started neutral at 0.00; values reflect user feedback within this session only{RESET}",
        "-" * 72,
        f"{'Dimension':<14} {'Session Start':>14} {'Session Final':>14} {'Session Delta':>14}   {'Status':<12}",
        "-" * 72,
    ]

    for i, label in enumerate(VECTOR_LABELS):
        s_val = start_vector[i]
        f_val = final_vector[i]
        d_val = f_val - s_val
        status = f"{GREEN}Updated{RESET}" if abs(d_val) >= 0.001 else f"{DIM}Preserved{RESET}"
        d_str = f"{d_val:+.2f}" if abs(d_val) >= 0.001 else " 0.00"
        lines.append(
            f"{label.capitalize():<14} {s_val:>14.2f} {f_val:>14.2f} {d_str:>14}   {status}"
        )

    lines.append("-" * 72)

    if prev_profile is not None and merged_long_term is not None:
        lines.extend([
            "",
            f"{BRIGHT}2. LONG-TERM PROFILE ACCUMULATION (Across Sessions){RESET}",
            f"{DIM}Multi-session persistence: untouched dimensions retain prior learning with ZERO decay{RESET}",
            "-" * 88,
            f"{'Dimension':<14} {'Prior Profile':>14} {'Session Final':>14} {'Aggregation':>18} {'New Long-Term':>14} {'Shift':>8}",
            "-" * 88,
        ])
        for i, label in enumerate(VECTOR_LABELS):
            p_val = prev_profile[i]
            s_val = final_vector[i]
            n_val = merged_long_term[i]
            shift = n_val - p_val
            shift_str = f"{shift:+.2f}" if abs(shift) >= 0.001 else " 0.00"
            if abs(s_val) < 0.001:
                action_str = f"{DIM}Preserved (0 decay){RESET}"
            else:
                action_str = f"{GREEN}Blended (feedback){RESET}"
            lines.append(
                f"{label.capitalize():<14} {p_val:>14.2f} {s_val:>14.2f} {action_str:>27} {n_val:>14.2f} {shift_str:>8}"
            )
        lines.append("-" * 88)
    elif merged_long_term is not None:
        lines.extend([
            "",
            f"{BRIGHT}2. INITIAL LONG-TERM PROFILE CREATION{RESET}",
            "-" * 72,
            f"{'Dimension':<14} {'Session Final':>14} {'Saved Long-Term':>16}",
            "-" * 72,
        ])
        for i, label in enumerate(VECTOR_LABELS):
            s_val = final_vector[i]
            m_val = merged_long_term[i]
            lines.append(f"{label.capitalize():<14} {s_val:>14.2f} {m_val:>16.2f}")
        lines.append("-" * 72)

    lines.extend([
        "",
        f"{BRIGHT}CHRONOLOGICAL FEEDBACK & EXPLANATION TRACE{RESET}",
        "-" * 72,
    ])
    if not events:
        lines.append("  (No feedback clicks recorded in this session)")
    else:
        for idx, ev in enumerate(events, 1):
            cmd = ev.get("command", "UNKNOWN")
            dim = ev.get("active_dimension")
            delta_val = ev.get("delta")
            if delta_val and dim in VECTOR_LABELS:
                d_idx = VECTOR_LABELS.index(dim)
                d_amt = delta_val[d_idx]
                d_repr = f"({d_amt:+.2f})"
            else:
                d_repr = ""
            lines.append(f"  {idx:2d}. {CYAN}{cmd:<18}{RESET} -> {dim or 'Terminal':<12} {d_repr}")

    lines.extend([
        "-" * 72,
        "",
        f"{BRIGHT}PERSISTENCE AUDIT{RESET}",
        f"  * User Profile Saved To : {saved_profile_path or 'data/users/'}",
        f"  * Session Record Saved  : {saved_session_path or 'data/sessions/'}",
        f"  * Merge Rule            : Selective dimension preservation (untouched dimensions never decay)",
        "",
        f"{YELLOW}TRUTHFULNESS & FIDELITY DECLARATION{RESET}",
        f"  * Audio rendering status   : {RED}NOT EXECUTED{RESET} (Standalone Terminal Evaluation)",
        f"  * Audio quality evaluation : {RED}NOT EVALUATED{RESET} (Satisfied indicates user feedback completion,",
        f"                               not objective acoustic benchmark verification)",
        "=" * 72,
    ])
    return "\n".join(lines)
