"""
Script to generate a publication-quality 3-page PDF integration specification summary
for M1, M2, M3, and PTA components.
"""

import sys
import types
from pathlib import Path

# Stub PIL to satisfy ReportLab without requiring external Pillow install
pil_mock = types.ModuleType('PIL')
pil_image = types.ModuleType('PIL.Image')
pil_mock.Image = pil_image
sys.modules['PIL'] = pil_mock
sys.modules['PIL.Image'] = pil_image

from reportlab.lib.pagesizes import letter
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, PageBreak, HRFlowable
)
from reportlab.pdfgen import canvas

class NumberedCanvas(canvas.Canvas):
    """Two-pass canvas to dynamically compute and draw total page numbers and headers/footers."""
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._saved_page_states = []

    def showPage(self):
        self._saved_page_states.append(dict(self.__dict__))
        self._startPage()

    def save(self):
        num_pages = len(self._saved_page_states)
        for state in self._saved_page_states:
            self.__dict__.update(state)
            self.draw_page_decorations(num_pages)
            super().showPage()
        super().save()

    def draw_page_decorations(self, page_count):
        self.saveState()
        self.setFont("Helvetica-Bold", 8)
        self.setFillColor(colors.HexColor("#4A5568"))
        
        # Header (pages 2+)
        if self._pageNumber > 1:
            self.drawString(54, letter[1] - 34, "Collaborative Audio FX Synthesis — Component Interfaces & PTA Contract")
            self.setFont("Helvetica", 8)
            self.drawRightString(letter[0] - 54, letter[1] - 34, "M1 • M2 • M3 • PTA")
            self.setStrokeColor(colors.HexColor("#CBD5E0"))
            self.setLineWidth(0.75)
            self.line(54, letter[1] - 40, letter[0] - 54, letter[1] - 40)
        
        # Footer
        self.setFont("Helvetica", 7.5)
        self.setFillColor(colors.HexColor("#718096"))
        page_str = f"Page {self._pageNumber} of {page_count}"
        self.drawRightString(letter[0] - 54, 28, page_str)
        self.drawString(54, 28, "RESEARCH PROJECT INTEGRATION SPECIFICATION — CONFIDENTIAL & PROPRIETARY")
        self.setStrokeColor(colors.HexColor("#E2E8F0"))
        self.setLineWidth(0.5)
        self.line(54, 38, letter[0] - 54, 38)
        
        self.restoreState()


def build_pdf(filename="M1_M2_M3_PTA_Integration_Summary.pdf"):
    doc = SimpleDocTemplate(
        filename,
        pagesize=letter,
        leftMargin=54,
        rightMargin=54,
        topMargin=46,
        bottomMargin=46
    )

    styles = getSampleStyleSheet()

    # Palette
    primary = colors.HexColor("#1A365D")     # Deep navy
    secondary = colors.HexColor("#2B6CB0")   # Slate blue
    dark_text = colors.HexColor("#2D3748")   # Charcoal
    light_bg = colors.HexColor("#F7FAFC")    # Warm white
    border_col = colors.HexColor("#CBD5E0")  # Soft gray
    box_bg = colors.HexColor("#EDF2F7")      # Tinted gray
    code_bg = colors.HexColor("#1A202C")     # Dark code background

    # Typography styles
    title_style = ParagraphStyle(
        'DocTitle',
        parent=styles['Normal'],
        fontName='Helvetica-Bold',
        fontSize=18,
        leading=22,
        textColor=primary,
        spaceAfter=4
    )

    subtitle_style = ParagraphStyle(
        'DocSubtitle',
        parent=styles['Normal'],
        fontName='Helvetica',
        fontSize=8.5,
        leading=12,
        textColor=colors.HexColor("#4A5568"),
        spaceAfter=10
    )

    h1_style = ParagraphStyle(
        'Heading1',
        parent=styles['Normal'],
        fontName='Helvetica-Bold',
        fontSize=11.5,
        leading=15,
        textColor=primary,
        spaceBefore=8,
        spaceAfter=4,
        keepWithNext=True
    )

    h2_style = ParagraphStyle(
        'Heading2',
        parent=styles['Normal'],
        fontName='Helvetica-Bold',
        fontSize=9.5,
        leading=13,
        textColor=secondary,
        spaceBefore=6,
        spaceAfter=3,
        keepWithNext=True
    )

    body_style = ParagraphStyle(
        'Body',
        parent=styles['Normal'],
        fontName='Helvetica',
        fontSize=8,
        leading=11.5,
        textColor=dark_text,
        spaceAfter=4
    )

    bullet_style = ParagraphStyle(
        'Bullet',
        parent=styles['Normal'],
        fontName='Helvetica',
        fontSize=8,
        leading=11.5,
        textColor=dark_text,
        leftIndent=10,
        spaceAfter=2
    )

    code_light_style = ParagraphStyle(
        'CodeLight',
        parent=styles['Normal'],
        fontName='Courier',
        fontSize=6.8,
        leading=9,
        textColor=colors.HexColor("#1A202C")
    )

    code_dark_style = ParagraphStyle(
        'CodeDark',
        parent=styles['Normal'],
        fontName='Courier',
        fontSize=7,
        leading=9.5,
        textColor=colors.HexColor("#F7FAFC")
    )

    th_style = ParagraphStyle(
        'TH',
        parent=styles['Normal'],
        fontName='Helvetica-Bold',
        fontSize=7.2,
        leading=9.5,
        textColor=colors.white
    )

    td_style = ParagraphStyle(
        'TD',
        parent=styles['Normal'],
        fontName='Helvetica',
        fontSize=7.2,
        leading=9.5,
        textColor=dark_text
    )

    td_bold = ParagraphStyle(
        'TDBold',
        parent=styles['Normal'],
        fontName='Helvetica-Bold',
        fontSize=7.2,
        leading=9.5,
        textColor=dark_text
    )

    status_impl = ParagraphStyle(
        'StatusImpl',
        parent=styles['Normal'],
        fontName='Helvetica-Bold',
        fontSize=7,
        leading=9,
        textColor=colors.HexColor("#22543D")
    )

    status_plan = ParagraphStyle(
        'StatusPlan',
        parent=styles['Normal'],
        fontName='Helvetica-Bold',
        fontSize=7,
        leading=9,
        textColor=colors.HexColor("#C05621")
    )

    story = []

    # =========================================================================
    # PAGE 1: TITLE, STATUS MATRIX & M1 AUDIO ANALYSIS
    # =========================================================================
    story.append(Paragraph("System Architecture & Component Interface Specification", title_style))
    story.append(Paragraph(
        "<b>Project Context:</b> Collaborative Audio Effects Chain Synthesis via Differentiable RL & Human-in-the-Loop Personalisation<br/>"
        "<b>Purpose:</b> Cross-component interface contracts, schemas, tensor formats, and PTA-to-M3 integration guidelines.",
        subtitle_style
    ))
    story.append(HRFlowable(width="100%", thickness=1.2, color=primary, spaceAfter=8))

    # Section 1: Matrix
    story.append(Paragraph("1. System Architecture & Status Matrix", h1_style))
    story.append(Paragraph(
        "This specification documents the current deliverables, tensor specifications, and interchange formats between <b>M1</b> (Analysis), <b>M2</b> (Inverse Processing), <b>M3</b> (RL FX Chain Generator), and <b>PTA</b> (Personalized Taste Adapter & XAI).",
        body_style
    ))

    matrix_rows = [
        [Paragraph("Module", th_style), Paragraph("Component Responsibility", th_style), Paragraph("Implementation Status", th_style), Paragraph("Key Deliverables & Code Pointers", th_style)],
        [
            Paragraph("<b>M1</b>", td_bold),
            Paragraph("Audio Analysis & Feature Extraction", td_style),
            Paragraph("IMPLEMENTED", status_impl),
            Paragraph("<code>examples/analyze_audio.py</code> (Pure PyTorch acoustic, temporal, dynamic, and spectral metrics extraction).", td_style)
        ],
        [
            Paragraph("<b>M2</b>", td_bold),
            Paragraph("Audio Understanding & Inverse FX", td_style),
            Paragraph("IMPLEMENTED & TRAINED", status_impl),
            Paragraph("<code>models/inverse_fx_net.py</code> (ResNet backbone + 7-effect classifier + 29-parameter regression + DAPN core).", td_style)
        ],
        [
            Paragraph("<b>M3</b>", td_bold),
            Paragraph("RL-Based FX-Chain Generation", td_style),
            Paragraph("PLANNED (Env Ready)", status_plan),
            Paragraph("<code>dapn/dapn.py</code> (Differentiable execution engine & MR-STFT loss ready; PPO/GAE agent loop in progress).", td_style)
        ],
        [
            Paragraph("<b>PTA</b>", td_bold),
            Paragraph("Personalized Taste Adapter & XAI", td_style),
            Paragraph("IN INTEGRATION PREP", status_plan),
            Paragraph("Preference vector conditioning, reward guidance shaping, and explainable constraints for M3 synthesis.", td_style)
        ]
    ]
    t_mat = Table(matrix_rows, colWidths=[42, 140, 104, 218])
    t_mat.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), primary),
        ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
        ('VALIGN', (0, 0), (-1, -1), 'TOP'),
        ('GRID', (0, 0), (-1, -1), 0.5, border_col),
        ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, light_bg]),
        ('TOPPADDING', (0, 0), (-1, -1), 3.5),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 3.5),
    ]))
    story.append(t_mat)
    story.append(Spacer(1, 8))

    # Section 2: M1
    story.append(Paragraph("2. M1 — Audio Analysis / Feature Extraction Interface", h1_style))
    story.append(Paragraph(
        "<b>Component Scope:</b> Deterministic acoustic extraction across <b>Dry un-processed audio (x)</b>, <b>Target Reference audio (y_ref)</b>, and <b>Generated wet audio (y_gen)</b>. Outputs a structured feature dictionary or PyTorch feature vectors.",
        body_style
    ))
    story.append(Paragraph("• <b>Audio Input:</b> WAV tensor <code>(C, T)</code> or batch <code>(B, C, T)</code> at 44.1 kHz, single-channel mono or stereo downmixed.", bullet_style))
    story.append(Paragraph("• <b>Front-end Delta:</b> In addition to absolute metrics, M2 computes differential Log-Mel spectrograms: <code>Mel_delta = Mel(wet) - Mel(dry)</code>.", bullet_style))

    m1_table = [
        [Paragraph("Feature Metric", th_style), Paragraph("Class", th_style), Paragraph("Physical Unit", th_style), Paragraph("Raw Range", th_style), Paragraph("Normalized ML Scale [0, 1]", th_style)],
        [Paragraph("<code>peak_linear</code> / <code>peak_dbFS</code>", td_bold), Paragraph("Dynamics", td_style), Paragraph("Linear / dBFS", td_style), Paragraph("[0.0, 1.0] / [-inf, 0]", td_style), Paragraph("<code>(dBFS + 80.0) / 80.0</code>", td_style)],
        [Paragraph("<code>rms_linear</code> / <code>rms_dbFS</code>", td_bold), Paragraph("Dynamics", td_style), Paragraph("Linear / dBFS", td_style), Paragraph("[0.0, 1.0] / [-inf, 0]", td_style), Paragraph("<code>(dBFS + 80.0) / 80.0</code>", td_style)],
        [Paragraph("<code>crest_factor_db</code>", td_bold), Paragraph("Dynamics", td_style), Paragraph("dB", td_style), Paragraph("[0.0, 40.0+]", td_style), Paragraph("<code>crest_db / 40.0</code>", td_style)],
        [Paragraph("<code>zero_crossing_rate</code>", td_bold), Paragraph("Temporal", td_style), Paragraph("rate per sample", td_style), Paragraph("[0.0, 1.0]", td_style), Paragraph("Direct normalized [0.0, 1.0]", td_style)],
        [Paragraph("<code>spectral_centroid_hz</code>", td_bold), Paragraph("Spectral", td_style), Paragraph("Hz", td_style), Paragraph("[20.0, 22050.0]", td_style), Paragraph("Log10 min-max or normalized [0, 1]", td_style)],
        [Paragraph("<code>spectral_bandwidth_hz</code>", td_bold), Paragraph("Spectral", td_style), Paragraph("Hz", td_style), Paragraph("[0.0, 11025.0]", td_style), Paragraph("<code>bandwidth / 11025.0</code>", td_style)],
        [Paragraph("<code>spectral_rolloff_85pct_hz</code>", td_bold), Paragraph("Spectral", td_style), Paragraph("Hz", td_style), Paragraph("[20.0, 22050.0]", td_style), Paragraph("<code>rolloff / 22050.0</code>", td_style)],
        [Paragraph("<code>spectral_flatness</code>", td_bold), Paragraph("Spectral", td_style), Paragraph("Wiener ratio", td_style), Paragraph("[0.0, 1.0]", td_style), Paragraph("Direct normalized [0.0, 1.0]", td_style)],
        [Paragraph("<code>low_sub_bass_under_250hz_pct</code>", td_bold), Paragraph("Band Energy", td_style), Paragraph("% of energy", td_style), Paragraph("[0.0, 100.0] %", td_style), Paragraph("<code>pct / 100.0</code>", td_style)],
        [Paragraph("<code>midrange_250_to_4000hz_pct</code>", td_bold), Paragraph("Band Energy", td_style), Paragraph("% of energy", td_style), Paragraph("[0.0, 100.0] %", td_style), Paragraph("<code>pct / 100.0</code>", td_style)],
        [Paragraph("<code>high_presence_above_4000hz_pct</code>", td_bold), Paragraph("Band Energy", td_style), Paragraph("% of energy", td_style), Paragraph("[0.0, 100.0] %", td_style), Paragraph("<code>pct / 100.0</code>", td_style)],
    ]
    t_m1 = Table(m1_table, colWidths=[134, 66, 70, 96, 138])
    t_m1.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), secondary),
        ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
        ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
        ('GRID', (0, 0), (-1, -1), 0.5, border_col),
        ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, light_bg]),
        ('TOPPADDING', (0, 0), (-1, -1), 2.5),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 2.5),
    ]))
    story.append(t_m1)

    story.append(PageBreak())

    # =========================================================================
    # PAGE 2: M2 AUDIO UNDERSTANDING / INVERSE PROCESSING
    # =========================================================================
    story.append(Paragraph("3. M2 — Audio Understanding / Inverse Processing Interface", h1_style))
    story.append(Paragraph(
        "<b>Model Architecture (<code>InverseFXNet</code>):</b> End-to-end deep convolutional neural network mapping paired dry and wet audio waveforms directly to active effect presence probabilities and continuous DSP physical parameters.",
        body_style
    ))
    story.append(Paragraph(
        "• <b>Input Features:</b> 3-channel Log-Mel Spectrogram (Dry, Wet, Delta) computed via <code>T.MelSpectrogram</code> (n_fft=1024, hop=512, n_mels=64) producing shape <code>(B, 3, 64, 173)</code> for 2.0s audio segments.<br/>"
        "• <b>Backbone:</b> 4-stage 2D Residual Convolutional Network (channels: 3 -> 32 -> 64 -> 128 -> 256) with BatchNorm and LeakyReLU(0.2).<br/>"
        "• <b>Dense Bottleneck:</b> Concatenated Global Average Pooling (GAP) + Global Max Pooling (GMP) -> 512 dimensions -> Linear -> <b>256-dimensional dense embedding (<code>emb</code>)</b>.<br/>"
        "• <b>Multi-Task Heads:</b><br/>"
        "&nbsp;&nbsp;&nbsp;&nbsp;1. <b>Head A (Effect Presence):</b> Linear(256 -> 7) logits for multi-label classification across all 7 supported effects.<br/>"
        "&nbsp;&nbsp;&nbsp;&nbsp;2. <b>Head B (Parameter Regression):</b> Linear(256 -> 29) + Sigmoid yielding continuous normalized parameters strictly in <code>[0.0, 1.0]</code>.",
        bullet_style
    ))

    story.append(Spacer(1, 4))
    story.append(Paragraph("Supported Effects & Parameter Ranges (Canonical 29 Continuous Knobs):", h2_style))

    fx_table = [
        [Paragraph("Effect Name", th_style), Paragraph("Continuous Parameters", th_style), Paragraph("Physical Unit", th_style), Paragraph("Physical DSP Range", th_style), Paragraph("Normalized Range", th_style)],
        [Paragraph("<b>EQ</b>", td_bold), Paragraph("frequency<br/>gain<br/>q", td_style), Paragraph("Hz<br/>dB<br/>dim", td_style), Paragraph("[20.0, 20000.0]<br/>[-12.0, 12.0]<br/>[0.1, 10.0]", td_style), Paragraph("[0.0, 1.0]<br/>[0.0, 1.0]<br/>[0.0, 1.0]", td_style)],
        [Paragraph("<b>Compressor</b>", td_bold), Paragraph("threshold<br/>ratio<br/>attack<br/>release<br/>makeup_gain", td_style), Paragraph("dB<br/>dim<br/>ms<br/>ms<br/>dB", td_style), Paragraph("[-60.0, 0.0]<br/>[1.0, 20.0]<br/>[0.1, 100.0]<br/>[10.0, 1000.0]<br/>[0.0, 12.0]", td_style), Paragraph("[0.0, 1.0]<br/>[0.0, 1.0]<br/>[0.0, 1.0]<br/>[0.0, 1.0]<br/>[0.0, 1.0]", td_style)],
        [Paragraph("<b>Delay</b>", td_bold), Paragraph("delay_time<br/>feedback<br/>wet_mix<br/>dry_mix", td_style), Paragraph("ms<br/>dim<br/>dim<br/>dim", td_style), Paragraph("[1.0, 2000.0]<br/>[0.0, 0.95]<br/>[0.0, 1.0]<br/>[0.0, 1.0]", td_style), Paragraph("[0.0, 1.0]<br/>[0.0, 1.0]<br/>[0.0, 1.0]<br/>[0.0, 1.0]", td_style)],
        [Paragraph("<b>Reverb</b>", td_bold), Paragraph("room_size<br/>damping<br/>wet_mix<br/>dry_mix<br/>decay", td_style), Paragraph("dim<br/>dim<br/>dim<br/>dim<br/>s", td_style), Paragraph("[0.0, 1.0]<br/>[0.0, 1.0]<br/>[0.0, 1.0]<br/>[0.0, 1.0]<br/>[0.1, 10.0]", td_style), Paragraph("[0.0, 1.0]<br/>[0.0, 1.0]<br/>[0.0, 1.0]<br/>[0.0, 1.0]<br/>[0.0, 1.0]", td_style)],
        [Paragraph("<b>Distortion</b>", td_bold), Paragraph("drive<br/>tone<br/>mix<br/>output_gain", td_style), Paragraph("dim<br/>dim<br/>dim<br/>dB", td_style), Paragraph("[0.0, 1.0]<br/>[0.0, 1.0]<br/>[0.0, 1.0]<br/>[-12.0, 6.0]", td_style), Paragraph("[0.0, 1.0]<br/>[0.0, 1.0]<br/>[0.0, 1.0]<br/>[0.0, 1.0]", td_style)],
        [Paragraph("<b>Chorus</b>", td_bold), Paragraph("rate<br/>depth<br/>delay<br/>mix", td_style), Paragraph("Hz<br/>dim<br/>ms<br/>dim", td_style), Paragraph("[0.1, 10.0]<br/>[0.0, 1.0]<br/>[1.0, 50.0]<br/>[0.0, 1.0]", td_style), Paragraph("[0.0, 1.0]<br/>[0.0, 1.0]<br/>[0.0, 1.0]<br/>[0.0, 1.0]", td_style)],
        [Paragraph("<b>Flanger</b>", td_bold), Paragraph("rate<br/>depth<br/>feedback<br/>mix", td_style), Paragraph("Hz<br/>dim<br/>dim<br/>dim", td_style), Paragraph("[0.1, 10.0]<br/>[0.0, 1.0]<br/>[0.0, 0.95]<br/>[0.0, 1.0]", td_style), Paragraph("[0.0, 1.0]<br/>[0.0, 1.0]<br/>[0.0, 1.0]<br/>[0.0, 1.0]", td_style)],
        [Paragraph("<b>STOP</b>", td_bold), Paragraph("Reserved RL Terminal Action", td_style), Paragraph("none", td_style), Paragraph("Terminal token for M3 agent", td_style), Paragraph("No parameters (rejected by DAPN)", td_style)]
    ]
    t_fx = Table(fx_table, colWidths=[74, 130, 80, 110, 110])
    t_fx.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), primary),
        ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
        ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
        ('GRID', (0, 0), (-1, -1), 0.5, border_col),
        ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, light_bg]),
        ('TOPPADDING', (0, 0), (-1, -1), 2.2),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 2.2),
    ]))
    story.append(t_fx)
    story.append(Spacer(1, 4))

    story.append(Paragraph("<b>Parameter Mapping Formulation:</b> Executed via <code>common.parameter_mapper.ParameterMapper</code>:", body_style))
    story.append(Paragraph("<code>physical = min_phys + clamped_norm * (max_phys - min_phys)</code> (Preserves autograd gradients for all values in [0, 1]).", bullet_style))

    story.append(PageBreak())

    # =========================================================================
    # PAGE 3: M3 RL-AGENT & PTA INTEGRATION CONTRACT
    # =========================================================================
    story.append(Paragraph("4. M3 — RL-Based FX-Chain Generation (GRRL-Agent)", h1_style))
    story.append(Paragraph(
        "<b>Component Status:</b> <b>Planned / Environment Ready</b>. The DAPN differentiable environment, 78/78 unit tests, Multi-Resolution STFT differentiable loss, and parameter mapping are fully operational. The PPO actor-critic loop is currently in active development.",
        body_style
    ))

    story.append(Paragraph("Formal MDP Specification for Chain Synthesis:", h2_style))
    story.append(Paragraph(
        "• <b>State Space (s_t):</b> Concatenation of (1) Acoustic feature embeddings <code>e_audio</code> from M1/M2 representing current rendered audio y_t, dry stem x, and target reference y_ref; (2) Chain step history encoding applied effects; and (3) PTA user taste vector <code>z_taste</code>.<br/>"
        "• <b>Action Space (a_t):</b> Hybrid discrete-continuous action. Discrete action chooses effect type <code>a_disc in {EQ, Compressor, Delay, Reverb, Distortion, Chorus, Flanger, STOP}</code>. Continuous head outputs normalized parameter knobs <code>a_cont in [0, 1]^K</code> for the selected effect.<br/>"
        "• <b>Multi-Objective Reward:</b> Total reward balances objective acoustic fidelity, personalized human taste, and chain brevity:",
        bullet_style
    ))

    reward_box = [
        [Paragraph(
            "<b>Total Reward:</b>  R_total = R_acoustic + lambda_pta * R_pta - beta * StepPenalty<br/>"
            "• <b>R_acoustic:</b> - MultiResolutionSTFTLoss(y_t, y_ref) computed across 512, 1024, 2048 FFT windows.<br/>"
            "• <b>R_pta:</b> Personalized scalar preference reward signal in [-1.0, 1.0] supplied by PTA.<br/>"
            "• <b>StepPenalty:</b> Regularization penalty encouraging compact, non-redundant effect chains.",
            code_light_style
        )]
    ]
    t_rew = Table(reward_box, colWidths=[504])
    t_rew.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, -1), box_bg),
        ('BOX', (0, 0), (-1, -1), 1, primary),
        ('PADDING', (0, 0), (-1, -1), 6),
    ]))
    story.append(t_rew)
    story.append(Spacer(1, 6))

    # Section 5: PTA Contract
    story.append(Paragraph("5. PTA-to-M3 Interface Contract & Guidance Modalities", h1_style))
    story.append(Paragraph(
        "The PTA component can interface with M3 through three mutually compatible guidance modes:",
        body_style
    ))

    pta_modes = [
        [Paragraph("Guidance Mode", th_style), Paragraph("Interface Payload / Structure", th_style), Paragraph("M3 Consumption & Behavior", th_style)],
        [
            Paragraph("<b>1. Reward Guidance</b><br/>(Preference Shaping)", td_bold),
            Paragraph("Scalar reward <code>R_pta in [-1.0, 1.0]</code> with optional confidence weight and explanation string.", td_style),
            Paragraph("Injected directly into PPO policy gradient and GAE advantage calculations.", td_style)
        ],
        [
            Paragraph("<b>2. Conditioning Vector</b><br/>(Taste Representation)", td_bold),
            Paragraph("Dense embedding vector <code>z_taste in R^D</code> (e.g. D=16 or 32, normalized in [-1.0, 1.0]).", td_style),
            Paragraph("Concatenated to policy state input: <code>s_t = [e_audio, e_chain, z_taste]</code>.", td_style)
        ],
        [
            Paragraph("<b>3. Constraint Masking</b><br/>(Rules & Priors)", td_bold),
            Paragraph("JSON dictionary defining prohibited effect types and parameter bounding intervals.", td_style),
            Paragraph("Masks discrete action logits in actor head and clamps continuous parameter actions.", td_style)
        ]
    ]
    t_pta = Table(pta_modes, colWidths=[114, 190, 200])
    t_pta.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), primary),
        ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
        ('VALIGN', (0, 0), (-1, -1), 'TOP'),
        ('GRID', (0, 0), (-1, -1), 0.5, border_col),
        ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, light_bg]),
        ('TOPPADDING', (0, 0), (-1, -1), 3),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 3),
    ]))
    story.append(t_pta)
    story.append(Spacer(1, 6))

    story.append(Paragraph("Canonical FX-Chain JSON Interchange Schema (DAPN Contract):", h2_style))
    schema_payload = [
        [Paragraph(
"""{
  "session_id": "sess_20261017_001",
  "sample_id": "sample_00482",
  "chain": [
    {
      "effect": "EQ",
      "parameters": {"frequency": 0.450, "gain": 0.650, "q": 0.500}
    },
    {
      "effect": "Compressor",
      "parameters": {"threshold": 0.500, "ratio": 0.400, "attack": 0.200, "release": 0.350, "makeup_gain": 0.300}
    },
    {
      "effect": "Reverb",
      "parameters": {"room_size": 0.550, "damping": 0.300, "wet_mix": 0.350, "dry_mix": 0.800, "decay": 0.400}
    }
  ]
}""", code_light_style)]
    ]
    t_sch = Table(schema_payload, colWidths=[504])
    t_sch.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, -1), colors.HexColor("#F7FAFC")),
        ('BOX', (0, 0), (-1, -1), 0.5, border_col),
        ('PADDING', (0, 0), (-1, -1), 4),
    ]))
    story.append(t_sch)
    story.append(Spacer(1, 4))

    story.append(Paragraph(
        "<b>Feedback Event Linking:</b> Human evaluation events occur post-generation (pairwise choice or 5-point rating). Feedback submissions pass back the exact <code>sample_id</code>, linking user ratings directly to the synthesized FX-chain and audio tensor checksum.",
        body_style
    ))

    # Build PDF
    doc.build(story, canvasmaker=NumberedCanvas)
    print(f"PDF successfully generated: {filename}")

if __name__ == "__main__":
    out_pdf = "M1_M2_M3_PTA_Integration_Summary.pdf"
    if len(sys.argv) > 1:
        out_pdf = sys.argv[1]
    build_pdf(out_pdf)
