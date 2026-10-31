"""All generator assumptions live here so they can be stated, varied and reported.

Every number below is an ASSUMPTION about how AI-generated / AI-edited documents differ from genuine ones.
None of them has been validated against real generative-model output. The `difficulty` presets exist so
you can show how results change when those assumptions become less favourable to the detector.
"""
from dataclasses import dataclass, asdict, field

LABELS = ["genuine", "ai_generated", "ai_edited"]


@dataclass
class GenConfig:
    n_cases: int = 1800
    seed: int = 42
    difficulty: str = "medium"            # easy | medium | hard | adversarial
    p_genuine: float = 0.60
    p_ai_generated: float = 0.15
    p_ai_edited: float = 0.25
    p_statement: float = 0.60             # remainder are payslips
    n_identities: int = 120               # signature identities (also the unit for leakage-free splits)

    # ---- overlap knobs (set by difficulty preset in `with_difficulty`) ----
    p_genuine_rescanned: float = 0.25     # genuine docs that were printed/re-scanned (noisy, re-saved, odd producer)
    p_genuine_arith_flaw: float = 0.05    # genuine docs with a real-world rounding/typo flaw
    p_edit_clean_font: float = 0.35       # AI/editor patch uses the same font (hard to see)
    p_edit_recalc: float = 0.40           # editor also recalculates the dependent numbers
    p_edit_benign_tool: float = 0.35      # edit done in a tool that looks like a normal bank/scanner tool
    p_edit_same_quality: float = 0.20     # re-save at same JPEG quality (weaker double-compression trace)
    p_edit_no_halo: float = 0.50          # no blur/seam around the patch
    p_genuine_odd_meta: float = 0.06      # genuine docs whose metadata looks odd (re-saved in an editor, stripped by a portal)
    p_edit_meta_reset: float = 0.30       # edit whose metadata trail was reset/cleaned
    p_edit_laundered: float = 0.15        # edited page is then printed/re-scanned, blurring the traces
    p_gen_arith_error: float = 0.65       # generated docs with arithmetic that does not reconcile
    p_gen_benford_off: float = 0.70       # generated docs with non-Benford amounts
    p_gen_text_layer: float = 0.30        # generated docs that still carry a text layer
    p_gen_meta_clean: float = 0.30        # generated docs with believable bank-system metadata
    gen_wobble_range: tuple = (0.2, 1.4)  # glyph jitter (px) of generated pages; low end ~ undetectable
    p_sig_edit_swapped: float = 0.45      # edited docs whose signature was replaced
    sig_forger_sigma_range: tuple = (2.0, 8.0)  # how far an AI-made signature drifts from the identity

    def with_difficulty(self, level: str) -> "GenConfig":
        c = GenConfig(**asdict(self)); c.difficulty = level
        presets = {
            "easy":   dict(p_genuine_rescanned=0.10, p_genuine_arith_flaw=0.02, p_edit_clean_font=0.15,
                           p_edit_recalc=0.20, p_edit_benign_tool=0.15, p_edit_same_quality=0.05,
                           p_edit_no_halo=0.25, p_edit_laundered=0.05, p_genuine_odd_meta=0.03, p_edit_meta_reset=0.15, p_gen_arith_error=0.85, p_gen_benford_off=0.85,
                           p_gen_text_layer=0.15, p_gen_meta_clean=0.10, gen_wobble_range=(0.8, 1.8),
                           p_sig_edit_swapped=0.55, sig_forger_sigma_range=(3.0, 9.0)),
            "medium": {},
            "hard":   dict(p_genuine_rescanned=0.40, p_genuine_arith_flaw=0.10, p_edit_clean_font=0.60,
                           p_edit_recalc=0.65, p_edit_benign_tool=0.55, p_edit_same_quality=0.40,
                           p_edit_no_halo=0.75, p_edit_laundered=0.30, p_genuine_odd_meta=0.12, p_edit_meta_reset=0.50, p_gen_arith_error=0.40, p_gen_benford_off=0.45,
                           p_gen_text_layer=0.50, p_gen_meta_clean=0.55, gen_wobble_range=(0.1, 0.9),
                           p_sig_edit_swapped=0.35, sig_forger_sigma_range=(1.2, 5.5)),
            "adversarial": dict(p_genuine_rescanned=0.60, p_genuine_arith_flaw=0.20, p_edit_clean_font=0.85, p_edit_recalc=0.90,
                           p_edit_benign_tool=0.80, p_edit_same_quality=0.70, p_edit_no_halo=0.90, p_edit_laundered=0.50,
                           p_gen_arith_error=0.15, p_gen_benford_off=0.20, p_gen_text_layer=0.80, p_gen_meta_clean=0.85,
                           gen_wobble_range=(0.0, 0.5), p_sig_edit_swapped=0.20, sig_forger_sigma_range=(0.8, 3.0)),
        }
        for k, v in presets[level].items():
            setattr(c, k, v)
        return c

    def to_dict(self):
        return asdict(self)
