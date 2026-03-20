"""
Expression Renderer — Component 7 (Phase 1b Prototype)
=======================================================
Converts the standing wave Ψ into fluent natural language.

Architecture constraints
------------------------
This component:
  - Has NO access to the manifold (Components 1–5)
  - Receives ONLY the standing wave Ψ from the Resonance Layer
  - Produces language output
  - Is the ONLY component that knows anything about language

Three-stage pipeline (spec §2 Component 7)
------------------------------------------

  STAGE 1 — SEGMENTATION
    Identify natural segments from the wave structure.
    A boundary occurs where Ψ has a local minimum.
    Not fixed length.  Not token-aligned.
    Segments emerge from the meaning structure itself.

  STAGE 2 — RESONANCE MATCHING
    For each segment Ψᵢ, find the linguistic expression E that minimises
        resonance_distance(Ψᵢ, semantic_wave(E))
    Constraint satisfaction, not token prediction.

  STAGE 3 — FLOW PRESERVATION
    Adjust expressions so that the rendering faithfully reflects the
    flow dynamics:
      fast flow    → short, direct sentences
      slow flow    → long, complex sentences
      loop         → anaphora, repetition
      sharp turn   → paragraph break / transitional phrase
      uncertainty  → hedged language
      crystallised → declarative, confident language
"""

from __future__ import annotations

import textwrap
from dataclasses import dataclass, field
from typing import List, Optional

import numpy as np

from .wave import StandingWave, WaveSegment, WavePoint
from .matcher import ResonanceMatcher, MatchResult

# Optional: Phase 10 Geometric Grammar Engine
try:
    from src.phase10.grammar.grammar_renderer import GrammarRenderer as _GrammarRenderer
    _HAS_GRAMMAR = True
except ImportError:
    _HAS_GRAMMAR = False


# ── Output type ────────────────────────────────────────────────────────────


@dataclass
class RenderedOutput:
    """
    The final output of the Expression Renderer.

    Attributes
    ----------
    text            : the rendered natural language string
    segments        : the wave segments that were rendered
    matches         : the resonance match result for each segment
    confidence      : overall rendering confidence (mean resonance score)
    flow_preserved  : whether flow dynamics were successfully preserved
    diagnostics     : per-segment diagnostic information
    """
    text:            str
    segments:        List[WaveSegment]
    matches:         List[MatchResult]
    confidence:      float
    flow_preserved:  bool
    diagnostics:     List[dict] = field(default_factory=list)

    def __repr__(self) -> str:
        return (
            f"RenderedOutput(\n"
            f"  confidence={self.confidence:.3f}\n"
            f"  n_segments={len(self.segments)}\n"
            f"  flow_preserved={self.flow_preserved}\n"
            f"  text={self.text[:120]!r}{'...' if len(self.text) > 120 else ''}\n"
            f")"
        )


class ExpressionRenderer:
    """
    Converts a standing wave Ψ into fluent natural language.

    This component knows nothing about the manifold.  It receives Ψ,
    segments it, matches segments to expressions, and assembles output.

    Language agnosticism
    --------------------
    The segmentation and matching are language-agnostic.
    Only the vocabulary in ResonanceMatcher is language-specific.
    To render into a different language, swap the matcher's vocabulary.
    """

    def __init__(self, dim: int = 104, use_grammar: bool = True) -> None:
        self.matcher = ResonanceMatcher(dim=dim)
        self.dim     = dim
        self._grammar = None
        if use_grammar and _HAS_GRAMMAR:
            self._grammar = _GrammarRenderer()

    # ------------------------------------------------------------------ #
    # Public API                                                            #
    # ------------------------------------------------------------------ #

    def render(
        self,
        wave: StandingWave,
        min_segments: int = 1,
        max_segments: int = 8,
        amplitude_threshold: float = 0.05,
    ) -> RenderedOutput:
        """
        Render a standing wave into natural language.

        Parameters
        ----------
        wave               : the standing wave Ψ to render
        min_segments       : minimum number of output segments
        max_segments       : maximum number of output segments
        amplitude_threshold: wave points below this normalised amplitude
                             are treated as background noise

        Returns
        -------
        RenderedOutput
        """
        # ── Stage 1: Segmentation ─────────────────────────────────────
        segments = self._segment(wave, min_segments, max_segments, amplitude_threshold)

        # ── Stage 2: Resonance matching ───────────────────────────────
        matches = self.matcher.match_all(segments)

        # ── Stage 3: Flow preservation ────────────────────────────────
        sentences, diagnostics = self._apply_flow_preservation(segments, matches)

        # ── Stage 3b: Grammar-enhanced rendering (Phase 10) ───────────
        if self._grammar is not None:
            sentences, diagnostics = self._grammar_enhance(
                segments, matches, sentences, diagnostics
            )

        # ── Assemble final text ───────────────────────────────────────
        text = self._assemble(sentences, segments)

        confidence = float(np.mean([m.resonance_score for m in matches])) if matches else 0.0
        flow_preserved = all(d.get("flow_preserved", True) for d in diagnostics)

        return RenderedOutput(
            text=text,
            segments=segments,
            matches=matches,
            confidence=confidence,
            flow_preserved=flow_preserved,
            diagnostics=diagnostics,
        )

    # ------------------------------------------------------------------ #
    # Stage 1 — Segmentation                                               #
    # ------------------------------------------------------------------ #

    def _segment(
        self,
        wave: StandingWave,
        min_segs: int,
        max_segs: int,
        threshold: float,
    ) -> List[WaveSegment]:
        """
        Identify meaningful segments from the wave's amplitude structure.

        Algorithm
        ---------
        1. Filter out below-threshold noise points.
        2. Sort remaining points by their τ-axis (causal time order).
        3. Detect local amplitude minima as segment boundaries.
        4. Assign each run of points between minima to a segment.
        5. Merge very small segments into neighbours.
        6. Split very large segments if needed.
        7. Compute segment-level statistics (coherence, uncertainty, flow_speed).
        """
        # Step 1: Filter noise
        norm_amps = wave.normalised_amplitudes
        active_points = [
            p for p, a in zip(wave.points, norm_amps) if a >= threshold
        ]
        if not active_points:
            # Fallback: use all points
            active_points = wave.points if wave.points else []

        if not active_points:
            return []

        # Step 2: Sort by τ (causal time → natural ordering)
        active_points_sorted = sorted(active_points, key=lambda p: p.tau)

        # Step 3: Detect boundaries via local amplitude minima
        amps = np.array([p.amplitude for p in active_points_sorted])
        boundaries = self._find_boundaries(amps, min_segs, max_segs)

        # Step 4: Build segments from runs
        raw_segments: List[List[WavePoint]] = []
        prev = 0
        for b in boundaries:
            raw_segments.append(active_points_sorted[prev:b])
            prev = b
        raw_segments.append(active_points_sorted[prev:])
        raw_segments = [s for s in raw_segments if s]

        # Step 5: Compute segment statistics and build WaveSegment objects
        segments: List[WaveSegment] = []
        for idx, pts in enumerate(raw_segments):
            seg = self._build_segment(pts, idx, wave)
            segments.append(seg)

        return segments

    def _find_boundaries(
        self,
        amps: np.ndarray,
        min_segs: int,
        max_segs: int,
    ) -> List[int]:
        """
        Find indices where segments should be split.

        Uses local minima detection with a minimum gap constraint.
        """
        n = len(amps)
        if n <= 1:
            return []

        # Find local minima (amplitude dips between peaks)
        minima = []
        for i in range(1, n - 1):
            if amps[i] < amps[i - 1] and amps[i] < amps[i + 1]:
                minima.append(i)

        # Ensure minimum gap between boundaries (avoid micro-segments)
        min_gap = max(2, n // (max_segs + 1))
        filtered = []
        last = -min_gap
        for m in minima:
            if m - last >= min_gap:
                filtered.append(m)
                last = m

        # Enforce min_segs: if not enough natural boundaries, add evenly spaced ones
        if len(filtered) + 1 < min_segs:
            n_needed = min_segs - 1
            step = n // max(n_needed, 1)
            even = list(range(step, n, step))[:n_needed]
            # Merge and deduplicate
            merged = sorted(set(filtered + even))
            filtered = merged

        # Enforce max_segs: trim to at most max_segs - 1 boundaries
        if len(filtered) >= max_segs:
            # Keep evenly-distributed subset
            indices = np.linspace(0, len(filtered) - 1, max_segs - 1, dtype=int)
            filtered = [filtered[i] for i in indices]

        return filtered

    def _build_segment(
        self,
        points: List[WavePoint],
        index: int,
        wave: StandingWave,
    ) -> WaveSegment:
        """Compute statistics for a set of wave points and build a WaveSegment."""
        amps = np.array([p.amplitude for p in points])
        mean_amp = float(amps.mean())
        peak     = max(points, key=lambda p: p.amplitude)

        # Coherence: how tightly clustered are the vectors?
        vecs = np.stack([p.vector[:self.dim] for p in points])
        if len(vecs) > 1:
            centroid   = vecs.mean(axis=0)
            dists      = np.linalg.norm(vecs - centroid, axis=1)
            spread     = float(dists.mean())
            # High spread = low coherence; normalise to [0, 1]
            coherence  = float(np.exp(-spread / 2.0))
        else:
            coherence  = 1.0

        # Uncertainty: from the wave's metadata or from tau variance
        taus = np.array([p.tau for p in points])
        tau_spread  = float(taus.std()) if len(taus) > 1 else 0.0
        wave_uncert = float(wave.metadata.get("uncertainty", 0.5))
        # Blend wave-level uncertainty with segment-level tau spread
        uncertainty = float(0.6 * wave_uncert + 0.4 * min(tau_spread * 2.0, 1.0))

        # Flow speed: from wave metadata or from amplitude gradient
        if len(points) > 1:
            amp_diff = abs(amps[-1] - amps[0]) / (mean_amp + 1e-12)
            flow_spd = float(min(amp_diff, 1.0))
        else:
            flow_spd = float(wave.metadata.get("flow_speed", 0.5))

        return WaveSegment(
            points=points,
            mean_amplitude=mean_amp,
            peak_point=peak,
            coherence=coherence,
            uncertainty=uncertainty,
            flow_speed=flow_spd,
            index=index,
        )

    # ------------------------------------------------------------------ #
    # Stage 3 — Flow Preservation                                          #
    # ------------------------------------------------------------------ #

    def _apply_flow_preservation(
        self,
        segments: List[WaveSegment],
        matches: List[MatchResult],
    ) -> tuple[List[str], List[dict]]:
        """
        Adjust matched expressions to preserve the flow's dynamics.

        Rules (from architecture spec §2 Component 7 Stage 3)
        -------------------------------------------------------
        fast flow    → keep expression short; strip elaboration
        slow flow    → use long, measured phrasing
        loop         → add anaphoric echo of previous segment's core concept
        sharp turn   → precede with a transition phrase
        uncertainty  → ensure hedging; replace confident structures
        crystallised → use declarative, confident structures

        Returns (sentences, diagnostics).
        """
        sentences: List[str] = []
        diagnostics: List[dict] = []

        prev_peak_label = ""
        prev_flow_speed  = None

        for seg, match in zip(segments, matches):
            text   = match.expression.text
            info   = {"segment_index": seg.index, "flow_preserved": True}

            # ── Detect flow dynamics ─────────────────────────────────
            sharp_turn = (
                prev_flow_speed is not None
                and abs(seg.flow_speed - prev_flow_speed) > 0.5
            )
            is_loop = (
                prev_peak_label
                and seg.peak_point.label
                and seg.peak_point.label == prev_peak_label
            )
            is_uncertain = seg.uncertainty > 0.65
            is_confident = seg.uncertainty < 0.25 and seg.coherence > 0.6
            is_fast      = seg.flow_speed > 0.7
            is_slow      = seg.flow_speed < 0.3

            # ── Apply transformations ────────────────────────────────

            # Sharp turn: add transition
            if sharp_turn:
                transition = self._transition_phrase(prev_flow_speed, seg.flow_speed)
                text = transition + " " + text
                info["transition"] = transition

            # Loop (reinforcement): add anaphoric echo
            if is_loop and prev_peak_label:
                readable_label = self._clean_label(prev_peak_label)
                text = "Returning to " + readable_label + ": " + text
                info["anaphora"] = True

            # Uncertainty: ensure hedging language
            if is_uncertain and not match.expression.hedging:
                text = self._add_hedge(text, seg.uncertainty)
                info["hedged"] = True

            # Confidence: strip hedging if present accidentally
            if is_confident and match.expression.hedging:
                best_confident = self._find_confident_alternative(
                    match.alternatives, seg
                )
                if best_confident:
                    text = best_confident.text
                    info["dehedged"] = True

            # Fast flow: truncate to essence
            if is_fast:
                text = self._condense(text)
                info["condensed"] = True

            # Slow flow: expand with measured rhythm
            if is_slow:
                text = self._expand(text, seg)
                info["expanded"] = True

            # ── Fill template placeholders ───────────────────────────
            text = self._fill_placeholders(text, seg)

            sentences.append(text)
            diagnostics.append(info)

            prev_peak_label = seg.peak_point.label if seg.peak_point else ""
            prev_flow_speed  = seg.flow_speed

        return sentences, diagnostics

    # ------------------------------------------------------------------ #
    # Stage 3 helpers                                                       #
    # ------------------------------------------------------------------ #

    def _transition_phrase(self, prev_speed: float, curr_speed: float) -> str:
        """Generate a transition phrase for a sharp direction change."""
        if curr_speed > prev_speed:
            # Acceleration: more urgent / direct
            phrases = ["Critically,", "More directly,", "To be precise,", "In short,"]
        else:
            # Deceleration: more contemplative
            phrases = ["However,", "Consider that", "On a related note,", "Looking deeper,"]
        idx = int(abs(hash(f"{prev_speed:.2f}{curr_speed:.2f}")) % len(phrases))
        return phrases[idx]

    def _add_hedge(self, text: str, uncertainty: float) -> str:
        """Prepend a hedge phrase proportional to uncertainty level."""
        if uncertainty > 0.85:
            prefix = "It is difficult to say with certainty, but "
        elif uncertainty > 0.7:
            prefix = "It seems likely that "
        else:
            prefix = "It appears that "
        # Lower-case the first char of text if prefix ends without punctuation
        if text and text[0].isupper():
            text = text[0].lower() + text[1:]
        return prefix + text

    def _find_confident_alternative(self, alternatives, seg: WaveSegment):
        """Find the nearest non-hedging alternative expression."""
        from .matcher import ExpressionEntry
        for alt in alternatives:
            if not alt.hedging and alt.uncertainty_fit < 0.4:
                return alt
        return None

    def _condense(self, text: str) -> str:
        """
        Shorten a sentence template to its essential clause.

        Strips typical elaboration starters while preserving meaning.
        """
        starters_to_strip = [
            "As a result, ", "Under certain conditions, ",
            "The process unfolds as follows: ",
            "Beginning with",
        ]
        for s in starters_to_strip:
            if text.startswith(s):
                text = text[len(s):]
                if text and text[0].islower():
                    text = text[0].upper() + text[1:]
                break
        return text

    @staticmethod
    def _clean_label(label: str) -> str:
        """Return a human-readable version of a manifold concept label.

        Strips the domain prefix (everything before '::') so that
        'causal::mechanism' becomes 'mechanism', 'domain::mathematical'
        becomes 'mathematical', and 'demo::causal_0' becomes 'causal 0'.
        """
        if "::" in label:
            label = label.split("::")[-1]
        label = label.replace("_", " ")
        return label.strip()

    def _expand(self, text: str, seg: WaveSegment) -> str:
        """
        Expand a short expression with a contextual qualifier.

        Adds a phrase that draws on the segment's concept labels.
        """
        labels = [self._clean_label(p.label) for p in seg.points if p.label][:2]
        if labels:
            qualifier = f" — specifically regarding {' and '.join(labels)} —"
            # Insert after first clause (before the first period or comma)
            for marker in [",", "."]:
                idx = text.find(marker)
                if idx > 0:
                    text = text[:idx] + qualifier + text[idx:]
                    break
        return text

    def _fill_placeholders(self, text: str, seg: WaveSegment) -> str:
        """
        Fill template placeholders ({}) with segment-derived content.

        Fills {} slots from the wave segment's concept labels and
        peak point information.  If not enough labels, uses generic
        placeholder text that signals unknown-but-present content.
        """
        labels = [p.label for p in seg.points if p.label]

        clean_labels = [self._clean_label(l) for l in labels if l]

        # Generic fallbacks — varied phrasing so they don't repeat verbatim
        fallbacks = [
            "their interaction",
            "this relationship",
            "these factors",
            "the connected structure",
            "the causal pathway",
            "the underlying dynamic",
            "the relevant mechanism",
            "this configuration",
        ]

        slot_count = text.count("{}")
        filled_labels = (clean_labels + fallbacks)[:slot_count]

        try:
            return text.format(*filled_labels)
        except (IndexError, KeyError):
            # Safety: if format fails, just return text with {} replaced generically
            result = text
            for fl in filled_labels:
                result = result.replace("{}", fl, 1)
            # Remove any remaining {}
            result = result.replace("{}", "...")
            return result

    # ------------------------------------------------------------------ #
    # Final assembly                                                         #
    # ------------------------------------------------------------------ #

    def _assemble(self, sentences: List[str], segments: List[WaveSegment]) -> str:
        """
        Join sentence strings into a coherent response.

        Rules:
        - Paragraph break before high-contrast segments (sharp direction changes)
        - Each sentence is properly capitalised and terminated
        - Consecutive short sentences are joined into a flowing paragraph
        """
        if not sentences:
            return ""

        # Detect paragraph breaks: large amplitude drops between segments
        breaks: List[int] = []
        for i in range(1, len(segments)):
            prev = segments[i - 1]
            curr = segments[i]
            is_break = (
                abs(curr.flow_speed - prev.flow_speed) > 0.6
                or curr.uncertainty > 0.7 and prev.uncertainty < 0.3
            )
            if is_break:
                breaks.append(i)

        # Build paragraphs
        paragraphs: List[List[str]] = []
        current: List[str] = []
        for i, sent in enumerate(sentences):
            if i in breaks and current:
                paragraphs.append(current)
                current = []
            current.append(sent)
        if current:
            paragraphs.append(current)

        # Format each paragraph
        formatted_paragraphs = []
        for para in paragraphs:
            para_text = " ".join(self._clean_sentence(s) for s in para)
            # Wrap at 80 chars for readability
            wrapped = textwrap.fill(para_text, width=80)
            formatted_paragraphs.append(wrapped)

        return "\n\n".join(formatted_paragraphs)

    def _clean_sentence(self, s: str) -> str:
        """Capitalise first letter and ensure terminal punctuation."""
        s = s.strip()
        if not s:
            return s
        # Capitalise
        s = s[0].upper() + s[1:]
        # Ensure terminal punctuation
        if s and s[-1] not in ".!?":
            s += "."
        return s

    # ------------------------------------------------------------------ #
    # Stage 3b — Grammar-enhanced rendering (Phase 10)                     #
    # ------------------------------------------------------------------ #

    def _grammar_enhance(
        self,
        segments: List[WaveSegment],
        matches: List[MatchResult],
        template_sentences: List[str],
        template_diagnostics: List[dict],
    ) -> tuple[List[str], List[dict]]:
        """Attempt grammar-enhanced rendering for segments with rich data.

        For each segment, if it has ≥2 labelled wave points, the grammar
        engine composes a sentence from S-V-O role assignments derived
        from the 104D geometry.  Otherwise, the template sentence is kept.

        Parameters
        ----------
        segments            : wave segments from Stage 1
        matches             : template matches from Stage 2
        template_sentences  : sentences from template Stage 3
        template_diagnostics: diagnostics from template Stage 3

        Returns
        -------
        (enhanced_sentences, enhanced_diagnostics)
        """
        enhanced_sentences: List[str] = []
        enhanced_diagnostics: List[dict] = []

        for i, seg in enumerate(segments):
            labelled_points = [p for p in seg.points if p.label]

            # Need ≥ 2 labelled concepts for compositional grammar
            if len(labelled_points) < 2:
                # Keep template output
                if i < len(template_sentences):
                    enhanced_sentences.append(template_sentences[i])
                    enhanced_diagnostics.append(
                        template_diagnostics[i] if i < len(template_diagnostics) else {}
                    )
                continue

            # Attempt grammar rendering
            try:
                vectors = [p.vector for p in labelled_points]
                labels = [p.label for p in labelled_points]
                amplitudes = [p.amplitude for p in labelled_points]
                taus = [p.tau for p in labelled_points]

                result = self._grammar.render_segment(
                    vectors=vectors,
                    labels=labels,
                    amplitudes=amplitudes,
                    taus=taus,
                    flow_speed=seg.flow_speed,
                    coherence=seg.coherence,
                    uncertainty=seg.uncertainty,
                )

                if result.text and len(result.text) > 5:
                    enhanced_sentences.append(result.text)
                    diag = template_diagnostics[i].copy() if i < len(template_diagnostics) else {}
                    diag["grammar_enhanced"] = True
                    diag["grammar_complexity"] = result.complexity
                    diag["grammar_tense"] = result.tense
                    diag["grammar_confidence"] = result.confidence
                    enhanced_diagnostics.append(diag)
                    continue
            except Exception:
                pass  # fall through to template

            # Fallback: keep template output
            if i < len(template_sentences):
                enhanced_sentences.append(template_sentences[i])
                enhanced_diagnostics.append(
                    template_diagnostics[i] if i < len(template_diagnostics) else {}
                )

        return enhanced_sentences, enhanced_diagnostics
