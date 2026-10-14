from __future__ import annotations

import re

from presentation_assessment.features.common import ModalityScore, clamp


STRUCTURE_MARKERS = {
    "first",
    "second",
    "third",
    "finally",
    "conclusion",
    "in conclusion",
    "result",
    "method",
    "experiment",
    "introduction",
    "\u9996\u5148",
    "\u5176\u6b21",
    "\u6700\u540e",
    "\u7ed3\u8bba",
    "\u65b9\u6cd5",
    "\u5b9e\u9a8c",
    "\u7ed3\u679c",
}

FILLER_MARKERS = {
    "um",
    "uh",
    "er",
    "ah",
    "like",
    "\u55ef",
    "\u554a",
    "\u7136\u540e",
}


class SpeechScorer:
    """Lightweight content and delivery scorer used as the initial baseline."""

    def score_content(self, transcript: str) -> ModalityScore:
        tokens = self._tokens(transcript)
        token_count = len(tokens)
        unique_count = len(set(tokens))
        lexical_richness = unique_count / token_count if token_count else 0.0

        lower_text = transcript.lower()
        marker_hits = sum(1 for marker in STRUCTURE_MARKERS if marker in lower_text)

        length_score = self._length_score(token_count)
        structure_score = clamp(marker_hits / 6 * 100)
        lexical_score = clamp(lexical_richness * 120)
        final_score = clamp(0.45 * length_score + 0.35 * structure_score + 0.20 * lexical_score)

        return ModalityScore(
            score=round(final_score, 2),
            explanation=(
                f"content baseline: {token_count} tokens, "
                f"{marker_hits} structure markers, lexical richness {lexical_richness:.2f}"
            ),
            features={
                "token_count": token_count,
                "unique_token_count": unique_count,
                "lexical_richness": round(lexical_richness, 4),
                "structure_marker_hits": marker_hits,
                "length_score": round(length_score, 2),
                "structure_score": round(structure_score, 2),
                "lexical_score": round(lexical_score, 2),
            },
        )

    def score_delivery(
        self, transcript: str, precomputed_score: float | None = None
    ) -> ModalityScore:
        if precomputed_score is not None:
            return ModalityScore(
                score=round(precomputed_score, 2),
                explanation="speech delivery score loaded from precomputed features or manifest",
                features={"source": "manifest"},
            )

        tokens = self._tokens(transcript)
        token_count = len(tokens)
        sentence_count = max(1, len(re.findall(r"[.!?\n]+", transcript)))
        avg_sentence_len = token_count / sentence_count if sentence_count else 0.0
        lower_text = transcript.lower()
        filler_hits = sum(1 for marker in FILLER_MARKERS if marker in lower_text)

        flow_score = self._sentence_flow_score(avg_sentence_len)
        filler_penalty = min(25.0, filler_hits * 5.0)
        length_support = min(100.0, token_count / 70 * 100) if token_count else 0.0
        final_score = clamp(0.60 * flow_score + 0.25 * length_support + 15.0 - filler_penalty)

        return ModalityScore(
            score=round(final_score, 2),
            explanation=(
                "speech delivery proxy: "
                f"{sentence_count} sentence units, average {avg_sentence_len:.1f} tokens, "
                f"{filler_hits} filler markers"
            ),
            features={
                "source": "transcript_proxy",
                "token_count": token_count,
                "sentence_count": sentence_count,
                "average_sentence_tokens": round(avg_sentence_len, 2),
                "filler_marker_hits": filler_hits,
                "flow_score": round(flow_score, 2),
                "length_support": round(length_support, 2),
            },
        )

    def score(self, transcript: str) -> ModalityScore:
        return self.score_content(transcript)

    @staticmethod
    def _tokens(text: str) -> list[str]:
        english_words = re.findall(r"[A-Za-z]+(?:'[A-Za-z]+)?", text.lower())
        cjk_chars = re.findall(r"[\u4e00-\u9fff]", text)
        return english_words + cjk_chars

    @staticmethod
    def _length_score(token_count: int) -> float:
        if token_count <= 0:
            return 0.0
        target = 90
        ratio = min(token_count / target, target / token_count)
        return clamp(ratio * 100)

    @staticmethod
    def _sentence_flow_score(avg_sentence_len: float) -> float:
        if avg_sentence_len <= 0:
            return 0.0
        target = 18.0
        ratio = min(avg_sentence_len / target, target / avg_sentence_len)
        return clamp(ratio * 100)
