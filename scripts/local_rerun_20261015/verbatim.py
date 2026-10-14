"""Station 7 measures on a verbatim transcript (plan: docs/research/2026-09-29-verbatim-plan.md).

`words` is a list of (start_s, end_s, text) in recording time; nothing here touches audio.
"""
import re

FILLERS = frozenset({"uh", "um", "uhm", "umm", "er", "erm", "ah", "eh", "hmm", "mm"})
_EDGE = re.compile(r"^\W+|\W+$")


def norm(text):
    """Lower-case a written word and strip surrounding punctuation, apostrophes and brackets ('[UM]' -> 'um')."""
    return _EDGE.sub("", text.lower())


def _inside(words, iv):
    if iv is None:
        return list(words)
    a, b = iv
    return [w for w in words if w[0] >= a and w[1] <= b]


def filler_spans(words, iv=None):
    """(start, end) of every filler word (D4), counting only words wholly inside `iv` when given."""
    return [(s, e) for s, e, t in _inside(words, iv) if norm(t) in FILLERS]


def repetition_spans(words, iv=None):
    """(start, end) of every repetition site (D5).

    Fillers are dropped first. A site is two identical neighbouring words ("the the") or two identical
    neighbouring word pairs ("I think I think"); overlapping hits ("the the the") count as one site.
    """
    kept = [(s, e, n) for s, e, t in _inside(words, iv) if (n := norm(t)) and n not in FILLERS]
    hits = []
    for i in range(len(kept) - 1):
        if kept[i][2] == kept[i + 1][2]:
            hits.append((i, i + 1))
        if i + 3 < len(kept) and (kept[i][2], kept[i + 1][2]) == (kept[i + 2][2], kept[i + 3][2]):
            hits.append((i, i + 3))
    sites = []
    for first, last in sorted(hits):
        if sites and first <= sites[-1][1]:
            sites[-1] = (sites[-1][0], max(sites[-1][1], last))
        else:
            sites.append((first, last))
    return [(kept[first][0], kept[last][1]) for first, last in sites]


def score_clip(words, window, human, tol=0.3):
    """Compare fillers and repetitions found inside one clip window with the human marks.

    `human` maps "uh" / "repeat" to (start, end) ranges in seconds from the clip start; `window` is the clip's
    (start, end) in recording time. Returns detected / human / matched counts per type.
    """
    s0 = window[0]
    found = {"uh": filler_spans(words, window), "repeat": repetition_spans(words, window)}
    out = {}
    for kind, spans in found.items():
        detected = [(a - s0, b - s0) for a, b in spans]
        marked = human.get(kind, [])
        out[kind] = {"detected": len(detected), "human": len(marked), "matched": match(detected, marked, tol)}
    return out


def match(detected, human, tol=0.0):
    """Largest number of one-to-one pairs between detected spans and human spans.

    A detected span, widened by `tol` seconds on both sides, may pair with any human span it overlaps
    (merely touching does not count). This is a maximum bipartite matching, so no pair is lost to pairing order.
    """
    edges = [[j for j, (h0, h1) in enumerate(human) if min(d1 + tol, h1) - max(d0 - tol, h0) > 0]
             for d0, d1 in detected]
    owner = {}

    def take(i, seen):
        for j in edges[i]:
            if j in seen:
                continue
            seen.add(j)
            if j not in owner or take(owner[j], seen):
                owner[j] = i
                return True
        return False

    return sum(take(i, set()) for i in range(len(detected)))
