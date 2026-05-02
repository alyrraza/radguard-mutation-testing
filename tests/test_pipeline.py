"""
Mutation-testing-focused test suite for inference/pipeline.py.

Target functions  : split_report, compute_elrrs, aggregate_verdicts
Mutation operators: ROR (relational), LCR (logical connector), SVR (return value)

Each test documents exactly which mutant it is designed to kill and why.
"""

import pytest
from inference.pipeline import (
    split_report,
    compute_elrrs,
    aggregate_verdicts,
    CONDITIONS,
)


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def make_verdicts(verdict_list):
    """
    Build a minimal report_verdicts dict from a flat list of verdict strings.
    compute_elrrs only needs v['verdict'] — other keys are not accessed.
    Keys are synthetic ('cond_0', 'cond_1', ...) so they never conflict with
    the mocked CONDITIONS list.
    """
    return {
        f'cond_{i}': {'verdict': v, 'confidence': 0.9}
        for i, v in enumerate(verdict_list)
    }


def make_t2_preds(present, confidence):
    """
    Return a one-sentence t2_preds list where ALL mocked CONDITIONS share
    the same present/confidence values.  aggregate_verdicts always reads
    all_t2_preds[0][cond] for conditions not mentioned in the report.
    """
    return [{cond: {'present': present, 'confidence': confidence}
             for cond in CONDITIONS}]


# ─────────────────────────────────────────────────────────────────────────────
# split_report tests  (Tests 1–6)
# ─────────────────────────────────────────────────────────────────────────────

def test_split_report_normal():
    """
    Normal multi-sentence report is split correctly.
    Baseline sanity check — kills SVR mutants that return [] or None.
    """
    result = split_report(
        "The heart is mildly enlarged. No pleural effusion is seen."
    )
    assert len(result) == 2, f"Expected 2 sentences, got {len(result)}"
    assert "The heart is mildly enlarged" in result[0]


def test_split_report_empty_string():
    """
    Empty string must return an empty list.
    Kills SVR mutant: return value changed from [] to non-empty default.
    """
    result = split_report("")
    assert result == [], f"Empty string must give [], got {result}"


def test_split_report_short_word_filtered():
    """
    A single token under 5 chars ('OK') must be filtered out.
    Kills ROR mutant: len > 5 changed to len >= 5 would still filter 'OK' (len=2),
    so this confirms the filter is present at all.
    """
    result = split_report("OK")
    assert result == [], f"Short token 'OK' (len=2) must be filtered, got {result}"


def test_split_report_exactly_5_chars_filtered():
    """
    A sentence of EXACTLY 5 characters must be filtered (boundary: len > 5).
    Original : len('Hello') = 5 > 5 → False → filtered  ✓
    ROR mutant: len('Hello') = 5 >= 5 → True  → NOT filtered  ✗
    This test asserts the original behaviour and therefore KILLS the >= mutant.
    """
    result = split_report("Hello")
    assert result == [], (
        "Sentence of exactly 5 chars must be filtered out. "
        "Kills ROR mutant: '> 5' changed to '>= 5'."
    )


def test_split_report_exactly_6_chars_passes():
    """
    A sentence of EXACTLY 6 characters must pass through (len=6 > 5 is True).
    Original : 6 > 5 → True  → kept    ✓
    ROR mutant: 6 >= 5 → True → kept    (same — does NOT kill this mutant)
    ROR mutant: 6 > 6  → False → filtered (would kill mutant if threshold changed)

    This test verifies the lower boundary passes, complementing Test 4.
    """
    result = split_report("Lungs.")
    assert len(result) == 1, (
        "Sentence of exactly 6 chars must be kept. "
        "len('Lungs.') = 6, and 6 > 5 is True."
    )
    assert result[0] == "Lungs."


def test_split_report_whitespace_only():
    """
    Whitespace-only string must return an empty list.
    .strip() reduces it to '', sent_tokenize('') returns [], so result is [].
    Kills SVR mutant that returns a non-empty list for blank input.
    """
    result = split_report("   \n\t  ")
    assert result == [], f"Whitespace-only input must give [], got {result}"


# ─────────────────────────────────────────────────────────────────────────────
# compute_elrrs tests  (Tests 7–14)
# ─────────────────────────────────────────────────────────────────────────────

def test_elrrs_all_supported():
    """
    All SUPPORTED → score = 100.0, grade = 'Excellent'.
    Verifies that perfect reports are correctly identified.
    """
    result = compute_elrrs(make_verdicts(['SUPPORTED'] * 5))
    assert result['score'] == 100.0,  f"Expected 100.0, got {result['score']}"
    assert result['grade'] == 'Excellent', f"Expected Excellent, got {result['grade']}"
    assert result['supported_count'] == 5


def test_elrrs_all_hallucinated():
    """
    All HALLUCINATED → score clipped to 0.0, grade = 'Critical'.
    np.clip prevents negative scores. Verifies clipping and worst-case grade.
    Kills SVR mutant: score returned as negative instead of clipped to 0.
    """
    result = compute_elrrs(make_verdicts(['HALLUCINATED'] * 5))
    assert result['score'] == 0.0,   f"Expected 0.0, got {result['score']}"
    assert result['grade'] == 'Critical', f"Expected Critical, got {result['grade']}"
    assert result['hallucinated_count'] == 5


def test_elrrs_all_not_mentioned_returns_excellent():
    """
    All NOT_MENTIONED → no active conditions → early-return with score=100, grade='Excellent'.
    Kills SVR mutant: early return value changed (e.g. score=0 or grade='Critical').
    Original code returns hardcoded {'score': 100.0, 'grade': 'Excellent', ...}.
    """
    result = compute_elrrs(make_verdicts(['NOT_MENTIONED'] * 5))
    assert result['score'] == 100.0, f"Expected 100.0, got {result['score']}"
    assert result['grade'] == 'Excellent', f"Expected Excellent, got {result['grade']}"
    assert result['active_count'] == 0
    assert result['not_mentioned_count'] == 5


def test_elrrs_grade_exactly_80():
    """
    Score of EXACTLY 80.0 must give grade 'Excellent'.

    *** PRIMARY ROR MUTANT KILLER ***

    Math (float-safe — uses only 1.0 and 0.5 which are exact in IEEE 754):
      13 SUPPORTED (weight +1.0) + 2 MISSING (weight -0.5)
      raw = 13*1.0 + 2*(-0.5) = 13.0 - 1.0 = 12.0  (exact)
      score = 12.0 / 15 * 100 = 80.0                 (exact)

    Original  : score=80.0 >= 80 → True  → grade='Excellent'  ✓
    ROR mutant: score=80.0 >  80 → False → drops to 'Good'     ✗
    This test asserts grade='Excellent' and therefore KILLS the mutant.
    """
    verdicts = ['SUPPORTED'] * 13 + ['MISSING'] * 2
    result = compute_elrrs(make_verdicts(verdicts))
    assert result['score'] == 80.0, (
        f"Expected score=80.0, got {result['score']}. "
        "Check float arithmetic: 12/15*100 should be exactly 80.0."
    )
    assert result['grade'] == 'Excellent', (
        f"score=80.0 must map to 'Excellent' (threshold: score >= 80). "
        f"Got '{result['grade']}'. Kills ROR mutant '>= 80' → '> 80'."
    )


def test_elrrs_grade_exactly_60():
    """
    Score of EXACTLY 60.0 must give grade 'Good'.

    Math: 11 SUPPORTED + 4 MISSING over 15 total
      raw = 11.0 - 2.0 = 9.0  →  score = 9/15*100 = 60.0  (exact)

    Original  : 60.0 >= 60 → True  → grade='Good'  ✓
    ROR mutant: 60.0 >  60 → False → drops to 'Fair' ✗
    """
    verdicts = ['SUPPORTED'] * 11 + ['MISSING'] * 4
    result = compute_elrrs(make_verdicts(verdicts))
    assert result['score'] == 60.0, f"Expected 60.0, got {result['score']}"
    assert result['grade'] == 'Good', (
        f"score=60.0 must map to 'Good'. Got '{result['grade']}'. "
        "Kills ROR mutant '>= 60' → '> 60'."
    )


def test_elrrs_grade_exactly_40():
    """
    Score of EXACTLY 40.0 must give grade 'Fair'.

    Math: 3 SUPPORTED + 2 MISSING over 5 total
      raw = 3.0 - 1.0 = 2.0  →  score = 2/5*100 = 40.0  (exact)

    Original  : 40.0 >= 40 → True  → grade='Fair'  ✓
    ROR mutant: 40.0 >  40 → False → drops to 'Poor' ✗
    """
    verdicts = ['SUPPORTED'] * 3 + ['MISSING'] * 2
    result = compute_elrrs(make_verdicts(verdicts))
    assert result['score'] == 40.0, f"Expected 40.0, got {result['score']}"
    assert result['grade'] == 'Fair', (
        f"score=40.0 must map to 'Fair'. Got '{result['grade']}'. "
        "Kills ROR mutant '>= 40' → '> 40'."
    )


def test_elrrs_grade_exactly_20():
    """
    Score of EXACTLY 20.0 must give grade 'Poor'.

    Math: 7 SUPPORTED + 8 MISSING over 15 total
      raw = 7.0 - 4.0 = 3.0  →  score = 3/15*100 = 20.0  (exact)

    Original  : 20.0 >= 20 → True  → grade='Poor'     ✓
    ROR mutant: 20.0 >  20 → False → drops to 'Critical' ✗
    """
    verdicts = ['SUPPORTED'] * 7 + ['MISSING'] * 8
    result = compute_elrrs(make_verdicts(verdicts))
    assert result['score'] == 20.0, f"Expected 20.0, got {result['score']}"
    assert result['grade'] == 'Poor', (
        f"score=20.0 must map to 'Poor'. Got '{result['grade']}'. "
        "Kills ROR mutant '>= 20' → '> 20'."
    )


def test_elrrs_mixed_verdict_counts():
    """
    Mixed verdicts: verify all individual counts are tracked correctly.
    Kills SVR / AOR mutants that corrupt the counts dict.
    Input: 2 SUPPORTED + 1 HALLUCINATED + 1 MISSING + 1 INACCURATE = 5 active
    """
    verdicts = ['SUPPORTED', 'SUPPORTED', 'HALLUCINATED', 'MISSING', 'INACCURATE']
    result = compute_elrrs(make_verdicts(verdicts))
    assert result['supported_count']    == 2, f"Expected 2 supported,    got {result['supported_count']}"
    assert result['hallucinated_count'] == 1, f"Expected 1 hallucinated, got {result['hallucinated_count']}"
    assert result['missing_count']      == 1, f"Expected 1 missing,      got {result['missing_count']}"
    assert result['inaccurate_count']   == 1, f"Expected 1 inaccurate,   got {result['inaccurate_count']}"
    assert result['active_count']       == 5, f"Expected 5 active,       got {result['active_count']}"


# ─────────────────────────────────────────────────────────────────────────────
# aggregate_verdicts tests  (Tests 15–18)
# ─────────────────────────────────────────────────────────────────────────────
#
# Strategy: pass empty chexbert labels ({}) so ALL conditions take the
# NOT_MENTIONED/MISSING branch (line 119-122 of pipeline.py).
# This isolates the two operators under test:
#   verdict = ('MISSING' if (t2_present and t2_conf >= t2_min) else 'NOT_MENTIONED')

def _run_aggregate(present, confidence):
    """
    Helper: run aggregate_verdicts with all conditions unmentiond in report,
    returning the verdict for the first condition in CONDITIONS.
    """
    result = aggregate_verdicts(
        sentences          = ["chest x-ray report sentence"],
        all_sentence_preds = [{}],             # never accessed (no mentioned conds)
        all_t2_preds       = make_t2_preds(present, confidence),
        all_chexbert_labels= [{}],             # empty → all conditions unmentioned
    )
    return result[CONDITIONS[0]]['verdict']


def test_aggregate_missing_at_exact_boundary():
    """
    t2_present=True, t2_conf=0.65 (exactly equal to t2_min=0.65) → MISSING.

    *** PRIMARY ROR MUTANT KILLER for aggregate_verdicts ***

    Original  : 0.65 >= 0.65 → True  → MISSING        ✓
    ROR mutant: 0.65 >  0.65 → False → NOT_MENTIONED   ✗

    Both sides of the comparison use the same float64 representation of 0.65,
    so >= is True and > is False — the boundary is sharp.
    """
    verdict = _run_aggregate(present=True, confidence=0.65)
    assert verdict == 'MISSING', (
        f"t2_present=True, t2_conf=0.65 (= t2_min) must give MISSING. "
        f"Got '{verdict}'. Kills ROR mutant '>= t2_min' → '> t2_min'."
    )


def test_aggregate_not_mentioned_just_below_boundary():
    """
    t2_present=True, t2_conf=0.64 (just below t2_min=0.65) → NOT_MENTIONED.

    Original  : 0.64 >= 0.65 → False → NOT_MENTIONED  ✓
    ROR mutant: 0.64 >  0.65 → False → NOT_MENTIONED  (same — does not kill)

    This is the complement of Test 15 — verifies the sub-threshold path works.
    """
    verdict = _run_aggregate(present=True, confidence=0.64)
    assert verdict == 'NOT_MENTIONED', (
        f"t2_conf=0.64 < t2_min=0.65 must give NOT_MENTIONED. Got '{verdict}'."
    )


def test_aggregate_lcr_false_present_high_confidence():
    """
    t2_present=False, t2_conf=0.90 → NOT_MENTIONED.

    *** PRIMARY LCR MUTANT KILLER ***

    The expression is: t2_present AND t2_conf >= t2_min

    Original  : False AND (0.90 >= 0.65) = False AND True = False → NOT_MENTIONED  ✓
    LCR mutant: False OR  (0.90 >= 0.65) = False OR  True = True  → MISSING        ✗

    t2_present=False means the finding is NOT visible on the X-ray.
    The LCR mutant would incorrectly flag it as MISSING — a dangerous false alarm.
    This test catches that error by asserting NOT_MENTIONED.
    """
    verdict = _run_aggregate(present=False, confidence=0.90)
    assert verdict == 'NOT_MENTIONED', (
        f"t2_present=False must give NOT_MENTIONED even with high confidence. "
        f"Got '{verdict}'. Kills LCR mutant 'and' → 'or'."
    )


def test_aggregate_missing_high_confidence():
    """
    t2_present=True, t2_conf=0.80 (well above t2_min) → MISSING.

    Verifies the MISSING path works for clearly above-threshold confidence.
    Complements the boundary test — ensures the condition is not only triggered
    at the exact boundary.
    """
    verdict = _run_aggregate(present=True, confidence=0.80)
    assert verdict == 'MISSING', (
        f"t2_present=True, t2_conf=0.80 >> t2_min=0.65 must give MISSING. "
        f"Got '{verdict}'."
    )


# ─────────────────────────────────────────────────────────────────────────────
# Weight magnitude tests  (Tests 19–21)
# ─────────────────────────────────────────────────────────────────────────────

def test_elrrs_inaccurate_weight_is_negative():
    """
    Verifies that INACCURATE verdicts produce a LOWER score than
    all-SUPPORTED verdicts of the same count.
    Kills mutmut #10 (AOR: INACCURATE weight -0.3 -> +0.3).
    Original: INACCURATE penalizes score (negative weight) -> score < 100
    Mutant:   INACCURATE rewards score (positive weight) -> score inflated
    4 SUPPORTED + 1 INACCURATE (5 total active):
    Original score = (4x1.0 + 1x(-0.3)) / 5 x 100 = 74.0
    Mutant score   = (4x1.0 + 1x(+0.3)) / 5 x 100 = 86.0
    """
    verdicts = make_verdicts(
        ['SUPPORTED'] * 4 + ['INACCURATE']
    )
    result = compute_elrrs(verdicts)
    assert result['score'] == 74.0, (
        f"Expected score 74.0 with 1 INACCURATE, got {result['score']}. "
        f"INACCURATE must penalize, not reward."
    )
    assert result['grade'] == 'Good', (
        f"Expected grade Good (score 74 is in 60-80 range), "
        f"got {result['grade']}"
    )


def test_elrrs_inaccurate_weight_magnitude():
    """
    Verifies INACCURATE penalty is -0.3, not a larger value.
    Kills mutmut #11 (AOR: INACCURATE weight -0.3 -> -1.3).
    4 SUPPORTED + 1 INACCURATE out of 5 active:
    Original (-0.3): score = (4 - 0.3) / 5 * 100 = 74.0 -> Good
    Mutant   (-1.3): score = (4 - 1.3) / 5 * 100 = 54.0 -> Fair
    """
    verdicts = make_verdicts(
        ['SUPPORTED'] * 4 + ['INACCURATE']
    )
    result = compute_elrrs(verdicts)
    assert result['score'] == 74.0, (
        f"Expected score 74.0, got {result['score']}. "
        f"INACCURATE weight must be -0.3, not more severe."
    )
    assert result['grade'] == 'Good', (
        f"Expected Good grade, got {result['grade']}. "
        f"Mutant #11 causes grade to drop to Fair."
    )


def test_elrrs_hallucinated_weight_magnitude():
    """
    Kills mutmut #17 (AOR: HALLUCINATED weight -0.7 -> -1.7).
    Uses mixed input: 3 SUPPORTED + 2 HALLUCINATED.
    Original (-0.7): raw = 3.0 - 1.4 = 1.6  -> score = 1.6/5*100 = 32.0 -> Poor
    Mutant   (-1.7): raw = 3.0 - 3.4 = -0.4 -> clipped to 0.0           -> Critical
    Score 32.0 != 0.0 kills the mutant.
    """
    verdicts = make_verdicts(
        ['SUPPORTED'] * 3 + ['HALLUCINATED'] * 2
    )
    result = compute_elrrs(verdicts)
    assert result['score'] == 32.0, (
        f"Expected score 32.0 for 3 SUPPORTED + 2 HALLUCINATED, "
        f"got {result['score']}. HALLUCINATED weight must be -0.7."
    )
    assert result['grade'] == 'Poor', (
        f"Expected Poor grade (score 32 is in 20-40 range), got {result['grade']}. "
        f"Mutant #17 (-1.7 weight) clips to 0.0 -> Critical."
    )
