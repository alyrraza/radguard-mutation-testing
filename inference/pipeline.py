"""
Pipeline — Sab kuch combine karta hai:
1. Report split karo sentences mein
2. CheXbert chalao
3. Model chalao per sentence
4. Results aggregate karo
5. ELRRs score nikalo

ELRRs Metric inspired by:
- Yu et al. (2023) "Evaluating progress in automatic chest X-ray
  radiology report generation" Patterns, Vol 4(9)
  DOI: 10.1016/j.patter.2023.100802
- Jain et al. (2021) "RadGraph: Extracting Clinical Entities
  and Relations from Radiology Reports" NeurIPS 2021
"""

import numpy as np
from PIL import Image
from nltk.tokenize import sent_tokenize
import nltk
nltk.download('punkt', quiet=True)
nltk.download('punkt_tab', quiet=True)

from .chexbert_runner import run_chexbert, chexbert_to_tensor
from .model import run_inference_on_sentence, CONDITIONS

# ── ELRRs Weights ─────────────────────────────────────────────
# Inspired by RadCliQ error taxonomy (Yu et al. 2023)
# Weights proportional to clinical severity:
# Hallucination > Missing > Inaccurate
ELRRS_WEIGHTS = {
    'SUPPORTED':     1.0,
    'NOT_MENTIONED': 0.0,
    'INACCURATE':   -0.3,
    'MISSING':      -0.5,
    'HALLUCINATED': -0.7,
}
ELRRS_GRADES = [
    (80, 'Excellent', '#4CAF50', 'Clinically safe — minimal errors'),
    (60, 'Good',      '#8BC34A', 'Minor errors — clinically acceptable'),
    (40, 'Fair',      '#FF9800', 'Moderate errors — review advised'),
    (20, 'Poor',      '#F44336', 'Significant errors — high risk'),
    (0,  'Critical',  '#9C27B0', 'Severe errors — unsafe'),
]

CHEXBERT_COLS = [
    'Enlarged Cardiomediastinum', 'Cardiomegaly',
    'Lung Opacity', 'Lung Lesion', 'Edema',
    'Consolidation', 'Pneumonia', 'Atelectasis',
    'Pneumothorax', 'Pleural Effusion', 'Pleural Other',
    'Fracture', 'Support Devices', 'No Finding'
]
ERROR_PRIORITY = {
    'HALLUCINATED': 3, 'INACCURATE': 2,
    'MISSING': 1, 'SUPPORTED': 0
}
ERROR_MEANING = {
    'SUPPORTED':     'AI report is correct — X-ray confirms it',
    'HALLUCINATED':  'AI made this up — not visible on X-ray',
    'MISSING':       'AI forgot this — it IS visible on X-ray',
    'INACCURATE':    'AI mentioned it but described it wrongly',
    'NOT_MENTIONED': 'Condition not present — AI correctly silent',
}


def split_report(report_text: str) -> list:
    """Report ko sentences mein split karo."""
    sents = sent_tokenize(report_text.strip())
    return [s.strip() for s in sents if len(s.strip()) > 5]


def aggregate_verdicts(sentences, all_sentence_preds,
                       all_t2_preds, all_chexbert_labels,
                       t2_min=0.65) -> dict:
    """
    Har sentence ke predictions ko report-level pe combine karo.
    Sabse severe verdict lo per condition.
    """
    import math
    mentioned_in = {cond: [] for cond in CONDITIONS}

    for si, chex_labels in enumerate(all_chexbert_labels):
        found = False
        for cond, chex_col in zip(CONDITIONS, CHEXBERT_COLS):
            v = chex_labels.get(chex_col, float('nan'))
            try:
                fv = float(v)
                if not math.isnan(fv):
                    mentioned_in[cond].append(si)
                    found = True
            except:
                pass

    report_verdicts = {}
    for cond in CONDITIONS:
        if mentioned_in[cond]:
            best_pred, best_conf, best_si = None, -1.0, -1
            for si in mentioned_in[cond]:
                pred = all_sentence_preds[si][cond]['prediction']
                conf = all_sentence_preds[si][cond]['confidence']
                if (best_pred is None or
                        ERROR_PRIORITY[pred] > ERROR_PRIORITY[best_pred] or
                        (ERROR_PRIORITY[pred] == ERROR_PRIORITY[best_pred]
                         and conf > best_conf)):
                    best_pred, best_conf, best_si = pred, conf, si

            report_verdicts[cond] = {
                'verdict':     best_pred,
                'confidence':  best_conf,
                'source':      f"Sentence {best_si + 1}",
                'source_text': sentences[best_si],
                'mentioned':   True,
                't2_present':  all_t2_preds[best_si][cond]['present'],
                't2_conf':     all_t2_preds[best_si][cond]['confidence'],
                'probs':       all_sentence_preds[best_si][cond]['probs'],
                'meaning':     ERROR_MEANING.get(best_pred, ''),
            }
        else:
            t2_present = all_t2_preds[0][cond]['present']
            t2_conf    = all_t2_preds[0][cond]['confidence']
            verdict    = ('MISSING' if (t2_present and t2_conf >= t2_min)
                          else 'NOT_MENTIONED')
            report_verdicts[cond] = {
                'verdict':     verdict,
                'confidence':  t2_conf,
                'source':      'Not mentioned in report',
                'source_text': '',
                'mentioned':   False,
                't2_present':  t2_present,
                't2_conf':     t2_conf,
                'probs':       {k: 0.0 for k in
                                ['SUPPORTED','HALLUCINATED',
                                 'MISSING','INACCURATE']},
                'meaning':     ERROR_MEANING.get(verdict, ''),
            }

    return report_verdicts


def compute_elrrs(report_verdicts: dict) -> dict:
    """ELRRs score compute karo — RadCliQ taxonomy inspired."""
    active = [(c, v) for c, v in report_verdicts.items()
              if v['verdict'] != 'NOT_MENTIONED']
    not_mentioned_count = sum(
        1 for v in report_verdicts.values()
        if v['verdict'] == 'NOT_MENTIONED')

    if not active:
        return {
            'score': 100.0, 'grade': 'Excellent',
            'grade_color': '#4CAF50',
            'grade_desc': 'No active conditions',
            'active_count': 0,
            'not_mentioned_count': not_mentioned_count,
            'supported_count': 0, 'hallucinated_count': 0,
            'missing_count': 0, 'inaccurate_count': 0,
        }

    raw_score    = 0.0
    max_possible = float(len(active))
    counts = {
        'SUPPORTED': 0, 'HALLUCINATED': 0,
        'MISSING': 0, 'INACCURATE': 0
    }

    for cond, v in active:
        verdict   = v['verdict']
        weight    = ELRRS_WEIGHTS.get(verdict, 0.0)
        raw_score += weight
        if verdict in counts:
            counts[verdict] += 1

    score = float(np.clip((raw_score / max_possible) * 100.0, 0, 100))

    grade, grade_color, grade_desc = 'Critical', '#9C27B0', 'Severe errors'
    for threshold, g, gc, gd in ELRRS_GRADES:
        if score >= threshold:
            grade, grade_color, grade_desc = g, gc, gd
            break

    return {
        'score':               round(score, 2),
        'grade':               grade,
        'grade_color':         grade_color,
        'grade_desc':          grade_desc,
        'active_count':        len(active),
        'not_mentioned_count': not_mentioned_count,
        'supported_count':     counts['SUPPORTED'],
        'hallucinated_count':  counts['HALLUCINATED'],
        'missing_count':       counts['MISSING'],
        'inaccurate_count':    counts['INACCURATE'],
    }


def run_full_pipeline(image: Image.Image,
                      ai_report: str,
                      device=None) -> dict:
    """
    MAIN FUNCTION — main.py se yeh call karo.

    Input:
        image     — PIL Image (X-ray)
        ai_report — string (AI generated report)
    Output:
        {elrrs, conditions, not_mentioned}
    """
    from .model import device as model_device
    if device is None:
        device = model_device

    # Step 1: Split
    sentences = split_report(ai_report)
    if not sentences:
        sentences = [ai_report]
    print(f"📝 {len(sentences)} sentence(s)")

    # Step 2: CheXbert
    all_chexbert = run_chexbert(sentences)
    print("🔬 CheXbert done")

    # Step 3: Model per sentence
    all_sentence_preds, all_t2_preds, all_attn = [], [], []
    for si, (sent, chex) in enumerate(zip(sentences, all_chexbert)):
        print(f"🤖 S{si+1}: {sent[:50]}...")
        chex_t = chexbert_to_tensor(chex, device)
        preds, t2_preds, attn = run_inference_on_sentence(
            image, sent, chex_t)
        all_sentence_preds.append(preds)
        all_t2_preds.append(t2_preds)
        all_attn.append(attn)

    # Step 4: Aggregate
    report_verdicts = aggregate_verdicts(
        sentences, all_sentence_preds,
        all_t2_preds, all_chexbert)
    print("📊 Aggregation done")

    # Step 5: ELRRs
    elrrs = compute_elrrs(report_verdicts)
    print(f"⭐ ELRRs: {elrrs['score']} — {elrrs['grade']}")

    # Step 6: Format output
    active_conditions = []
    not_mentioned     = []

    for cond in CONDITIONS:
        v = report_verdicts[cond]
        if v['verdict'] != 'NOT_MENTIONED':
            active_conditions.append({
                'name':        cond,
                'verdict':     v['verdict'],
                'confidence':  round(v['confidence'], 4),
                'meaning':     v['meaning'],
                'source':      v['source'],
                'source_text': v['source_text'],
                'mentioned':   v['mentioned'],
                'xray_present': v['t2_present'],
            })
        else:
            not_mentioned.append(cond)

    return {
        'elrrs':            elrrs,
        'conditions':       active_conditions,
        'not_mentioned':    not_mentioned,
        'all_attn':         all_attn,
        'sentences':        sentences,
        'all_chexbert':     all_chexbert,
    }
