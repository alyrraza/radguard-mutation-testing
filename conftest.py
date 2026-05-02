"""
conftest.py — Mock heavy imports BEFORE pipeline.py is loaded.

inference.model  loads 550 MB of PyTorch weights at import time.
inference.chexbert_runner spawns a BERT subprocess.
Both are replaced with lightweight mocks so tests run in milliseconds.

This file is executed by pytest BEFORE any test module is imported,
so sys.modules manipulation here prevents the real modules from ever loading.
"""

import sys
from unittest.mock import MagicMock

# ── Conditions used throughout tests (subset of real 14) ──────────────────
CONDITIONS_MOCK = [
    'Cardiomegaly',
    'Edema',
    'Consolidation',
    'Atelectasis',
    'Pleural Effusion',
]

# ── Mock inference.chexbert_runner ─────────────────────────────────────────
_mock_chexbert = MagicMock()
_mock_chexbert.run_chexbert.return_value = [{}]
_mock_chexbert.chexbert_to_tensor.return_value = MagicMock()
_mock_chexbert.keyword_fallback.return_value = {}

# ── Mock inference.model ───────────────────────────────────────────────────
_mock_model = MagicMock()
_mock_model.CONDITIONS = CONDITIONS_MOCK
_mock_model.run_inference_on_sentence.return_value = (
    MagicMock(), MagicMock(), MagicMock()
)
_mock_model.device = 'cpu'

# Register mocks BEFORE pipeline.py's top-level imports execute
sys.modules['inference.chexbert_runner'] = _mock_chexbert
sys.modules['inference.model']           = _mock_model
