# RadGuard Mutation Testing Assignment

## CS-4006 Software Testing — FAST NUCES Spring 2026

## FYP: RadGuard — AI-Based Radiology Report Error Detection

### What This Repository Contains

This is the mutation testing assignment for the RadGuard FYP.
Target module: inference/pipeline.py (ELRRs scoring and verdict logic)

### Installation

pip install -r requirements-test.txt
python -c "import nltk; nltk.download('punkt'); nltk.download('punkt_tab')"

### Run Baseline Coverage (Task 1)

pytest tests/ -v --cov=inference.pipeline --cov-report=term-missing --cov-report=html:reports/baseline_coverage

### Run Mutation Testing (Task 2)

mutmut run
mutmut results
mutmut html

### Run Final Mutation Testing (Task 4)

mutmut run
mutmut results
mutmut html

### Branch

mutation-testing-assignment

### Folder Structure

radguard-mutation-testing/
├── inference/
│   ├── __init__.py
│   └── pipeline.py          <- target module
├── tests/
│   └── test_pipeline.py     <- 21 test cases
├── reports/
│   ├── baseline_coverage/   <- Task 1 HTML coverage report
│   ├── mutation_baseline/   <- Task 2 HTML mutation report (before)
│   └── mutation_final/      <- Task 4 HTML mutation report (after)
├── conftest.py
├── setup.cfg
├── requirements-test.txt
└── README.md

### Results Summary

- Total Mutants: 280
- Baseline Mutation Score (Task 2): 35.0% (98/280 killed)
- Final Mutation Score (Task 4): 37.1% (104/280 killed)
- Tests: 21 total (18 baseline + 3 new)
