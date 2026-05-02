# RadGuard Mutation Testing Assignment

## CS-4006 Software Testing — FAST NUCES (Spring 2026)

### Final Year Project: RadGuard — AI-Based Radiology Report Error Detection

---

## Overview

This repository contains the mutation testing assignment for the RadGuard FYP.

The focus is on evaluating the robustness of the inference logic using:
- Code coverage analysis
- Mutation testing (mutmut)

**Target Module:**
`inference/pipeline.py` — Implements ELRR scoring and verdict logic.

---

## Installation

Install required dependencies:

```bash
pip install -r requirements-test.txt
```
Download required NLTK datasets:

```bash
python -c "import nltk; nltk.download('punkt'); nltk.download('punkt_tab')"
```
## Running Baseline Coverage (Task 1)

Run pytest with coverage reporting:
```bash
pytest tests/ -v \
  --cov=inference.pipeline \
  --cov-report=term-missing \
  --cov-report=html:reports/baseline_coverage
```
Generated report will be available in:
```bash
reports/baseline_coverage/index.html
```
## Running Mutation Testing (Task 2)

Run mutation testing using mutmut:
```bash
mutmut run
mutmut results
mutmut html
```
Final report will reflect improved mutation score.

## Git Branch
mutation-testing-assignment

## Project Structure

radguard-mutation-testing/
├── inference/
│   ├── __init__.py
│   └── pipeline.py          # Target module (ELRR logic)
├── tests/
│   └── test_pipeline.py     # 21 test cases
├── reports/
│   ├── baseline_coverage/   # Task 1 coverage report
│   ├── mutation_baseline/   # Task 2 mutation report (before)
│   └── mutation_final/      # Task 4 mutation report (after)
├── conftest.py
├── setup.cfg
├── requirements-test.txt
└── README.md

## Results Summary

- Total Mutants: 280
- Baseline Mutation Score (Task 2): 35.0% (98/280 killed)
- Final Mutation Score (Task 4): 37.1% (104/280 killed)
- Tests: 21 total (18 baseline + 3 new)

## Tools Used
- pytest (testing framework)
- pytest-cov (coverage analysis)
- mutmut (mutation testing)
- NLTK (text processing utilities)

## Notes
- Focus was on improving test effectiveness, not just coverage
- Additional test cases were added to kill more mutants
- Mutation score improvement demonstrates stronger test suite quality
