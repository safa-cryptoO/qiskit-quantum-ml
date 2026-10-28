# Quantum Machine Learning: Variational Quantum Classifier

A small, reproducible quantum machine-learning demo using Qiskit. The project trains a variational quantum circuit to classify a two-class dataset and compares its test accuracy with a classical baseline.

> **Important:** This is a starter project, not a claim of quantum advantage. Run the notebook/script and report the results actually produced before submitting. Confirm that this project matches the official KUQCI Qiskit Fall Fest '26 challenge requirements.

## Project overview

- **Task:** binary classification
- **Dataset:** scikit-learn Breast Cancer Wisconsin dataset, reduced to two classes and two features for a compact demonstration
- **Quantum model:** angle-encoded features + parameterized `ZZFeatureMap` / `RealAmplitudes` circuit via Qiskit Machine Learning's `VQC`
- **Baseline:** scikit-learn logistic regression
- **Evaluation:** held-out test accuracy and confusion matrix

## Setup

Python 3.10 or 3.11 is recommended.

```bash
python -m venv .venv
# macOS/Linux:
source .venv/bin/activate
# Windows:
# .venv\\Scripts\\activate
pip install -r requirements.txt
```

## Run

```bash
python src/train.py
```

The script prints test accuracy and writes a confusion-matrix image to `outputs/confusion_matrix.png` (after you run it). Results are not included in advance because they should come from your actual run.

## Repository layout

```text
.
├── README.md
├── requirements.txt
├── src/
│   └── train.py
└── outputs/
    └── .gitkeep
```

## Method

1. Load the dataset and select two numeric features.
2. Split into training and test sets with a fixed random seed.
3. Standardize features and map them to a range suitable for rotation gates.
4. Train a classical logistic-regression baseline.
5. Train a variational quantum classifier using a Qiskit simulator.
6. Compare held-out accuracy and inspect the confusion matrix.

## Limitations

- A simulator is used; this is not a demonstration on quantum hardware.
- The feature subset and small circuit are chosen for a compact experiment, not to maximize medical prediction quality.
- Small datasets and simulator experiments do not establish quantum advantage.
- This dataset is for educational experimentation only, not for medical diagnosis.

## Before submitting

- Run the script successfully in a clean environment.
- Add your team members and explain each person's contribution if the competition requires it.
- Include the actual console output and discuss the results honestly.
- Check the official track brief and judging rubric; adapt the task if a specific challenge dataset or method is required.
- Do not claim results that you did not run or verify.
