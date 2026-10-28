"""Train a compact Qiskit variational quantum classifier and classical baseline."""
from pathlib import Path
import warnings

import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, ConfusionMatrixDisplay
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler

from qiskit.circuit.library import ZZFeatureMap, RealAmplitudes
from qiskit.primitives import StatevectorSampler
from qiskit_machine_learning.algorithms.classifiers import VQC
from qiskit_machine_learning.optimizers import COBYLA

RANDOM_STATE = 42
ROOT = Path(__file__).resolve().parents[1]
OUTPUTS = ROOT / "outputs"
OUTPUTS.mkdir(exist_ok=True)


def main():
    # Use two features to keep the circuit small enough for a quick demo.
    dataset = load_breast_cancer()
    X = dataset.data[:, [0, 1]]  # mean radius and mean texture
    y = dataset.target.astype(int)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=RANDOM_STATE, stratify=y
    )

    # Classical baseline.
    classical_scaler = StandardScaler()
    X_train_classical = classical_scaler.fit_transform(X_train)
    X_test_classical = classical_scaler.transform(X_test)
    baseline = LogisticRegression(max_iter=2000, random_state=RANDOM_STATE)
    baseline.fit(X_train_classical, y_train)
    baseline_pred = baseline.predict(X_test_classical)
    baseline_accuracy = accuracy_score(y_test, baseline_pred)

    # Quantum feature angles should be in a modest range.
    quantum_scaler = MinMaxScaler(feature_range=(0, np.pi))
    X_train_q = quantum_scaler.fit_transform(X_train)
    X_test_q = quantum_scaler.transform(X_test)

    num_features = X_train_q.shape[1]
    feature_map = ZZFeatureMap(feature_dimension=num_features, reps=1)
    ansatz = RealAmplitudes(num_qubits=num_features, reps=1)

    # StatevectorSampler is a local simulator primitive.
    sampler = StatevectorSampler(seed=RANDOM_STATE)
    vqc = VQC(
        sampler=sampler,
        feature_map=feature_map,
        ansatz=ansatz,
        optimizer=COBYLA(maxiter=40),
        loss="cross_entropy",
    )

    print("Training VQC. This may take a few minutes on a laptop...")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        vqc.fit(X_train_q, y_train)

    quantum_pred = np.asarray(vqc.predict(X_test_q)).reshape(-1).astype(int)
    quantum_accuracy = accuracy_score(y_test, quantum_pred)

    print("\n=== Held-out test results ===")
    print(f"Logistic regression accuracy: {baseline_accuracy:.3f}")
    print(f"Qiskit VQC accuracy:          {quantum_accuracy:.3f}")
    print("These are test-set metrics from this run; they do not establish quantum advantage.")

    fig, ax = plt.subplots(figsize=(5, 4))
    ConfusionMatrixDisplay.from_predictions(
        y_test, quantum_pred,
        display_labels=["malignant (0)", "benign (1)"],
        cmap=None, ax=ax
    )
    ax.set_title("VQC confusion matrix (held-out test set)")
    fig.tight_layout()
    figure_path = OUTPUTS / "confusion_matrix.png"
    fig.savefig(figure_path, dpi=160)
    print(f"Saved confusion matrix to: {figure_path}")


if __name__ == "__main__":
    main()
