# src/train.py

import argparse
import os
import joblib
import matplotlib.pyplot as plt

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay


def main(test_size, random_state):
    # Load data
    iris = load_iris()
    X = iris.data
    y = iris.target

    # Train / test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )

    # Train model
    model = DecisionTreeClassifier(random_state=random_state)
    model.fit(X_train, y_train)

    # Predictions
    y_pred = model.predict(X_test)

    # Accuracy
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Accuracy: {accuracy:.2f}")

    # Save outputs
    os.makedirs("outputs", exist_ok=True)

    cm = confusion_matrix(y_test, y_pred)
    disp = ConfusionMatrixDisplay(
        confusion_matrix=cm,
        display_labels=iris.target_names
    )
    disp.plot()
    plt.savefig("outputs/confusion_matrix.png")
    plt.close()

    joblib.dump(model, "outputs/iris_model.joblib")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Iris Decision Tree Classifier")
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--random-state", type=int, default=42)

    args = parser.parse_args()
    main(args.test_size, args.random_state)
