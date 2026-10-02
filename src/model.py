from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score


IRIS_CLASSES = ["setosa", "versicolor", "virginica"]


def train_model():
    """Train a simple iris classifier and return the model."""
    iris = load_iris()
    X, y = iris.data, iris.target

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    model = RandomForestClassifier(n_estimators=10, random_state=42)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)

    return model, accuracy


def predict(model, features):
    """Make a prediction given a trained model and features."""
    prediction = model.predict(features)[0]
    probability = model.predict_proba(features)[0]

    return {
        "class": IRIS_CLASSES[prediction],
        "class_id": int(prediction),
        "probabilities": {
            IRIS_CLASSES[i]: float(prob)
            for i, prob in enumerate(probability)
        }
    }
