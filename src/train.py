"""Script to train and save the iris classifier model."""
from .model import train_model
from .utils import save_model


def main():
    print("Training iris classifier...")
    model, accuracy = train_model()
    print(f"Training accuracy: {accuracy:.4f}")

    model_path = "model/iris_model.pkl"
    save_model(model, model_path)
    print(f"Model saved to {model_path}")


if __name__ == "__main__":
    main()
