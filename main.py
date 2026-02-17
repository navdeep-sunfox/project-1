import os
import yaml
import tensorflow as tf
import mlflow
import mlflow.tensorflow
from datetime import datetime

# Set experiment (optional but good)
mlflow.set_experiment("mnist-tf-experiment")

# Load config
with open("config.yaml", "r") as f:
    config = yaml.safe_load(f)

run_name = f"{datetime.now().strftime('%Y%m%d_%H%M')}"
# Start MLflow run
with mlflow.start_run(run_name=run_name):

    # Log parameters
    mlflow.log_param("hidden_units", config["model"]["hidden_units"])
    mlflow.log_param("optimizer", config["model"]["optimizer"])
    mlflow.log_param("loss", config["model"]["loss"])
    mlflow.log_param("epochs", config["training"]["epochs"])
    mlflow.log_param("batch_size", config["training"]["batch_size"])
    mlflow.log_param("normalize", config["data"]["normalize"])

    # Load MNIST
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

    # Normalize
    if config["data"]["normalize"]:
        x_train = x_train / 255.0
        x_test = x_test / 255.0

    # Build model
    model = tf.keras.models.Sequential([
        tf.keras.layers.Flatten(input_shape=(28, 28)),
        tf.keras.layers.Dense(
            config["model"]["hidden_units"],
            activation="relu"
        ),
        tf.keras.layers.Dense(10, activation="softmax")
    ])

    # Compile
    model.compile(
        optimizer=config["model"]["optimizer"],
        loss=config["model"]["loss"],
        metrics=["accuracy"]
    )

    # Train
    history = model.fit(
        x_train,
        y_train,
        epochs=config["training"]["epochs"],
        batch_size=config["training"]["batch_size"],
        validation_split=0.1
    )

    # Evaluate
    test_loss, test_acc = model.evaluate(x_test, y_test)

    print("Accuracy:", test_acc)

    # Log metrics
    mlflow.log_metric("test_loss", test_loss)
    mlflow.log_metric("test_accuracy", test_acc)

    # Log training metrics per epoch
    for i, acc in enumerate(history.history["accuracy"]):
        mlflow.log_metric("train_accuracy", acc, step=i)

    # Save model
    os.makedirs(config["paths"]["model_dir"], exist_ok=True)
    model_path = os.path.join(
        config["paths"]["model_dir"],
        config["paths"]["model_name"]
    )

    model.save(model_path)

    # Log model to MLflow
    mlflow.tensorflow.log_model(model,artifact_path="model")
