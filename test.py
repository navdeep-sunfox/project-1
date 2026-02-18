import os
import yaml
import tensorflow as tf
from flask import Flask, request, jsonify
import numpy as np

app = Flask(__name__)


# Load config
with open("config.yaml", "r") as f:
    config = yaml.safe_load(f)


# Load MNIST test data
(_, _), (x_test, y_test) = tf.keras.datasets.mnist.load_data()


# Normalize (same as training)
if config["data"]["normalize"]:
    x_test = x_test / 255.0


# Load trained model
model_path = os.path.join(config["paths"]["model_dir"],config["paths"]["model_name"])

print("Loading model from:", model_path)

model = tf.keras.models.load_model(model_path)

@app.route("/")
def home():
    return "MNIST Model Running"


# # Print results
# print("=" * 40)
# print("Test Loss    :", test_loss)
# print("Test Accuracy:", test_acc)
# print("=" * 40)


@app.route("/predict", methods=["POST"])
def predict():


    test_loss, test_acc = model.evaluate(x_test, y_test, verbose=1)

    return jsonify({"prediction": f"test loss {test_loss}, test accuracy {test_acc}"})


if __name__ == "__main__":

    port = int(os.environ.get("PORT", 8080))

    app.run(host="0.0.0.0", port=port)
