import streamlit as st
import torch
import torch.nn as nn
import numpy as np
from PIL import Image
import joblib

# Define the MnistModel class (same architecture as used during training)
class MnistModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(784, 10)

    def forward(self, xb):
        xb = xb.reshape(-1, 784)
        out = self.linear(xb)
        return out

def load_models():
    # Load the trained neural network model
    model = MnistModel()
    model.load_state_dict(torch.load("mnist_model.pth"))
    model.eval()

    # Load the trained random forest model
    rf = joblib.load("random_forest_model.joblib")

    return model, rf

def preprocess_image(image):
    image = image.convert('L').resize((28, 28))
    image = np.array(image) / 255.0
    image = image.reshape(1, -1)
    return image

def get_predictions(model, rf, image):
    image_tensor = torch.tensor(image, dtype=torch.float32)
    nn_probs = model(image_tensor).softmax(dim=1).detach().numpy()
    rf_probs = rf.predict_proba(image)
    rf_pred = np.argmax(rf_probs)
    ensemble_probs = (nn_probs + rf_probs) / 2
    ensemble_pred = np.argmax(ensemble_probs)
    return rf_pred, rf_probs

# Load models
model, rf = load_models()

# Streamlit app
st.title("MNIST Digit Classifier")
st.write("Upload an image of a digit to classify.")

uploaded_file = st.file_uploader("Choose an image...", type="png")

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded Image.', use_column_width=True)
    st.write("")
    st.write("Classifying...")

    image = preprocess_image(image)
    rf_pred, rf_probs = get_predictions(model, rf, image)

    st.write(f"Predicted Digit: {rf_pred}")
    st.write("Prediction Probabilities:")
    st.write(rf_probs)

