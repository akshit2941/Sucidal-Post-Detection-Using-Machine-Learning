from fastapi import FastAPI
from pydantic import BaseModel
import pickle
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
import streamlit as st
import uvicorn
import sys

# Load tokenizer
with open('tokenizer.pickle', 'rb') as handle:
    tokenizer = pickle.load(handle)

# Load pre-trained models
model1 = load_model("best_model.keras")
model2 = load_model("Model_Basic_GAP.keras")
model3 = load_model("Model_Embedding_Conv1D.keras")

# Function to classify predictions
def classify_prediction(prediction):
    if prediction[0][0] > 0.5:
        return "Suicidal Post Detected"
    else:
        return "Non-Suicidal Thought Detected"

# FastAPI setup
app = FastAPI()

# Input data model for FastAPI
class TextInput(BaseModel):
    text: str

# FastAPI prediction endpoint
@app.post("/predict")
async def predict(input: TextInput):
    # Preprocess the input text
    sequences = tokenizer.texts_to_sequences([input.text])
    padded_sequences = pad_sequences(sequences, maxlen=100)  # Adjust maxlen as needed

    # Get predictions from all three models
    prediction1 = model1.predict(padded_sequences).tolist()
    prediction2 = model2.predict(padded_sequences).tolist()
    prediction3 = model3.predict(padded_sequences).tolist()

    # Classify predictions
    classification1 = classify_prediction(prediction1)
    classification2 = classify_prediction(prediction2)
    classification3 = classify_prediction(prediction3)

    # Return the predictions and classifications as JSON
    return {
        "model1": {
            "raw_prediction": prediction1,
            "classification": classification1
        },
        "model2": {
            "raw_prediction": prediction2,
            "classification": classification2
        },
        "model3": {
            "raw_prediction": prediction3,
            "classification": classification3
        },
    }

# Streamlit UI for interaction
def run_streamlit():
    st.title("Text Prediction App")
    st.write("Enter text below to get predictions from three models.")

    # Input field for text
    input_text = st.text_area("Input Text", placeholder="Type your text here...")

    if st.button("Predict"):
        if input_text.strip():
            # Preprocess the input text
            sequences = tokenizer.texts_to_sequences([input_text])
            padded_sequences = pad_sequences(sequences, maxlen=100)  # Adjust maxlen as needed

            # Get predictions from all three models
            prediction1 = model1.predict(padded_sequences)
            prediction2 = model2.predict(padded_sequences)
            prediction3 = model3.predict(padded_sequences)

            # Classify predictions
            classification1 = classify_prediction(prediction1)
            classification2 = classify_prediction(prediction2)
            classification3 = classify_prediction(prediction3)

            # Display predictions and classifications
            st.success("Predictions:")
            st.write(f"Best Model Prediction: {prediction1[0][0]} - {classification1}")
            st.write(f"GAP Model Prediction: {prediction2[0][0]} - {classification2}")
            st.write(f"Conv1D Model Prediction: {prediction3[0][0]} - {classification3}")
        else:
            st.warning("Please enter some text before clicking Predict.")

# Main entry point to manage FastAPI or Streamlit
if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "api":
        # Run FastAPI app
        uvicorn.run(app, host="127.0.0.1", port=8000)
    else:
        # Run Streamlit app
        run_streamlit()
