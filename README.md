# Suicidal Ideation Detection Using Deep Learning Models

This project is designed to detect suicidal ideation in text data using Natural Language Processing (NLP) and Deep Learning models. It leverages pre-trained models and a user-friendly application interface for seamless interaction and predictions.

---

## Project Structure

The project is organized into the following key components:

1. **`Keras Model/`**

   - Contains saved Keras models used for training and evaluation.

2. **`Pre-Trained Model/`**

   - Houses pre-trained models with high accuracy for inference.

3. **Jupyter Notebooks (`.ipynb` files)**

   - Used for training, evaluation, and experimentation with different architectures.

4. **`app.py`**
   - The main application file integrating the prediction system via Streamlit (UI) and FastAPI (API).

---

## Features

- Implements various deep learning architectures, including Bidirectional LSTMs and Convolutional Neural Networks (CNN).
- Provides predictions from three pre-trained models for a given input.
- Classifies input as either:
  - **"Suicidal Post Detected"**
  - **"Non-Suicidal Thought Detected"**

---

## Dataset

The dataset used for this project is sourced from Kaggle: [SuicideWatch Dataset](https://www.kaggle.com/datasets/nikhileswarkomati/suicide-watch).

- It includes posts from "SuicideWatch" and "Depression" subreddits collected via the Pushshift API.
- Data is preprocessed to remove symbols, emojis, numbers, and special characters, followed by stopword removal and stemming using NLTK.

---

## Preprocessing Steps

1. Convert text to lowercase.
2. Remove unwanted characters, emojis, and numbers.
3. Remove stopwords using the NLTK library.
4. Apply stemming for converting words to their base forms.

---

## Training Process

1. **Data Splitting:**  
   The Dataset is split into 80% training and 20% validation data.
2. **Feature Extraction:**

   - Tokenization using TensorFlow's `Tokenizer()`.
   - Sequence padding to standardize input lengths.

3. **Model Training:**  
   Multiple deep learning models are trained to classify text into suicidal and non-suicidal categories.

---

## Deep Learning Models

| No  | Model Description                                    | Total Params | Accuracy (Validation) |
| --- | ---------------------------------------------------- | ------------ | --------------------- |
| 1   | Word Embedding + Bidirectional LSTM                  | 107,137      | 92.56%                |
| 2   | Word Embedding + Convolutional Neural Networks (CNN) | 220,033      | 94.52%                |
| 3   | Word Embedding + Global Average Pooling              | 195,329      | 93.53%                |

---

## Usage

### 1. Install Dependencies

Ensure you have the required Python packages installed:

```bash
pip install -r requirements.txt
```

### 2. Run the Application

The project provides two interfaces:

- **Streamlit (Frontend):**

  ```bash
  streamlit run .\app.py
  ```

  Opens a web-based interface where you can input text and view predictions.

- **FastAPI (API Backend):**
  ```bash
  python app.py api
  ```
  Starts a FastAPI server for handling prediction requests.

---

## Example Predictions

### Input:

```text
"I feel like giving up. There is no point in continuing anymore."
```

### Outputs:

| Model                 | Prediction Score | Classification                |
| --------------------- | ---------------- | ----------------------------- |
| Word Embedding + LSTM | 0.85             | Suicidal Post Detected        |
| Word Embedding + CNN  | 0.72             | Suicidal Post Detected        |
| Word Embedding + GAP  | 0.45             | Non-Suicidal Thought Detected |

---

## How to Contribute

1. Fork this repository.
2. Create a new branch for your feature/bug fix.
3. Commit your changes and open a pull request.

---

## Acknowledgments

This project utilizes the SuicideWatch dataset, which is a collection of posts from Reddit. Special thanks to the authors of this dataset and the Kaggle platform for hosting it.

---

## License

This project is licensed under the MIT License. Feel free to use, modify, and distribute this software as per the license terms.
