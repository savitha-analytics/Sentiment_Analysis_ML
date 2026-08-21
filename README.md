# Sentiment Analysis using Machine Learning

## 📌 Project Overview

This project is a **Sentiment Analysis system** that classifies text reviews into two categories:

* 😊 **Positive**
* 😞 **Negative**

The model is trained using the provided positive and negative review datasets. The improved version uses **TF-IDF word and character features combined with a Linear Support Vector Machine (SVM)**.

The project demonstrates text preprocessing, feature extraction, machine learning classification, model evaluation, and sentiment prediction.

## 📂 Dataset

The project uses two JSON datasets:

* `positive_data.json` – contains positive reviews.
* `negative_data.json` – contains negative reviews.

Each record contains:

```text
sentence
label
```

For example:

```json
{
    "sentence": "I love this product! It's fantastic.",
    "label": "positive"
}
```

The negative dataset similarly contains sentences labelled `"negative"`. 

## 🛠️ Technologies Used

* Python
* Pandas
* NumPy
* Scikit-learn
* TF-IDF Vectorization
* Linear SVM
* Joblib
* Jupyter Notebook

## 🔄 Workflow

```text
Positive Dataset ──┐
                   ├──> Combine Dataset
Negative Dataset ──┘
                         ↓
                    Train/Test Split
                         ↓
                  TF-IDF Feature Extraction
                         ↓
             Word + Character N-Grams
                         ↓
                    Linear SVM
                         ↓
                    Prediction
                         ↓
              Accuracy & Evaluation
```

## 🤖 Machine Learning Model

The improved model uses:

### 1. TF-IDF

TF-IDF converts text into numerical features based on the importance of words in the dataset.

Two types of features are used:

* **Word n-grams:** `(1,2)`
* **Character n-grams:** `(3,5)`

Using both helps the model capture complete words as well as smaller character patterns.

### 2. Linear SVM

A `LinearSVC` classifier is used for sentiment classification.

```python
LinearSVC(C=2.0)
```

## 📊 Model Performance

The models were compared using **5-fold cross-validation** on the provided dataset.

| Model                                    |   Accuracy |
| ---------------------------------------- | ---------: |
| CountVectorizer + Naive Bayes            |     41.33% |
| TF-IDF + Logistic Regression             |     44.41% |
| TF-IDF + Linear SVM                      |     48.49% |
| **Word + Character TF-IDF + Linear SVM** | **52.05%** |

The improved model achieved approximately **52.05% cross-validation accuracy** on this dataset.

> Note: The dataset is relatively small and contains highly structured/synthetic sentences, so the accuracy should not be treated as representative of performance on real-world reviews.

## 🚀 How to Run

### 1. Clone the repository

```bash
git clone <your-repository-url>
cd sentiment-analysis
```

### 2. Install dependencies

```bash
pip install pandas numpy scikit-learn joblib jupyter
```

### 3. Start Jupyter Notebook

```bash
jupyter notebook
```

Open:

```text
Sentimentanalysis.ipynb
```

### 4. Run the notebook

Make sure these files are in the same project directory:

```text
Sentimentanalysis.ipynb
positive_data.json
negative_data.json
```

## 🔮 Example

```text
Input:
"This product is amazing and the quality is excellent."

Output:
Positive
```

```text
Input:
"The service was terrible and disappointing."

Output:
Negative
```

## 📁 Project Structure

```text
sentiment-analysis/
│
├── Sentimentanalysis.ipynb
├── positive_data.json
├── negative_data.json
├── improved_sentiment_model.joblib
├── README.md
└── requirements.txt
```

## 🔮 Future Improvements

* Use a much larger real-world review dataset.
* Add neutral sentiment classification.
* Perform advanced text preprocessing.
* Tune SVM hyperparameters using GridSearchCV.
* Compare with Random Forest and Logistic Regression.
* Implement BERT/transformer-based sentiment analysis.
* Build a **Streamlit web application** for real-time predictions.
* Add visualization of sentiment distribution and model performance.

## 👩‍💻 Author

**Savitha L.**

B.Tech – Artificial Intelligence & Data Science
