# Amazon Fine Food Reviews Sentiment Analysis

This project performs sentiment analysis on the **Amazon Fine Food Reviews dataset** from Kaggle. The goal is to build various machine learning and deep learning models to classify reviews as positive or negative and identify the best performing approach.

---

## 📁 **Dataset**

- **Source:** [Kaggle Amazon Fine Food Reviews](https://www.kaggle.com/code/robikscube/sentiment-analysis-python-youtube-tutorial/input)
- **Columns Used:**  
  - `Text` (Review text)  
  - `Score` (Used to derive Sentiment: Positive if Score > 3, Negative if Score < 3)

---

## 🔧 **Dependencies**

Install required libraries:

pip install pandas numpy matplotlib seaborn scikit-learn nltk wordcloud xgboost keras tensorflow transformers

## 💻 **Machine Learning Algorithms Used**

- **Logistic Regression**
- **Multinomial Naive Bayes**
- **Support Vector Machine (SVM)**
- **Random Forest Classifier**
- **XGBoost Classifier**

---

## 🤖 Deep Learning Models Used

- **Feedforward Neural Network (Dense layers with Keras)**
- **Recurrent Neural Network (LSTM with Keras)**
- **Convolutional Neural Network (1D CNN with Keras)**
- **Hybrid CNN + LSTM model**
---

## 🛠️ Feature Engineering

- **TF-IDF Vectorization:** Used to convert cleaned text into numerical feature vectors for classical ML models.
- **Word Embeddings:** Used Embedding layers for deep learning models like LSTM and CNN to capture semantic meaning.

---

## 🏆 Best Performing Model

- **Model:** LSTM-CNN Hybrid Model
- **Performance:**
  - **Accuracy:** 93.9%
  - **Precision:** 94.9%
  - **Recall:** 98%
  - **F1 Score:** 96.4%
  - **False Negative Rate:** 0.0195


---

## 📊 Exploratory Data Analysis

- Review length distribution
- Most frequent words in positive and negative reviews
- WordCloud visualizations
- Time-based sentiment trends

---

## 🚀 How to Run

1. Clone the repository
2. Install dependencies using pip
3. Run the notebook step-by-step to train and evaluate models

---

## 💡 Future Work

- Hyperparameter tuning for deep learning models
- Deployment as a Flask API or Streamlit web app
- Integration with MLflow for experiment tracking
