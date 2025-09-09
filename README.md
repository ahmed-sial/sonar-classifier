# 🌊 Sonar Dataset Classifier

A Streamlit web app that classifies sonar signals as either Rock (M) or Mine (R) using a trained machine learning pipeline.
Users can upload CSV files with sonar data (60 features per row, with or without labels) to get real-time predictions, visualizations, and downloadable results.

## ✨ Features

- 📂 Upload sonar dataset (CSV format, 60–61 columns)

- ✅ Auto-detects labels if present in the dataset

- 🤖 Predicts whether signals correspond to Rock (M) or Mine (R)

- 💾 Download predictions as CSV

- 🌐 Web-based interface powered by Streamlit

## 🛠️ Tech Stack

Frontend/Interface: Streamlit

Data Processing: Pandas, NumPy

Machine Learning: Scikit-learn (trained model saved with Joblib)

## 🚀 Getting Started

1. Clone the repository
```bash
git clone https://github.com/your-username/sonar-classifier.git
cd sonar-classifier
```
2. Install dependencies

Make sure you have Python 3.9+ installed. Then install requirements:
```bash
pip install -r requirements.txt
```
3. Run the app
```bash
streamlit run app.py
```
The app will start locally, usually at:
http://localhost:8501

## Demonstration

[Click Here](https://sonar-classifier.streamlit.app/)
