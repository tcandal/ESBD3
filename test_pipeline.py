import pandas as pd
import joblib
import os
from sklearn.metrics import accuracy_score

def test_model_files_exist():
    assert os.path.exists("./model.joblib"), "Model file does not exist."
    assert os.path.exists("vectorizer.joblib"), "Vectorizer file does not exist."

def test_vectorizer_output_shape():
    vectorizer = joblib.load("./vectorizer.joblib")
    sample = ["O produto e ruim."]
    vetor = vectorizer.transform(sample)
    assert vetor.shape[0] == 1, "Vectorized output should have one row for the sample text."

def test_vectorizer_output_shape_neutro():
    vectorizer = joblib.load("vectorizer.joblib")
    sample = ["O produto e ruim."]
    vetor = vectorizer.transform(sample)
    assert vetor.shape[0] == 1, "Vectorized output should have one row for the sample text."

def test_model_prediction_lablels():
    model = joblib.load("./model.joblib")
    vectorizer = joblib.load("vectorizer.joblib")
    sample = ["O produto e ruim."]
    vetor = vectorizer.transform(sample)
    pred = model.predict(vetor)[0]
    assert pred in ['positivo','negativo'], f"Vectorized output should have one row for the sample text {pred}."

def test_data_validation():
    df = pd.read_csv("./data/tweets_limpo.csv")
    assert 'text' in df.columns, "DataFrame should contain 'text' column."
    assert 'label' in df.columns, "DataFrame should contain 'label' column."
    assert not df.empty, "DataFrame should not be empty."
    assert df['label'].isin(["positivo","negativo"]).all(), "All text entries should be non-null."

#def test_placeholder():
    # Placeholder test to ensure pytest runs
#    assert True