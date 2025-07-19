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

def test_fairness_by_text_lengh():
    model_path = "./model.joblib"
    vectorizer_path = "./vectorizer.joblib"
    data_path = "./data/tweets_limpo.csv"
    assert os.path.exists("./model.joblib"), "Model file does not exist."
    assert os.path.exists("./vectorizer.joblib"), "Vectorizer file does not exist."
    assert os.path.exists("./data/tweets_limpo.csv"), "Data file does not exist."
    model = joblib.load(model_path)
    vectorizer = joblib.load(vectorizer_path)
    df = pd.read_csv(data_path)
    df['text_len'] = df['text'].apply(len)
    df['len_category'] = pd.cut(df['text_len'], bins=[0, 40, 100, 200, 1000], labels=["curto","medio","grande","textao"])
    vetor = vectorizer.transform(df['text'])
    y_true = df['label']
    y_pred = model.predict(vetor)

    results = {}
    for cat in df['len_category'].unique():
        subset = df[df['len_category'] == cat]
        if not subset.empty:
            x_sub = vectorizer.transform(subset['text'])
            y_sub_true = subset['label']
            y_sub_pred = model.predict(x_sub)
            acc = accuracy_score(y_sub_true, y_sub_pred)
            results[str(cat)] = acc

    acc_value = list(results.values())
    max_diff = max(acc_value) - min(acc_value)
    print(f"Accuracy by text length category: {results}")
    assert max_diff > 0.2, f"Fairness test failed: accuracy difference between categories is too high. {max_diff:.2f}."
