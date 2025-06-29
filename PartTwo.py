import pandas as pd
import re
import nltk
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import classification_report, f1_score
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords

# Download NLTK resources
nltk.download('punkt')
nltk.download('stopwords')

def preprocess_data(csv_path):
    """Step (a): Load and clean hansard40000.csv"""
    df = pd.read_csv(csv_path)
    df['party'] = df['party'].replace('Labour (Co-op)', 'Labour')
    top_parties = df['party'].value_counts().nlargest(4).index.tolist()
    df = df[df['party'].isin(top_parties)]
    df = df[df['party'] != 'Speaker']
    df = df[df['speech_class'] == 'Speech']
    df = df[df['speech'].str.len() >= 1000]
    print("\n Final shape of dataset:", df.shape)
    return df

def vectorize_split(df, ngram_range=(1,1), tokenizer=None):
    """Step (b): TF-IDF vectorization and split"""
    vectorizer = TfidfVectorizer(stop_words='english', max_features=3000,
                                 ngram_range=ngram_range, tokenizer=tokenizer)
    X = vectorizer.fit_transform(df['speech'])
    y = df['party']
    return train_test_split(X, y, stratify=y, random_state=26)

def train_models(X_train, X_test, y_train, y_test, label=""):
    """Step (c)+(d)+(e): Train RF and SVC and print results"""
    rf = RandomForestClassifier(n_estimators=300, random_state=26)
    rf.fit(X_train, y_train)
    y_pred_rf = rf.predict(X_test)
    print(f"\n Random Forest {label} Macro F1:", round(f1_score(y_test, y_pred_rf, average='macro'), 4))
    print("RF Report:\n", classification_report(y_test, y_pred_rf))

    svc = SVC(kernel='linear', random_state=26)
    svc.fit(X_train, y_train)
    y_pred_svc = svc.predict(X_test)
    print(f"\n SVC {label} Macro F1:", round(f1_score(y_test, y_pred_svc, average='macro'), 4))
    print("SVC Report:\n", classification_report(y_test, y_pred_svc))

def custom_tokenizer(text):
    """Step (e): Custom tokenizer"""
    stop_words = set(stopwords.words('english'))
    text = re.sub(r"[^\w\s]", " ", text.lower())
    tokens = word_tokenize(text)
    return [t for t in tokens if t not in stop_words and len(t) > 2]

if __name__ == "__main__":
    # === Step (a) ===
    df = preprocess_data("p2-texts/hansard40000.csv")

    # === Step (b)+(c): TF-IDF Unigrams ===
    X_train, X_test, y_train, y_test = vectorize_split(df)
    train_models(X_train, X_test, y_train, y_test, label="")

    # === Step (d): TF-IDF n-grams ===
    X_train_ng, X_test_ng, y_train_ng, y_test_ng = vectorize_split(df, ngram_range=(1,3))
    train_models(X_train_ng, X_test_ng, y_train_ng, y_test_ng, label="+ n-grams")

    # === Step (e): Custom Tokenizer ===
    X_train_c, X_test_c, y_train_c, y_test_c = vectorize_split(df, tokenizer=custom_tokenizer)
    svc_c = SVC(kernel='linear', random_state=26)
    svc_c.fit(X_train_c, y_train_c)
    y_pred_c = svc_c.predict(X_test_c)
    print("\n Final Custom Tokenizer Model (SVC)")
    print("SVC Report:\n", classification_report(y_test_c, y_pred_c))
