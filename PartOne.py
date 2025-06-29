# Re-assessment template 2025
import nltk
import spacy
import pickle
import math
import pandas as pd
from pathlib import Path
from collections import Counter
from nltk.tokenize import word_tokenize, sent_tokenize

# Download required NLTK resources if needed
nltk.download('punkt')
nltk.download('cmudict')

# Load CMU dictionary once
cmudict = nltk.corpus.cmudict.dict()

# Load spaCy model
nlp = spacy.load("en_core_web_sm")
nlp.max_length = 2000000

def count_syl(word, d):
    """Counts the number of syllables in a word given a dictionary of syllables per word.
    If the word is not in the dictionary, syllables are estimated by counting vowel clusters."""
    word = word.lower()
    if word in d:
        return len([ph for ph in d[word][0] if ph[-1].isdigit()])
    else:
        return 1  # fallback


def fk_level(text, d):
    """Returns the Flesch-Kincaid Grade Level of a text (higher grade is more difficult)."""
    sents = sent_tokenize(text)
    tokens = word_tokenize(text)
    words = [w for w in tokens if w.isalpha()]
    syllables = sum(count_syl(w, d) for w in words)
    if len(sents) > 0 and len(words) > 0:
        score = 0.39 * (len(words) / len(sents)) + 11.8 * (syllables / len(words)) - 15.59
        return round(score, 4)
    else:
        return 0


def read_novels(path=Path.cwd() / "p1-texts" / "novels"):
    """Reads texts from a directory of .txt files and returns a DataFrame with the text, title,
    author, and year"""
    data = []
    for file in path.glob("*.txt"):
        parts = file.stem.split('-')
        title = parts[0].replace('_', ' ')
        author = parts[1].replace('_', ' ')
        year = int(parts[2])
        text = file.read_text(encoding="utf-8")
        data.append({
            'text': text,
            'title': title,
            'author': author,
            'year': year
        })
    df = pd.DataFrame(data)
    df = df.sort_values(by='year').reset_index(drop=True)
    return df


def parse(df, store_path=Path.cwd() / "pickles", out_name="parsed.pickle"):
    """Parses the text of a DataFrame using spaCy, stores the parsed docs as a column and writes 
    the resulting DataFrame to a pickle file"""
    df['parsed'] = df['text'].apply(lambda txt: nlp(txt))
    store_path.mkdir(exist_ok=True)
    with open(store_path / out_name, 'wb') as f:
        pickle.dump(df, f)


def nltk_ttr(text):
    """Calculates the type-token ratio of a text. Text is tokenized using nltk.word_tokenize."""
    tokens = word_tokenize(text.lower())
    words = [w for w in tokens if w.isalpha()]
    return round(len(set(words)) / len(words), 4) if len(words) > 0 else 0


def get_ttrs(df):
    """Helper function to add TTR to a dataframe"""
    results = {}
    for i, row in df.iterrows():
        results[row["title"]] = nltk_ttr(row["text"])
    return results


def get_fks(df):
    """Helper function to add FK scores to a dataframe"""
    results = {}
    for i, row in df.iterrows():
        results[row["title"]] = fk_level(row["text"], cmudict)
    return results


def subjects_by_verb_pmi(doc, target_verb):
    """Extracts the PMI scores for subjects of a given verb in a parsed document."""
    subj_freq = Counter()
    co_occur = Counter()
    total_tokens = len(doc)

    for tok in doc:
        if tok.dep_ == "nsubj":
            subj = tok.text.lower()
            subj_freq[subj] += 1
            if tok.head.lemma_ == target_verb:
                co_occur[subj] += 1

    total_subjs = sum(subj_freq.values())
    total_targets = sum(co_occur.values())

    pmi_scores = {}
    for subj in co_occur:
        p_subj = subj_freq[subj] / total_subjs
        p_target = total_targets / total_tokens
        p_joint = co_occur[subj] / total_tokens
        if p_subj > 0 and p_target > 0:
            pmi_scores[subj] = round(math.log2(p_joint / (p_subj * p_target)), 4)

    return sorted(pmi_scores.items(), key=lambda x: x[1], reverse=True)[:10]


def subjects_by_verb_count(doc, verb):
    """Extracts the most common subjects of a given verb in a parsed document."""
    return Counter([
        tok.text.lower()
        for tok in doc
        if tok.dep_ == "nsubj" and tok.head.lemma_ == verb
    ]).most_common(10)


def adjective_counts(doc):
    """Extracts the most common adjectives in a parsed document."""
    return Counter([
        tok.text.lower()
        for tok in doc
        if tok.pos_ == "ADJ"
    ]).most_common(10)


if __name__ == "__main__":
    
    path = Path.cwd() / "p1-texts" / "novels"
    df = read_novels(path)
    print(df.head())

    parse(df)  # saves to pickles/parsed.pickle

    print("TTRs:", get_ttrs(df))
    print("FK Scores:", get_fks(df))

    # Load parsed
    with open(Path.cwd() / "pickles" / "parsed.pickle", 'rb') as f:
        df = pickle.load(f)

    for i, row in df.iterrows():
        print(f"\n{row['title']}")
        print("Subjects by verb count (hear):", subjects_by_verb_count(row['parsed'], "hear"))
        print("Subjects by verb PMI (hear):", subjects_by_verb_pmi(row['parsed'], "hear"))
        print("Adjective counts:", adjective_counts(row['parsed']))
