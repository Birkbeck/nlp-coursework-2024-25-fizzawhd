# NLP Coursework Complete Report (Parts 1 & 2)

## Full Name (as on student record): Fizza Waheed
## Student number: 14111888


## Academic Declaration

I have read and understood the sections of plagiarism in the College Policy on assessment offences and confirm that the work is my own, with the work of others clearly acknowledged. I give my permission to submit my report to the plagiarism testing database that the College is using and test it using plagiarism detection software, search engines or meta-searching software.
[![Review Assignment Due Date](https://classroom.github.com/assets/deadline-readme-button-22041afd0340ce965d47ae6ef1cefeee28c7c493a6346c4f15d667ab976d596c.svg)](https://classroom.github.com/a/8qgh5WxD)

## Requirements
pip install pandas nltk scikit-learn matplotlib seaborn
nltk.download('punkt')
nltk.download('stopwords')
pip install nltk spacy pandas
python -m spacy download en_core_web_sm

# NLP Assignment Part One  
*Analysis of 19th Century Novels using Python, NLTK, and spaCy*

## Overview

The project employs Natural Language Processing (NLP) on a set of 13 novels, utilizing a range of linguistic analysis methods. It also includes records on token statistics, readability scores, syntactic patterns, and caches intermediate-parsed data for reuse.


## Part 1 Tasks Covered

### Step (a): Reading Novels  
Reads and structures `.txt` novel files into a sorted Pandas DataFrame with columns:
`text` (full novel)
`title`
`author`
`year`

### Step (b): Type-Token Ratio (TTR)  
Uses `nltk.tokenize` to compute the **lexical richness** of each novel.  
Formula: `TTR = unique words / total words`

### Step (c): Flesch-Kincaid Grade Level  
Calculates readability score using `nltk.corpus.cmudict`.  
Formula:  

0.39 * (total words / total sentences) + 
11.8 * (syllables / total words) 15.59


### Step (d): Explanation (answers.txt)  
Explains limitations of the Flesch-Kincaid method in modern NLP.  
See `answers.txt` for details.

### Step (e): spaCy Parsing + Pickle  
Parses novels with `spaCy (en_core_web_sm)` and saves full results as a Pickle file to avoid reprocessing.  
File: `parsed_novels.pkl`

### Step (f): Syntactic Analysis  
From parsed documents:
Top 10 direct objects (`dobj`)
Top 10 subjects of the verb **hear**
PMI scores of subjects of **hear**


# NLP Coursework Part 2: Lexical and Classification Analysis

In the present script, the given UK Parliamentary speeches (`hansard40000.csv`) are analyzed using preprocessing, TF-IDF vectors, and classification. The objective is to categorize the party affiliations through the words of the speech by various NLP and ML applications.


## Part 2 Breakdown

### Step (a): Preprocess Dataset
Read the `hansard40000.csv` file into a DataFrame.
Standardize the `party` column:
  `'Labour (Co-op)'` is replaced with `'Labour'`
  `'Speaker'` entries are dropped
Keep only the 4 most common parties.
Filter rows where `speech_class == 'Speech'`
Remove speeches with fewer than 1000 characters.
Final dataset shape: `(7815, 8)`

### Step (b): TF-IDF Vectorisation
Applied `TfidfVectorizer` from `sklearn`:
  `stop_words='english'`
  `max_features=3000`
Split data into training/testing using:

### Step (c): Train Classifiers
Trained models:
  `RandomForestClassifier(n_estimators=300)`
  `SVC(kernel='linear')`
Evaluation:
  Macro F1-score
  Classification report

### Step (d): Add n-grams
Updated `TfidfVectorizer`:
  `ngram_range=(1, 3)`
  `max_features=3000`
Re-trained both classifiers.
Notable improvement in SVC macro F1.

### Step (e): Custom Tokenizer
Custom tokenizer uses:
  Lowercasing
  Regex-based punctuation removal
  `nltk.word_tokenize`
  Stopword removal
  Short word filtering (length ≤ 2)
Passed via `tokenizer=...` in TfidfVectorizer.
Final report printed **only for the best model (SVC)**.

### Step (f): Tokenizer Explanation
See `answers.txt` for details.





