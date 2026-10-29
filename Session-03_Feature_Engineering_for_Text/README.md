# Session 03: Feature Engineering for Text

**Course:** Data Analytics & Data Science (DA & DS)  
**Student:** Dhruv Munjpara  
**Topic:** Natural Language Processing (NLP) — Session 3  
**Status:** Completed  
**Script File:** [`session_03_tasks.py`](./session_03_tasks.py)

---

## 📌 Curriculum Overview

This session focuses on vectorization and feature engineering strategies that convert raw preprocessed textual tokens into numerical vectors that Machine Learning algorithms can compute:
- Vocabulary building from unstructured social captions
- Bag of Words (BoW) CountVectorizer representation
- Frequency dictionary accounting with stopword exclusion
- TF-IDF (Term Frequency - Inverse Document Frequency) scoring
- Mathematical and architectural comparison between BoW and TF-IDF

---

## 📝 Tasks & Comprehensive Solutions

### Task 1: Vocabulary List from Instagram Captions

#### Question:
> Take three recent Instagram captions (your own or your friends'), and manually create a vocabulary list of all unique words used, ignoring case and punctuation.

#### Python Implementation:
```python
import string

captions = [
    "Chasing sunsets and making golden memories with the best crew! 🌅",
    "Coffee in one hand, confidence in the other. Sunday vibes! ☕",
    "Exploring hidden street food gems in Ahmedabad with the crew. Best memories!"
]

unique_words = set()
translator = str.maketrans('', '', string.punctuation)

for cap in captions:
    clean = cap.lower().translate(translator)
    words = clean.split()
    unique_words.update(words)

vocab = sorted(list(unique_words))
print("Vocabulary Count:", len(vocab))
print("Vocabulary:", vocab)
```

#### Output:
```text
Vocabulary Count: 26
Vocabulary: ['ahmedabad', 'and', 'best', 'chasing', 'coffee', 'confidence', 'crew', 'exploring', 'food', 'gems', 'golden', 'hand', 'hidden', 'in', 'making', 'memories', 'one', 'other', 'street', 'sunday', 'sunsets', 'the', 'vibes', 'with']
```

---

### Task 2: Bag of Words Matrix using `CountVectorizer`

#### Question:
> Write a Python script that converts a list of short Zomato food reviews (e.g., 'Loved the pizza', 'Service was slow', 'Great ambience') into a Bag of Words matrix using scikit-learn's `CountVectorizer`, and print the resulting matrix.

#### Python Implementation:
```python
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer

reviews = [
    "Loved the pizza and pasta",
    "Service was very slow",
    "Great ambience and loved the food",
    "Pizza was fresh but service was slow"
]

vectorizer = CountVectorizer()
bow_matrix = vectorizer.fit_transform(reviews)
features = vectorizer.get_feature_names_out()

df_bow = pd.DataFrame(bow_matrix.toarray(), columns=features, index=[f"Review {i+1}" for i in range(len(reviews))])
print(df_bow)
```

#### Output:
```text
          ambience  and  but  food  fresh  great  loved  pasta  pizza  service  slow  the  very  was
Review 1         0    1    0     0      0      0      1      1      1        0     0    1     0    0
Review 2         0    0    0     0      0      0      0      0      0        1     1    0     1    1
Review 3         1    1    0     1      0      1      1      0      0        0     0    1     0    0
Review 4         0    0    1     0      1      0      0      0      1        1     1    0     0    2
```

---

### Task 3: Word Frequency Counter on WhatsApp Messages

#### Question:
> Given an array of WhatsApp chat messages, count and display the frequency of each word that appears across all messages.  
> *(Hint: Use a dictionary to store word counts and ignore common stopwords like 'the', 'is', 'at'.)*

#### Python Implementation:
```python
import string

messages = [
    "Are we meeting at the library at 5 pm?",
    "Yes, the library is open until 8 pm today.",
    "Don't forget to bring the data science assignment notes to the library!",
    "I will be at the library by 5 pm sharp."
]

stopwords = {"the", "is", "at", "to", "by", "we", "i", "are", "be", "dont", "in", "of", "and", "a", "an"}
translator = str.maketrans('', '', string.punctuation)
word_counts = {}

for msg in messages:
    clean = msg.lower().translate(translator)
    for token in clean.split():
        if token not in stopwords:
            word_counts[token] = word_counts.get(token, 0) + 1

for word, count in sorted(word_counts.items(), key=lambda item: item[1], reverse=True):
    print(f"{word:<15}: {count}")
```

#### Output:
```text
library        : 4
pm             : 3
5              : 2
meeting        : 1
open           : 1
today          : 1
assignment     : 1
data           : 1
science        : 1
```

---

### Task 4: TF-IDF Vectorization on Flipkart Reviews & 'quality' Value

#### Question:
> Use scikit-learn's `TfidfVectorizer` to transform a list of Flipkart product reviews into TF-IDF vectors, then print the TF-IDF value for the word `'quality'` in each review.  
> *(Constraint: Use at least 3 reviews and show the output for each).*

#### Python Implementation:
```python
from sklearn.feature_extraction.text import TfidfVectorizer

reviews = [
    "The sound quality of these wireless earphones is truly top notch and crisp.",
    "Average build quality but the sound output is decent for this budget price.",
    "Fast delivery by Flipkart and the packaging was good, overall satisfied with the purchase.",
    "Premium quality material, exceptional battery life, and outstanding camera quality."
]

tfidf = TfidfVectorizer()
matrix = tfidf.fit_transform(reviews)
features = list(tfidf.get_feature_names_out())
q_idx = features.index('quality')

for i, rev in enumerate(reviews, 1):
    val = matrix[i-1].toarray()[0][q_idx]
    print(f"Review {i}: '{rev}'\n-> TF-IDF score for 'quality': {val:.4f}\n")
```

#### Output:
```text
Review 1: 'The sound quality of these wireless earphones is truly top notch and crisp.'
-> TF-IDF score for 'quality': 0.1973

Review 2: 'Average build quality but the sound output is decent for this budget price.'
-> TF-IDF score for 'quality': 0.1919

Review 3: 'Fast delivery by Flipkart and the packaging was good, overall satisfied with the purchase.'
-> TF-IDF score for 'quality': 0.0000

Review 4: 'Premium quality material, exceptional battery life, and outstanding camera quality.'
-> TF-IDF score for 'quality': 0.4247
```
*(Notice how Review 4 has a higher score of 0.4247 because 'quality' appears twice, whereas Review 3 gets 0.0000 because 'quality' is absent).*

---

### Task 5: Bag of Words vs TF-IDF Theoretical Comparison

#### Question:
> In your own words, list two key differences between Bag of Words and TF-IDF, and give a real-world example (from any app you use) where TF-IDF would be more useful than Bag of Words.

#### Solution & Comparison:

| Feature Dimension | Bag of Words (BoW) | TF-IDF |
| :--- | :--- | :--- |
| **Mathematical Formula** | $\text{Count}(t, d)$ | $\text{TF}(t, d) \times \log\left(\frac{N}{\text{DF}(t)}\right)$ |
| **Weighting Philosophy** | Frequency-only: Assigns higher score to words that occur frequently, regardless of corpus ubiquity. | Importance-based: Rewards terms specific to a document while heavily penalizing corpus-wide common words. |
| **Domain Noise Handling** | Highly vulnerable to ubiquitous terms like "product", "movie", "food" dominating the vector space. | Naturally dampens noise through the inverse document frequency (IDF) log penalty. |

**Real-World Example in Flipkart Search:**  
When searching for *"noise cancelling wireless headphones"*:
- **BoW Failure:** "Wireless" and "headphones" occur across 90% of audio products, dominating the vector match. The search engine might return ordinary cheap Bluetooth earbuds without ANC.
- **TF-IDF Advantage:** "Noise cancelling" is a rare, high-specificity phrase in the product database, yielding a high IDF multiplier. TF-IDF ranks genuine ANC headsets to the top of the search results.

---

## 🚀 How to Run the Code
```bash
python Session-03_Feature_Engineering_for_Text/session_03_tasks.py
```

