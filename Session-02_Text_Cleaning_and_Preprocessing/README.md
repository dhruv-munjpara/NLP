# Session 02: Text Cleaning & Preprocessing

**Course:** Data Analytics & Data Science (DA & DS)  
**Student:** Dhruv Munjpara  
**Topic:** Natural Language Processing (NLP) — Session 2  
**Status:** Completed  
**Script File:** [`session_02_tasks.py`](./session_02_tasks.py)

---

## 📌 Curriculum Overview

This session covers fundamental text cleaning and preprocessing techniques essential for preparing raw human text for machine learning and deep learning pipelines:
- Case normalization (lowercasing)
- Punctuation removal
- Word tokenization
- Stopword removal with NLTK
- Morphological stemming using PorterStemmer

---

## 📝 Tasks & Comprehensive Solutions

### Task 1: Lowercase Instagram Bio

#### Question:
> Write a Python function that takes a user’s Instagram bio as input and returns the text in all lowercase letters.

#### Python Implementation:
```python
def lowercase_instagram_bio(bio: str) -> str:
    return bio.lower()

bio = "Data Scientist & AI Researcher | Tech Enthusiast @Ahmedabad | Living Life"
print(lowercase_instagram_bio(bio))
```

#### Output:
```text
data scientist & ai researcher | tech enthusiast @ahmedabad | living life
```

---

### Task 2: Remove Punctuation from Flipkart Product Review

#### Question:
> Create a function to remove all punctuation marks from a Flipkart product review string and return the cleaned text.  
> *(Hint: Use the `string.punctuation` constant from the Python `string` module.)*

#### Python Implementation:
```python
import string

def remove_punctuation(review_text: str) -> str:
    # str.maketrans maps each punctuation mark to None
    translator = str.maketrans('', '', string.punctuation)
    return review_text.translate(translator)

review = "Amazing sound quality!! Truly worth the price tag; battery backup is 10/10... highly recommended!"
print(remove_punctuation(review))
```

#### Output:
```text
Amazing sound quality Truly worth the price tag battery backup is 1010 highly recommended
```

---

### Task 3: Tokenize Sample Zomato Food Description

#### Question:
> Given a sample Zomato food description, split the sentence into individual words (tokens) and print the resulting list.

#### Python Implementation:
```python
def tokenize_food_description(description: str) -> list:
    return description.split()

desc = "Crispy butter masala dosa served with hot spicy sambar and fresh coconut chutney."
tokens = tokenize_food_description(desc)
print("Tokens:", tokens)
print("Count:", len(tokens))
```

#### Output:
```text
Tokens: ['Crispy', 'butter', 'masala', 'dosa', 'served', 'with', 'hot', 'spicy', 'sambar', 'and', 'fresh', 'coconut', 'chutney.']
Count: 13
```

---

### Task 4: Remove English Stopwords from WhatsApp Message using NLTK

#### Question:
> Download a list of English stopwords using NLTK and write a function that removes all stopwords from a WhatsApp message string.

#### Python Implementation:
```python
import string
import nltk
from nltk.corpus import stopwords

nltk.download('stopwords', quiet=True)
stop_words_set = set(stopwords.words('english'))

def remove_stopwords_whatsapp(message: str) -> str:
    words = message.split()
    filtered = [w for w in words if w.lower().strip(string.punctuation) not in stop_words_set]
    return " ".join(filtered)

msg = "Hey, are you going to the college today or will you be staying at home for the online lecture?"
print("Original:", msg)
print("Cleaned :", remove_stopwords_whatsapp(msg))
```

#### Output:
```text
Original: Hey, are you going to the college today or will you be staying at home for the online lecture?
Cleaned : Hey, going college today staying home online lecture?
```

---

### Task 5: Stem Twitter Hashtags using NLTK's PorterStemmer

#### Question:
> Use NLTK’s `PorterStemmer` to stem each word in a list of trending hashtags from Twitter, and print both the original and stemmed versions side by side.  
> *(Hint: Example hashtags: `['#playing', '#dancing', '#eating']`)*

#### Python Implementation:
```python
from nltk.stem import PorterStemmer

stemmer = PorterStemmer()
hashtags = ["#playing", "#dancing", "#eating", "#coding", "#innovative", "#organizing"]

print(f"{'Original Hashtag':<25} | {'Clean Word':<15} | {'Stemmed Form':<15}")
print("-" * 60)
for tag in hashtags:
    word = tag.lstrip('#').lower()
    stemmed = stemmer.stem(word)
    print(f"{tag:<25} | {word:<15} | {stemmed:<15}")
```

#### Output:
```text
Original Hashtag          | Clean Word      | Stemmed Form   
------------------------------------------------------------
#playing                  | playing         | play           
#dancing                  | dancing         | danc           
#eating                   | eating          | eat            
#coding                   | coding          | code           
#innovative               | innovative      | innov          
#organizing               | organizing      | organ          
```

---

## 🚀 How to Run the Code
```bash
python Session-02_Text_Cleaning_and_Preprocessing/session_02_tasks.py
```
