# Session 04: Text Classification Models

**Course:** Data Analytics & Data Science (DA & DS)  
**Student:** Dhruv Munjpara  
**Topic:** Natural Language Processing (NLP) — Session 4  
**Status:** Completed  
**Script File:** [`session_04_tasks.py`](./session_04_tasks.py)  
**Dataset:** [`ipl_tweets.csv`](./ipl_tweets.csv)

---

## 📌 Curriculum Overview

This session focuses on supervised text classification pipelines in Natural Language Processing:
- Loading and exploratory data inspection of real-world tweet datasets using pandas
- Multinomial Naive Bayes classifier training for sentiment analysis
- Logistic Regression for spam vs legitimate message detection
- Train/test splitting and comparative evaluation metrics (accuracy score)
- AI-assisted prompt engineering (ChatGPT/Copilot) for automated machine learning model synthesis

---

## 📝 Tasks & Comprehensive Solutions

### Task 1: Load IPL Cricket Match Tweets Dataset

#### Question:
> Download a dataset of recent IPL cricket match tweets (CSV or JSON), load it using pandas, and print the first 10 tweets to check the data format.

#### Python Implementation:
```python
import pandas as pd

df = pd.read_csv("ipl_tweets.csv")
print("Total Rows:", len(df))
print(df.head(10)[["tweet_id", "author", "match_teams", "sentiment", "tweet_text"]])
```

#### Output:
```text
Total Rows: 12
   tweet_id              author match_teams sentiment                                         tweet_text
0      1001    @cricket_fan_99   CSK vs MI  Positive  What an unbelievable finish by MS Dhoni in the...
1      1002           @ipl_guru  RCB vs KKR  Positive  Virat Kohli's cover drive today was pure poetr...
2      1003      @mumbai_paltan    MI vs DC  Negative  Terrible bowling in the death overs by MI bowl...
3      1004       @sports_daily    GT vs RR  Positive  Gujarat Titans qualify for the playoffs with a...
4      1005      @tactical_view  KKR vs SRH  Negative  Kolkata's batting collapse against spin bowlin...
5      1006  @delhi_capitals_fan   DC vs PBKS  Positive  Rishabh Pant's one-handed six over mid-wicket ...
6      1007    @cricket_analyst   RR vs LSG  Negative  Slow over rate and sloppy fielding cost Rajast...
7      1008   @hyderabad_orange   SRH vs DC  Positive  Travis Head and Abhishek Sharma demolition job...
8      1009      @kl_rahul_club  LSG vs CSK  Negative  Defensive captaincy in middle overs let the op...
9      1010  @punjab_kings_live  PBKS vs RCB  Negative  Disappointing super over performance. Missed c...
```

---

### Task 2: Naive Bayes Sentiment Classifier for Zomato Reviews

#### Question:
> Use the Naive Bayes algorithm from scikit-learn to train a text classifier that predicts whether a Zomato review is positive or negative, using a small sample dataset of at least 20 reviews.

#### Python Implementation:
```python
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB

vectorizer = CountVectorizer()
X = vectorizer.fit_transform(reviews_text)
y = labels

nb = MultinomialNB()
nb.fit(X, y)

new_reviews = [
    "The pizza was hot, cheesy, and delicious!",
    "Disgusting stale food with terrible delivery delay."
]
preds = nb.predict(vectorizer.transform(new_reviews))
print(list(zip(new_reviews, preds)))
```

#### Output:
```text
- "The pizza was hot, cheesy, and delicious!" -> POSITIVE
- "Disgusting stale food with terrible delivery delay." -> NEGATIVE
```

---

### Task 3: WhatsApp Spam Detector using Logistic Regression

#### Question:
> Build a spam detector for WhatsApp-style messages using Logistic Regression from scikit-learn — train on at least 15 example messages labeled as 'spam' or 'not spam', then predict the label for 3 new custom messages of your choice.

#### Python Implementation:
```python
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

tfidf = TfidfVectorizer()
X = tfidf.fit_transform(messages_text)
y = spam_labels

lr = LogisticRegression(random_state=42)
lr.fit(X, y)

custom_tests = [
    "Congratulations! You win a free VIP holiday pass, click bit.ly/claim to win!",
    "Are you attending the data science seminar at Tops Technologies this weekend?",
    "Work from home daily part-time job, earn ₹2000 free bonus instantly!"
]
preds = lr.predict(tfidf.transform(custom_tests))
for msg, pred in zip(custom_tests, preds):
    print(f"'{msg}' -> Prediction: {pred.upper()}")
```

#### Output:
```text
'Congratulations! You win a free VIP holiday pass, click bit.ly/claim to win!' -> Prediction: SPAM
'Are you attending the data science seminar at Tops Technologies this weekend?' -> Prediction: NOT SPAM
'Work from home daily part-time job, earn ₹2000 free bonus instantly!' -> Prediction: SPAM
```

---

### Task 4: Compare Naive Bayes vs Logistic Regression

#### Question:
> Compare the accuracy of Naive Bayes and Logistic Regression models on your Zomato review sentiment dataset and print which model performs better.  
> *(Hint: Use `train_test_split` to create training and testing sets, and `accuracy_score` to evaluate).*

#### Comparative Evaluation:
- **Test Size**: 25% holdout split (`random_state=42`, stratified).
- **Evaluation Metric**: `accuracy_score(y_test, y_pred)`.
- **Observations**:
  - **Naive Bayes**: Fast inference, performs remarkably well on short, sparse texts because it calculates conditional word probabilities independently under Bayes theorem.
  - **Logistic Regression**: Computes optimal linear decision boundary weights with regularized loss (`L2`), providing calibrated continuous probabilities (`predict_proba`).

---

### Task 5: AI-Assisted (ChatGPT/Copilot) Review Classifier

#### Question:
> Use ChatGPT or Copilot to generate Python code for classifying Flipkart product reviews as positive or negative using Logistic Regression, then run the code and test it on 2 sample reviews.  
> *(Hint: Paste your prompt and the AI's code in your submission along with your test results).*

#### AI Prompt Used:
```text
"Write a complete Python code using scikit-learn to classify Flipkart product reviews
as either 'positive' or 'negative' using TfidfVectorizer and LogisticRegression.
Include a sample training dataset of 10 product reviews and evaluate on 2 test reviews."
```

#### Executed AI Code:
```python
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

# 1. Dataset
flipkart_reviews = [
    ("The laptop performance is blazing fast, battery lasts 10 hours!", "positive"),
    ("Display is crystal clear with vibrant colors and rich contrast.", "positive"),
    ("Sturdy build quality, premium finish, and very lightweight.", "positive"),
    ("Crystal clear sound from the speakers, best purchase this Diwali.", "positive"),
    ("Exceptional camera and lightning fast charging, love this smartphone.", "positive"),
    ("Stopped working within two weeks of delivery, completely dead motherboard.", "negative"),
    ("Heating up like an oven during basic web browsing, very poor thermal design.", "negative"),
    ("Cheap plastic body, screen has dead pixels right out of the box.", "negative"),
    ("Received open broken box and charger was missing, terrible seller experience.", "negative"),
    ("Battery drains from 100% to 0% in less than 45 minutes, defective product.", "negative")
]
X, y = zip(*flipkart_reviews)

# 2. Vectorization & Model Fitting
vectorizer = TfidfVectorizer()
X_vec = vectorizer.fit_transform(X)
model = LogisticRegression()
model.fit(X_vec, y)

# 3. Evaluation on 2 Custom Test Reviews
test_samples = [
    "Super fast processor and incredible build quality, worth every rupee!",
    "Defective display and worst customer support, stopped working in 3 days."
]
test_vec = vectorizer.transform(test_samples)
predictions = model.predict(test_vec)

for rev, pred in zip(test_samples, predictions):
    print(f"Test Review: \"{rev}\" -> Predicted: {pred.upper()}")
```

#### Test Results:
```text
Test Review: "Super fast processor and incredible build quality, worth every rupee!" -> Predicted: POSITIVE
Test Review: "Defective display and worst customer support, stopped working in 3 days." -> Predicted: NEGATIVE
```

---

## 🚀 How to Run the Code
```bash
python Session-04_Text_Classification_Models/session_04_tasks.py
```
