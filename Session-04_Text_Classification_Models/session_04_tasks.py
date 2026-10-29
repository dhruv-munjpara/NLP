"""
Session 04: Text Classification Models
Course: Data Analytics & Data Science (DA & DS)
Student: Dhruv Munjpara
Institute: TOPS Technologies

Tasks Covered:
- Task 1: Load and inspect IPL cricket match tweets CSV using pandas (print first 10 tweets)
- Task 2: Multinomial Naive Bayes classifier on 20+ Zomato reviews (Positive vs Negative)
- Task 3: WhatsApp spam detector using Logistic Regression on 15+ messages & predict on 3 new messages
- Task 4: Compare accuracy of Naive Bayes vs Logistic Regression with train_test_split
- Task 5: AI-assisted (ChatGPT/Copilot) Logistic Regression for Flipkart reviews with prompt & test verification
"""

import sys
import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report

if sys.stdout.encoding.lower() != 'utf-8':
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except Exception:
        pass


# =====================================================================
# Task 1: Load IPL Cricket Match Tweets CSV (First 10 Tweets)
# =====================================================================
def task_1_load_ipl_tweets(csv_path: str = None):
    print("[Task 1: Load IPL Cricket Match Tweets Dataset]")
    if csv_path is None:
        csv_path = os.path.join(os.path.dirname(__file__), "ipl_tweets.csv")
        
    df = pd.read_csv(csv_path)
    print(f"Dataset successfully loaded from: {csv_path}")
    print(f"Total Rows: {len(df)} | Columns: {list(df.columns)}\n")
    
    print("--- First 10 IPL Tweets ---")
    for i, row in df.head(10).iterrows():
        print(f"Tweet #{row['tweet_id']} | {row['author']} | Match: {row['match_teams']}")
        print(f"  Content   : {row['tweet_text']}")
        print(f"  Sentiment : {row['sentiment']}\n")
        
    return df


# =====================================================================
# Task 2: Naive Bayes Sentiment Classifier for Zomato Reviews
# =====================================================================
def get_zomato_dataset():
    """
    Returns 24 curated Zomato reviews with ground-truth sentiment labels.
    """
    reviews = [
        # Positive (12)
        ("The butter paneer and garlic naan were absolutely delicious!", "positive"),
        ("Fast delivery, hot packaging, and generous portion sizes. Loved it!", "positive"),
        ("Exceptional pizza crust with authentic Italian mozzarella and basil.", "positive"),
        ("Best cold coffee in Ahmedabad, great ambience and polite staff.", "positive"),
        ("Crispy dosas with mouthwatering spicy chutney. Highly recommended!", "positive"),
        ("Outstanding burger! Juicy patty and crispy fries cooked to perfection.", "positive"),
        ("The dessert was heavenly, rich dark chocolate and fresh strawberries.", "positive"),
        ("Super quick delivery in under 20 minutes, rider was very courteous.", "positive"),
        ("Authentic Gujarati thali with great taste and unlimited refills.", "positive"),
        ("Amazing biryani with fragrant basmati rice and perfectly cooked tender meat.", "positive"),
        ("Five stars for hygiene, spotless packaging, and savory food quality.", "positive"),
        ("Wonderful experience, excellent flavors and very affordable prices.", "positive"),
        
        # Negative (12)
        ("Terrible experience, food was delivered cold and completely tasteless.", "negative"),
        ("Found hair in the gravy, extremely unhygienic and disgusting service.", "negative"),
        ("Order was delayed by more than an hour, customer care was unhelpful.", "negative"),
        ("Soggy fries and stale burger bun, waste of hard-earned money.", "negative"),
        ("Soup was completely spilled inside the polythene bag, messy packaging.", "negative"),
        ("Way too salty and oily curry, could not eat more than two bites.", "negative"),
        ("Missing items from the combo order, and refund request was rejected.", "negative"),
        ("Stale paneer that smelled sour and spoiled, dangerous food quality.", "negative"),
        ("Worst pizza ever! Burnt crust and barely any cheese topping.", "negative"),
        ("Rider was extremely rude and demanded extra delivery cash.", "negative"),
        ("Very small portion size for the exorbitant ₹450 price tag. Bad value.", "negative"),
        ("Cold, soggy, and completely flavorless noodles. Deeply disappointed.", "negative")
    ]
    texts, labels = zip(*reviews)
    return list(texts), list(labels)


def task_2_naive_bayes_classifier():
    print("[Task 2: Train Naive Bayes Classifier on Zomato Reviews]")
    texts, labels = get_zomato_dataset()
    
    vectorizer = CountVectorizer()
    X = vectorizer.fit_transform(texts)
    y = labels
    
    nb_model = MultinomialNB()
    nb_model.fit(X, y)
    
    # Test on unseen reviews
    test_samples = [
        "The pizza was hot, cheesy, and delicious!",
        "Disgusting stale food with terrible delivery delay.",
        "Loved the fresh ingredients and great packaging."
    ]
    
    X_test = vectorizer.transform(test_samples)
    preds = nb_model.predict(X_test)
    
    print("Model: MultinomialNB trained on 24 Zomato reviews.")
    print("Predictions on New Reviews:")
    for rev, pred in zip(test_samples, preds):
        print(f"  Review    : \"{rev}\"")
        print(f"  Prediction: {pred.upper()}\n")
        
    return nb_model, vectorizer


# =====================================================================
# Task 3: WhatsApp Spam Detector using Logistic Regression
# =====================================================================
def get_whatsapp_spam_dataset():
    """
    Returns 16 realistic WhatsApp-style messages labeled as 'spam' or 'not spam'.
    """
    data = [
        # Spam (8)
        ("Congratulations! You won a cash lottery of ₹1,00,000. Click here to claim: bit.ly/win", "spam"),
        ("URGENT: Your KYC is expired. Update immediately to prevent account suspension.", "spam"),
        ("Earn ₹5,000 per day working from home 1 hour! No investment needed, message now.", "spam"),
        ("Claim your free gift voucher worth ₹2,000 on Amazon now. Limited time offer!", "spam"),
        ("You have been pre-approved for an instant personal loan at 0% interest, apply now.", "spam"),
        ("Free recharge of ₹499 for all WhatsApp users. Forward to 5 groups to activate.", "spam"),
        ("Win an iPhone 16 Pro Max today! Enter lucky spin contest now for free.", "spam"),
        ("Exclusive stock trading tips: guaranteed 100% daily profit. Join VIP group.", "spam"),
        
        # Not Spam / Legitimate (8)
        ("Hey Dhruv, are we still meeting at the campus library at 4 PM for the project?", "not spam"),
        ("Can you please share the machine learning lecture notes and assignment slides?", "not spam"),
        ("Mom asked if you will be home for dinner tonight or dining out?", "not spam"),
        ("Your OTP for HDFC Netbanking login is 482910. Do not share it with anyone.", "not spam"),
        ("The professor rescheduled tomorrow's lab session to Friday morning 10 AM.", "not spam"),
        ("Shared the GitHub repository link for our NLP assignment. Please review PR.", "not spam"),
        ("Let me know once you reach the metro station, I will pick you up.", "not spam"),
        ("Happy birthday Dhruv! Wishing you a wonderful and successful year ahead!", "not spam")
    ]
    texts, labels = zip(*data)
    return list(texts), list(labels)


def task_3_whatsapp_spam_detector():
    print("[Task 3: WhatsApp Spam Detector using Logistic Regression]")
    texts, labels = get_whatsapp_spam_dataset()
    
    tfidf = TfidfVectorizer()
    X = tfidf.fit_transform(texts)
    y = labels
    
    lr_model = LogisticRegression(random_state=42)
    lr_model.fit(X, y)
    
    # 3 New custom test messages
    custom_messages = [
        "Congratulations! You win a free VIP holiday pass, click bit.ly/claim to win!",
        "Are you attending the data science seminar at Tops Technologies this weekend?",
        "Work from home daily part-time job, earn ₹2000 free bonus instantly!"
    ]
    
    X_custom = tfidf.transform(custom_messages)
    preds = lr_model.predict(X_custom)
    probs = lr_model.predict_proba(X_custom)
    
    print(f"Trained Logistic Regression on {len(texts)} WhatsApp messages.\n")
    print("Predictions on 3 New Custom Messages:")
    for msg, pred, prob in zip(custom_messages, preds, probs):
        spam_idx = list(lr_model.classes_).index('spam')
        spam_prob = prob[spam_idx] * 100
        print(f"Message   : \"{msg}\"")
        print(f"Prediction: {pred.upper()} (Spam Confidence: {spam_prob:.1f}%)\n")
        
    return lr_model, tfidf


# =====================================================================
# Task 4: Compare Naive Bayes vs Logistic Regression with train_test_split
# =====================================================================
def task_4_model_comparison():
    print("[Task 4: Model Evaluation & Comparison (Naive Bayes vs Logistic Regression)]")
    texts, labels = get_zomato_dataset()
    
    # 75% train, 25% test
    X_train_raw, X_test_raw, y_train, y_test = train_test_split(
        texts, labels, test_size=0.25, random_state=42, stratify=labels
    )
    
    tfidf = TfidfVectorizer()
    X_train = tfidf.fit_transform(X_train_raw)
    X_test = tfidf.transform(X_test_raw)
    
    # Naive Bayes
    nb = MultinomialNB()
    nb.fit(X_train, y_train)
    nb_preds = nb.predict(X_test)
    nb_acc = accuracy_score(y_test, nb_preds)
    
    # Logistic Regression
    lr = LogisticRegression(random_state=42)
    lr.fit(X_train, y_train)
    lr_preds = lr.predict(X_test)
    lr_acc = accuracy_score(y_test, lr_preds)
    
    print(f"Training Set Size: {len(X_train_raw)} | Test Set Size: {len(X_test_raw)}")
    print(f"Naive Bayes Accuracy         : {nb_acc * 100:.2f}%")
    print(f"Logistic Regression Accuracy : {lr_acc * 100:.2f}%\n")
    
    if nb_acc > lr_acc:
        winner = "Multinomial Naive Bayes performs better on this small text dataset."
    elif lr_acc > nb_acc:
        winner = "Logistic Regression performs better on this small text dataset."
    else:
        winner = "Both models achieved identical accuracy (tie) on the test holdout set."
        
    print("Conclusion:")
    print(f"  {winner}")
    print("  Note: Naive Bayes excels on sparse small text corpora due to strong independence assumptions,")
    print("  while Logistic Regression scales smoothly with calibrated probability boundaries on larger corpora.")
    return nb_acc, lr_acc


# =====================================================================
# Task 5: AI-Assisted (ChatGPT/Copilot) Logistic Regression for Flipkart
# =====================================================================
def task_5_ai_generated_flipkart_classifier():
    print("[Task 5: AI-Generated Flipkart Review Classifier using Logistic Regression]")
    prompt_used = """
Prompt given to ChatGPT/Copilot:
\"Write a complete Python code using scikit-learn to classify Flipkart product reviews
as either 'positive' or 'negative' using TfidfVectorizer and LogisticRegression.
Include a sample training dataset of 10 product reviews and evaluate on 2 test reviews.\"
"""
    print(prompt_used.strip())
    print("\nExecuting the AI-Generated Implementation:\n")
    
    # AI-Generated Code Block:
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
    
    X_train_rev, y_train_rev = zip(*flipkart_reviews)
    
    # 2. Vectorization & Training
    vectorizer = TfidfVectorizer()
    X_train_vec = vectorizer.fit_transform(X_train_rev)
    
    model = LogisticRegression()
    model.fit(X_train_vec, y_train_rev)
    
    # 3. Testing on 2 Sample Reviews
    sample_tests = [
        "Super fast processor and incredible build quality, worth every rupee!",
        "Defective display and worst customer support, stopped working in 3 days."
    ]
    
    sample_vec = vectorizer.transform(sample_tests)
    predictions = model.predict(sample_vec)
    
    for rev, pred in zip(sample_tests, predictions):
        print(f"Test Review: \"{rev}\"")
        print(f"Predicted  : {pred.upper()}\n")
        
    return model, vectorizer


# =====================================================================
# Main Runner Entry Point
# =====================================================================
def main():
    print("=" * 70)
    print("SESSION 4: TEXT CLASSIFICATION MODELS - ALL TASKS")
    print("=" * 70)
    print()
    task_1_load_ipl_tweets()
    print("\n" + "-" * 70 + "\n")
    task_2_naive_bayes_classifier()
    print("\n" + "-" * 70 + "\n")
    task_3_whatsapp_spam_detector()
    print("\n" + "-" * 70 + "\n")
    task_4_model_comparison()
    print("\n" + "-" * 70 + "\n")
    task_5_ai_generated_flipkart_classifier()
    print("=" * 70)
    print("SESSION 4 TASKS COMPLETED SUCCESSFULLY!")
    print("=" * 70)


if __name__ == "__main__":
    main()

