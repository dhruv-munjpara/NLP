# Session 01: Introduction to NLP & Use Cases

**Course:** Data Analytics & Data Science (DA & DS)  
**Student:** Dhruv Munjpara  
**Topic:** Natural Language Processing (NLP) — Session 1  
**Status:** Completed  
**Script File:** [`session_01_tasks.py`](./session_01_tasks.py)

---

## 📌 Curriculum Overview

This session establishes the foundational principles of Natural Language Processing (NLP), demonstrating how computers process human language and where NLP is applied in everyday software applications.

---

## 📝 Tasks & Comprehensive Solutions

### Task 1: Explanation of NLP & Daily App Example

#### Question:
> Write a short paragraph explaining in your own words what Natural Language Processing (NLP) is and give one example of how you see it used in any app you use daily (like WhatsApp, Instagram, Zomato, etc.).

#### Solution & Explanation:
**What is NLP?**  
Natural Language Processing (NLP) is an interdisciplinary field of Artificial Intelligence, Computer Science, and Linguistics that empowers computers to understand, interpret, process, and generate human language in text or speech formats. Instead of requiring structured numeric matrices or fixed relational tables, NLP enables machines to decipher syntactic structures, semantic meanings, sentiments, and intent from raw human conversation.

**Real-World Example in WhatsApp:**  
In **WhatsApp**, NLP is actively used in **Smart Replies** and **In-Chat Search**:
- When someone sends *"Are you reaching soon?"*, WhatsApp’s on-device NLP model understands the intent and instantly recommends quick one-tap replies like *"On my way!"*, *"Yes, in 5 mins"*, or *"Almost there"*.
- The search functionality uses lemmatization and token matching, allowing users to find messages even if they search for approximate keywords or verb conjugations.

---

### Task 2: Three Ways NLP Makes AI Smarter for Daily Apps

#### Question:
> List three ways that NLP is important in making AI smarter for apps you use, such as chatbots in Swiggy or spam filters in Gmail, and briefly describe how each one helps the user experience.

#### Solution & Analysis:

| # | Use Case & App | How NLP Operates | Impact on User Experience |
|---|----------------|------------------|---------------------------|
| **1** | **Customer Support Chatbots** *(Swiggy / Zomato)* | Uses Intent Classification and Named Entity Recognition (NER) to comprehend complex queries like *"My order #489 is delayed and the drink was spilled"*. | Resolves complaints 24/7 without waiting in human queues, automatically triggers refunds, and routes critical emergencies directly to human executives. |
| **2** | **Automated Content Security & Spam Filtering** *(Gmail)* | Employs Naive Bayes and Transformer-based text classifiers to evaluate email subjects, body content, sender reputation, and phishing semantics. | Shields users from fraudulent scams, phishing links, and promotional clutter, ensuring over 99.9% of inbox clutter is filtered silently into the Spam folder. |
| **3** | **Context-Aware Autocomplete & Query Correction** *(Google / Gboard / Instagram)* | Analyzes language n-grams and transformer attention weights to suggest next words and correct typos in real time. | Drastically speeds up typing, reduces spelling errors, and helps users find relevant content even when searching with partial or misspelled terms. |

---

### Task 3: Tokenization using `split()`

#### Question:
> Given the sentence `'I love ordering pizza from Zomato!'`, break it down into individual words (tokens) using Python's `split()` function, and print the list of tokens.  
> *(Hint: Use `sentence.split()` to get the words.)*

#### Python Implementation:
```python
sentence = "I love ordering pizza from Zomato!"
tokens = sentence.split()

print("Original Sentence:", sentence)
print("Tokens (Words):", tokens)
print("Total Token Count:", len(tokens))
```

#### Output:
```text
Original Sentence: I love ordering pizza from Zomato!
Tokens (Words): ['I', 'love', 'ordering', 'pizza', 'from', 'Zomato!']
Total Token Count: 6
```

---

### Task 4: Simple Movie Review Sentiment Checker (Basic Python)

#### Question:
> Build a simple sentiment checker in Python: ask the user to enter a review about a movie (like on BookMyShow), then check if the review contains the word `'good'` or `'bad'` and print `'Positive'`, `'Negative'`, or `'Neutral'`.  
> *(Constraint: Only use basic Python, no external libraries).*

#### Python Implementation:
```python
def check_sentiment(review_text: str) -> str:
    clean_review = review_text.lower()
    has_good = "good" in clean_review
    has_bad = "bad" in clean_review

    if has_good and not has_bad:
        return "Positive"
    elif has_bad and not has_good:
        return "Negative"
    elif has_good and has_bad:
        return "Mixed / Neutral"
    else:
        return "Neutral"

# Verification with test cases
test_cases = [
    "The movie was so good, the climax scene was thrilling!",
    "Terrible direction and bad acting overall.",
    "The movie was okay, an average one-time watch."
]

for review in test_cases:
    print(f"Review: \"{review}\" -> Sentiment: {check_sentiment(review)}")
```

#### Output:
```text
Review: "The movie was so good, the climax scene was thrilling!" -> Sentiment: Positive
Review: "Terrible direction and bad acting overall." -> Sentiment: Negative
Review: "The movie was okay, an average one-time watch." -> Sentiment: Neutral
```

---

### Task 5: Email Spam Filter Function `is_spam()`

#### Question:
> Imagine you are designing a spam filter for an email app. Write a Python function `is_spam(email_text)` that returns `True` if the `email_text` contains the word `'win'` or `'free'`, otherwise returns `False`. Test your function with two examples.

#### Python Implementation:
```python
def is_spam(email_text: str) -> bool:
    """
    Returns True if email_text contains 'win' or 'free' (case-insensitive),
    otherwise returns False.
    """
    text_lower = email_text.lower()
    if "win" in text_lower or "free" in text_lower:
        return True
    return False

# Testing with examples
example_1 = "Congratulations! You win a brand new iPhone, claim now!"
example_2 = "Hey Dhruv, are we still meeting tomorrow for the NLP study session?"

print(f"Example 1: \"{example_1}\"")
print(f"is_spam() result: {is_spam(example_1)} (Expected: True)\n")

print(f"Example 2: \"{example_2}\"")
print(f"is_spam() result: {is_spam(example_2)} (Expected: False)")
```

#### Output:
```text
Example 1: "Congratulations! You win a brand new iPhone, claim now!"
is_spam() result: True (Expected: True)

Example 2: "Hey Dhruv, are we still meeting tomorrow for the NLP study session?"
is_spam() result: False (Expected: False)
```

---

## 🚀 How to Run the Code
To execute the automated test runner for all 5 tasks:
```bash
python Session-01_Introduction_to_NLP_and_Use_Cases/session_01_tasks.py
```

