"""
Session 01: Introduction to NLP & Use Cases
Course: Data Analytics & Data Science (DA & DS)
Student: Dhruv Munjpara
Institute: TOPS Technologies

Tasks Covered:
- Task 1: Explanation of NLP & Real-World Daily App Example
- Task 2: 3 Ways NLP Makes AI Smarter (Chatbots, Spam Filters, Sentiment/Search)
- Task 3: Tokenization using split() on 'I love ordering pizza from Zomato!'
- Task 4: Simple Movie Review Sentiment Checker (Basic Python, No external libraries)
- Task 5: Email Spam Filter Function is_spam(email_text) checking 'win' or 'free'
"""

import sys

# =====================================================================
# Task 1: Concept Explanation & Daily App Example
# =====================================================================
def task_1_explanation():
    explanation = """
[Task 1: What is NLP & Daily App Example]
Natural Language Processing (NLP) is a branch of Artificial Intelligence (AI)
that enables computers to read, understand, interpret, and generate human 
language in a valuable and context-aware manner. Instead of dealing strictly
with structured numeric databases, NLP allows algorithms to process unstructured
text and voice data spoken or written by humans.

Real-World Example in Daily Apps:
In WhatsApp, NLP powers features like 'Smart Reply' and message search. When a friend 
texts "Are you free tonight?", WhatsApp analyzes the sentence semantics and context 
using NLP and instantly suggests quick response chips such as "Yes!", "Not sure yet", 
or "Call you later", saving time and streamlining typing.
"""
    print(explanation.strip())
    return explanation


# =====================================================================
# Task 2: Three Ways NLP Makes AI Smarter
# =====================================================================
def task_2_importance_of_nlp():
    points = """
[Task 2: Three Ways NLP Makes AI Smarter for Daily Apps]

1. Conversational Understanding & Intent Recognition (e.g., Swiggy / Zomato Chatbots):
   - Role: Parses complex, conversational customer queries like "My order is delayed and missing extra dips".
   - User Impact: Instead of forcing users through rigid IVR trees, the bot understands intent and emotion, 
     automatically fetches live order tracking, and issues instant refunds or live agent escalations.

2. Automated Content Filtering & Security (e.g., Gmail Spam Filter):
   - Role: Analyzes email subject lines, body semantics, and linguistic anomalies using classification models.
   - User Impact: Shields users from phishing scams, malicious links, and inbox clutter by automatically 
     routing suspicious messages to the Spam folder with over 99.9% accuracy.

3. Context-Aware Predictive Text & Autocorrect (e.g., Google Search / Gboard / Instagram Search):
   - Role: Employs language modeling and n-gram statistical predictions to decipher misspelled queries and intent.
   - User Impact: Displays "Did you mean...?" and auto-completes sentences smoothly, reducing typing effort 
     and accelerating information retrieval.
"""
    print(points.strip())
    return points


# =====================================================================
# Task 3: Tokenization using sentence.split()
# =====================================================================
def task_3_tokenization(sentence: str = "I love ordering pizza from Zomato!"):
    """
    Breaks down a given sentence into individual words (tokens) using split().
    """
    tokens = sentence.split()
    print("[Task 3: Tokenization using split()]")
    print(f"Input Sentence : '{sentence}'")
    print(f"Tokens (Words) : {tokens}")
    print(f"Total Tokens   : {len(tokens)}")
    return tokens


# =====================================================================
# Task 4: Simple Movie Review Sentiment Checker (Basic Python Only)
# =====================================================================
def check_sentiment(review_text: str) -> str:
    """
    Checks if review contains 'good' or 'bad' (case-insensitive).
    Constraint: Only basic Python, no external libraries.
    """
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


def task_4_sentiment_checker(sample_reviews=None):
    print("[Task 4: Simple Sentiment Checker (BookMyShow Movie Reviews)]")
    if sample_reviews is None:
        sample_reviews = [
            "The movie was so good, the climax scene was thrilling!",
            "Terrible direction and bad acting overall.",
            "The movie was okay, an average one-time watch.",
            "The acting was good but the script was bad."
        ]
    
    for review in sample_reviews:
        sentiment = check_sentiment(review)
        print(f"Review   : \"{review}\"")
        print(f"Sentiment: {sentiment}\n")


# =====================================================================
# Task 5: Email Spam Filter Function is_spam()
# =====================================================================
def is_spam(email_text: str) -> bool:
    """
    Returns True if email_text contains the word 'win' or 'free' (case-insensitive),
    otherwise returns False.
    """
    text_lower = email_text.lower()
    # Check for keyword occurrence
    if "win" in text_lower or "free" in text_lower:
        return True
    return False


def task_5_spam_filter():
    print("[Task 5: Email Spam Filter is_spam()]")
    test_emails = [
        "Congratulations! You win a brand new iPhone, claim your prize now!",
        "Click here to get a free coupon code for your next shopping cart.",
        "Hey Dhruv, are we still meeting tomorrow for the NLP study session?",
        "Attached is the monthly progress report for our Data Science project."
    ]

    for email in test_emails:
        result = is_spam(email)
        status = "SPAM DETECTED [True]" if result else "HAM / LEGITIMATE [False]"
        print(f"Email: \"{email}\"")
        print(f"Result: is_spam() -> {result} | Status: {status}\n")


# =====================================================================
# Main Interactive / Runner Entry Point
# =====================================================================
def main():
    print("=" * 70)
    print("SESSION 1: INTRODUCTION TO NLP & USE CASES - ALL TASKS")
    print("=" * 70)
    print()
    task_1_explanation()
    print("\n" + "-" * 70 + "\n")
    task_2_importance_of_nlp()
    print("\n" + "-" * 70 + "\n")
    task_3_tokenization()
    print("\n" + "-" * 70 + "\n")
    task_4_sentiment_checker()
    print("-" * 70 + "\n")
    task_5_spam_filter()
    print("=" * 70)
    print("SESSION 1 TASKS COMPLETED SUCCESSFULLY!")
    print("=" * 70)


if __name__ == "__main__":
    main()
