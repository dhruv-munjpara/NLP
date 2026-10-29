"""
Session 02: Text Cleaning & Preprocessing
Course: Data Analytics & Data Science (DA & DS)
Student: Dhruv Munjpara
Institute: TOPS Technologies

Tasks Covered:
- Task 1: Instagram bio lowercase converter
- Task 2: Remove punctuation from Flipkart product review using string.punctuation
- Task 3: Tokenize sample Zomato food description into individual words
- Task 4: Remove English stopwords from a WhatsApp message using NLTK
- Task 5: Stem Twitter hashtags using NLTK PorterStemmer (print original vs stemmed)
"""

import string
import sys
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

if sys.stdout.encoding.lower() != 'utf-8':
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except Exception:
        pass

# Ensure NLTK resources are available
try:
    stop_words_set = set(stopwords.words('english'))
except LookupError:
    nltk.download('stopwords', quiet=True)
    stop_words_set = set(stopwords.words('english'))


# =====================================================================
# Task 1: Lowercase Instagram Bio
# =====================================================================
def lowercase_instagram_bio(bio: str) -> str:
    """
    Takes a user's Instagram bio as input and returns the text in all lowercase.
    """
    return bio.lower()


def task_1_demo():
    print("[Task 1: Lowercase Instagram Bio]")
    sample_bio = "Data Scientist & AI Researcher 💻 | Tech Enthusiast @Ahmedabad | Living Life 🚀"
    cleaned = lowercase_instagram_bio(sample_bio)
    print(f"Original Bio: {sample_bio}")
    print(f"Lowercased  : {cleaned}")
    return cleaned


# =====================================================================
# Task 2: Remove Punctuation from Flipkart Review
# =====================================================================
def remove_punctuation(review_text: str) -> str:
    """
    Removes all punctuation marks from a review string using string.punctuation.
    """
    # Create translation table mapping every punctuation character to None
    translator = str.maketrans('', '', string.punctuation)
    return review_text.translate(translator)


def task_2_demo():
    print("[Task 2: Remove Punctuation from Flipkart Product Review]")
    sample_review = "Amazing sound quality!! Truly worth the ₹1,999 price tag; battery backup is 10/10... highly recommended!"
    cleaned = remove_punctuation(sample_review)
    print(f"Original Review: {sample_review}")
    print(f"Punctuation-Free: {cleaned}")
    return cleaned


# =====================================================================
# Task 3: Tokenize Zomato Food Description
# =====================================================================
def tokenize_food_description(description: str) -> list:
    """
    Splits the sample Zomato food description into individual tokens (words).
    """
    tokens = description.split()
    return tokens


def task_3_demo():
    print("[Task 3: Tokenize Zomato Food Description]")
    sample_desc = "Crispy butter masala dosa served with hot spicy sambar and fresh coconut chutney."
    tokens = tokenize_food_description(sample_desc)
    print(f"Food Description: \"{sample_desc}\"")
    print(f"Tokens (List)   : {tokens}")
    print(f"Total Words     : {len(tokens)}")
    return tokens


# =====================================================================
# Task 4: Remove Stopwords from WhatsApp Message
# =====================================================================
def remove_stopwords_whatsapp(message: str) -> str:
    """
    Removes English stopwords from a WhatsApp message using NLTK.
    """
    words = message.split()
    filtered_words = [w for w in words if w.lower().strip(string.punctuation) not in stop_words_set]
    return " ".join(filtered_words)


def task_4_demo():
    print("[Task 4: Remove NLTK Stopwords from WhatsApp Message]")
    whatsapp_msg = "Hey, are you going to the college today or will you be staying at home for the online lecture?"
    filtered_msg = remove_stopwords_whatsapp(whatsapp_msg)
    print(f"Original Message : \"{whatsapp_msg}\"")
    print(f"Stopwords-Free   : \"{filtered_msg}\"")
    return filtered_msg


# =====================================================================
# Task 5: Stem Twitter Hashtags using PorterStemmer
# =====================================================================
def stem_twitter_hashtags(hashtags: list = None):
    """
    Uses NLTK's PorterStemmer to stem each word in hashtags,
    displaying original vs stemmed side-by-side.
    """
    if hashtags is None:
        hashtags = ["#playing", "#dancing", "#eating", "#coding", "#innovative", "#organizing"]
    
    stemmer = PorterStemmer()
    results = []
    
    print(f"{'Original Hashtag':<25} | {'Clean Word':<15} | {'Stemmed Form':<15}")
    print("-" * 60)
    for tag in hashtags:
        clean_word = tag.lstrip('#').lower()
        stemmed = stemmer.stem(clean_word)
        results.append((tag, clean_word, stemmed))
        print(f"{tag:<25} | {clean_word:<15} | {stemmed:<15}")
    
    return results


def task_5_demo():
    print("[Task 5: NLTK PorterStemmer on Twitter Hashtags]")
    return stem_twitter_hashtags()


# =====================================================================
# Main Runner Entry Point
# =====================================================================
def main():
    print("=" * 70)
    print("SESSION 2: TEXT CLEANING & PREPROCESSING - ALL TASKS")
    print("=" * 70)
    print()
    task_1_demo()
    print("\n" + "-" * 70 + "\n")
    task_2_demo()
    print("\n" + "-" * 70 + "\n")
    task_3_demo()
    print("\n" + "-" * 70 + "\n")
    task_4_demo()
    print("\n" + "-" * 70 + "\n")
    task_5_demo()
    print("=" * 70)
    print("SESSION 2 TASKS COMPLETED SUCCESSFULLY!")
    print("=" * 70)


if __name__ == "__main__":
    main()
