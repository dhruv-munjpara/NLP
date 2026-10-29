"""
Session 03: Feature Engineering for Text
Course: Data Analytics & Data Science (DA & DS)
Student: Dhruv Munjpara
Institute: TOPS Technologies

Tasks Covered:
- Task 1: Vocabulary extraction from 3 Instagram captions (ignoring case & punctuation)
- Task 2: Bag of Words (BoW) matrix using scikit-learn CountVectorizer on Zomato reviews
- Task 3: Word frequency counter on WhatsApp chat messages (ignoring stopwords)
- Task 4: TF-IDF vectorization on Flipkart product reviews & 'quality' value extraction
- Task 5: BoW vs TF-IDF comparison & real-world application analysis
"""

import sys
import string
import pandas as pd
from collections import Counter
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer

if sys.stdout.encoding.lower() != 'utf-8':
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except Exception:
        pass


# =====================================================================
# Task 1: Vocabulary List from Instagram Captions
# =====================================================================
def task_1_instagram_vocabulary():
    print("[Task 1: Vocabulary Extraction from Instagram Captions]")
    captions = [
        "Chasing sunsets and making golden memories with the best crew! 🌅",
        "Coffee in one hand, confidence in the other. Sunday vibes! ☕",
        "Exploring hidden street food gems in Ahmedabad with the crew. Best memories!"
    ]
    
    unique_words = set()
    translator = str.maketrans('', '', string.punctuation)
    
    for cap in captions:
        clean_text = cap.lower().translate(translator)
        words = clean_text.split()
        unique_words.update(words)
        
    vocab_list = sorted(list(unique_words))
    
    print("Captions Analyzed:")
    for i, c in enumerate(captions, 1):
        print(f"  {i}. \"{c}\"")
        
    print(f"\nUnique Vocabulary Count: {len(vocab_list)}")
    print("Vocabulary List:")
    print(vocab_list)
    return vocab_list


# =====================================================================
# Task 2: Bag of Words Matrix using CountVectorizer
# =====================================================================
def task_2_bag_of_words():
    print("[Task 2: Bag of Words (BoW) Matrix on Zomato Food Reviews]")
    reviews = [
        "Loved the pizza and pasta",
        "Service was very slow",
        "Great ambience and loved the food",
        "Pizza was fresh but service was slow"
    ]
    
    vectorizer = CountVectorizer()
    bow_matrix = vectorizer.fit_transform(reviews)
    feature_names = vectorizer.get_feature_names_out()
    
    df_bow = pd.DataFrame(bow_matrix.toarray(), columns=feature_names, index=[f"Review {i+1}" for i in range(len(reviews))])
    
    print("Sample Reviews:")
    for i, r in enumerate(reviews, 1):
        print(f"  Review {i}: \"{r}\"")
        
    print("\nVocabulary Features (Columns):")
    print(list(feature_names))
    print("\nBag of Words Matrix (DataFrame Representation):")
    print(df_bow.to_string())
    return df_bow


# =====================================================================
# Task 3: Word Frequency Counter on WhatsApp Messages
# =====================================================================
def task_3_whatsapp_word_frequency():
    print("[Task 3: Word Frequency Counter on WhatsApp Messages]")
    messages = [
        "Are we meeting at the library at 5 pm?",
        "Yes, the library is open until 8 pm today.",
        "Don't forget to bring the data science assignment notes to the library!",
        "I will be at the library by 5 pm sharp."
    ]
    
    # Common stopwords to ignore
    stopwords = {"the", "is", "at", "to", "by", "we", "i", "are", "be", "dont", "in", "of", "and", "a", "an"}
    
    translator = str.maketrans('', '', string.punctuation)
    word_counts = {}
    
    for msg in messages:
        clean = msg.lower().translate(translator)
        tokens = clean.split()
        for token in tokens:
            if token not in stopwords:
                word_counts[token] = word_counts.get(token, 0) + 1
                
    # Sort by frequency descending
    sorted_counts = dict(sorted(word_counts.items(), key=lambda item: item[1], reverse=True))
    
    print("Input Messages:")
    for i, m in enumerate(messages, 1):
        print(f"  Msg {i}: \"{m}\"")
        
    print("\nWord Frequency Table (Excluding Stopwords):")
    print(f"{'Word / Token':<20} | {'Frequency Count':<15}")
    print("-" * 40)
    for word, count in sorted_counts.items():
        print(f"{word:<20} | {count:<15}")
        
    return sorted_counts


# =====================================================================
# Task 4: TF-IDF Vectorization on Flipkart Reviews & 'quality' Score
# =====================================================================
def task_4_tfidf_quality_score():
    print("[Task 4: TF-IDF Vectorizer on Flipkart Reviews ('quality' Weight)]")
    reviews = [
        "The sound quality of these wireless earphones is truly top notch and crisp.",
        "Average build quality but the sound output is decent for this budget price.",
        "Fast delivery by Flipkart and the packaging was good, overall satisfied with the purchase.",
        "Premium quality material, exceptional battery life, and outstanding camera quality."
    ]
    
    tfidf_vectorizer = TfidfVectorizer()
    tfidf_matrix = tfidf_vectorizer.fit_transform(reviews)
    feature_names = tfidf_vectorizer.get_feature_names_out()
    
    # Find column index of target word 'quality'
    if 'quality' in feature_names:
        quality_idx = list(feature_names).index('quality')
    else:
        quality_idx = -1
        
    print(f"Total Reviews: {len(reviews)}")
    print(f"Total Unique Vocabulary Terms: {len(feature_names)}\n")
    
    for i, review in enumerate(reviews, 1):
        row_vector = tfidf_matrix[i - 1].toarray()[0]
        quality_score = row_vector[quality_idx] if quality_idx != -1 else 0.0
        print(f"Review {i}: \"{review}\"")
        print(f"-> TF-IDF Score for 'quality': {quality_score:.4f}\n")
        
    return tfidf_matrix


# =====================================================================
# Task 5: BoW vs TF-IDF Theoretical Comparison & Real-World Use Case
# =====================================================================
def task_5_comparison_and_use_case():
    comparison_text = """
[Task 5: Bag of Words vs TF-IDF Comparison & Real-World Application]

Key Differences:
1. Weighting Mechanism:
   - Bag of Words (BoW): Counts raw term frequencies. Common words appearing across every document 
     receive high counts simply because they appear frequently, without considering document specificity.
   - TF-IDF (Term Frequency-Inverse Document Frequency): Penalizes words that appear everywhere across 
     the corpus (IDF penalty) and elevates rare, domain-specific discriminative keywords that define a document.

2. Document Discrimination:
   - BoW treats every occurrence equally; a document mentioning 'food' 20 times in a restaurant review 
     dataset gets dominated by that term, even though 'food' doesn't differentiate between Italian or Mexican eateries.
   - TF-IDF dampens pervasive common terms and emphasizes distinctive identifiers like 'biryani', 'tacos', or 'alfredo'.

Real-World Example in Zomato / Amazon Search:
In an eCommerce app like Flipkart or Amazon, when a user searches for "noise cancelling wireless earbuds":
- Bag of Words would give disproportionately high weight to the words "earbuds" or "wireless" because they appear 
  in thousands of generic listings.
- TF-IDF drastically boosts the term "noise cancelling" (which is rarer across the whole catalog), ensuring the top 
  search results directly show Active Noise Cancellation (ANC) premium audio products rather than generic cheap earphones.
"""
    print(comparison_text.strip())
    return comparison_text


# =====================================================================
# Main Runner Entry Point
# =====================================================================
def main():
    print("=" * 70)
    print("SESSION 3: FEATURE ENGINEERING FOR TEXT - ALL TASKS")
    print("=" * 70)
    print()
    task_1_instagram_vocabulary()
    print("\n" + "-" * 70 + "\n")
    task_2_bag_of_words()
    print("\n" + "-" * 70 + "\n")
    task_3_whatsapp_word_frequency()
    print("\n" + "-" * 70 + "\n")
    task_4_tfidf_quality_score()
    print("-" * 70 + "\n")
    task_5_comparison_and_use_case()
    print("=" * 70)
    print("SESSION 3 TASKS COMPLETED SUCCESSFULLY!")
    print("=" * 70)


if __name__ == "__main__":
    main()
