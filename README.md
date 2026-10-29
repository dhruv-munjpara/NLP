# Natural Language Processing (NLP) — TOPS Technologies

**Course:** Data Analytics & Data Science (DA & DS) Career Program  
**Student Name:** Dhruv Munjpara  
**Portal:** [TOPS Technologies Career Center](https://careercenter.tops-int.com/program-enrollment/58978?tab=assignments)  
**Module:** Natural Language Processing (NLP) — Sessions 1 to 5 (All 25 Tasks)  
**Repository:** [https://github.com/dhruv-munjpara/NLP.git](https://github.com/dhruv-munjpara/NLP.git)  

---

## 📚 Curriculum & Session Index

| Session | Title | Key Topics & Modules | Status | Directory Link |
| :---: | :--- | :--- | :---: | :--- |
| **01** | **Introduction to NLP & Use Cases** | NLP Definitions, Daily App Applications (WhatsApp/Swiggy), Tokenization via `split()`, Rule-based Sentiment Analysis, Email Spam Filter (`is_spam`) | Completed | [Session-01_Introduction_to_NLP_and_Use_Cases](./Session-01_Introduction_to_NLP_and_Use_Cases) |
| **02** | **Text Cleaning & Preprocessing** | Case Normalization, Punctuation Stripping (`string.punctuation`), Word Tokenization, NLTK English Stopwords Removal, NLTK `PorterStemmer` on Twitter Hashtags | Completed | [Session-02_Text_Cleaning_and_Preprocessing](./Session-02_Text_Cleaning_and_Preprocessing) |
| **03** | **Feature Engineering for Text** | Social Caption Vocabulary Extraction, Scikit-Learn `CountVectorizer` Bag of Words (BoW) Matrix, Chat Frequency Counter, `TfidfVectorizer` Scoring, BoW vs TF-IDF | Completed | [Session-03_Feature_Engineering_for_Text](./Session-03_Feature_Engineering_for_Text) |
| **04** | **Text Classification Models** | IPL Match Tweets Pandas Inspection, Multinomial Naive Bayes Classifier on Zomato Reviews, WhatsApp Spam Detector using Logistic Regression, Model Accuracy Comparison, AI-Assisted Flipkart Classifier | Completed | [Session-04_Text_Classification_Models](./Session-04_Text_Classification_Models) |
| **05** | **Deep Learning for NLP** | Deep Learning Setup (`torch` & `transformers`), Limitations of Traditional NLP on Chat Text, Hugging Face Pipeline Sentiment Inference, SOTA Transformer Analysis (RoBERTa vs BERT), Scaled Dot-Product Attention Mechanism | Completed | [Session-05_Deep_Learning_for_NLP](./Session-05_Deep_Learning_for_NLP) |

---

## 📁 Repository Directory Structure

```
D:/DA&DS/NLP/
│
├── README.md                                                 # Master Curriculum & Documentation
├── run_all_sessions.py                                       # Automated Runner for All 5 Sessions
│
├── Session-01_Introduction_to_NLP_and_Use_Cases/
│   ├── README.md                                             # Detailed Theory, Q&A, and Implementation Guide
│   └── session_01_tasks.py                                   # Python Code for Tasks 1 to 5
│
├── Session-02_Text_Cleaning_and_Preprocessing/
│   ├── README.md                                             # Detailed Theory, Q&A, and Implementation Guide
│   └── session_02_tasks.py                                   # Python Code for Tasks 1 to 5
│
├── Session-03_Feature_Engineering_for_Text/
│   ├── README.md                                             # Detailed Theory, Q&A, and Implementation Guide
│   └── session_03_tasks.py                                   # Python Code for Tasks 1 to 5
│
├── Session-04_Text_Classification_Models/
│   ├── README.md                                             # Detailed Theory, Q&A, and Implementation Guide
│   ├── session_04_tasks.py                                   # Python Code for Tasks 1 to 5
│   └── ipl_tweets.csv                                        # IPL Cricket Match Tweets Dataset (Task 1)
│
└── Session-05_Deep_Learning_for_NLP/
    ├── README.md                                             # Detailed Theory, Q&A, and Implementation Guide
    └── session_05_tasks.py                                   # Python Code for Tasks 1 to 5
```

---

## 🛠️ Summary of Completed Session Tasks

### [Session 1: Introduction to NLP & Use Cases](./Session-01_Introduction_to_NLP_and_Use_Cases/)
- **Task 1:** Plain-language explanation of NLP and daily application in WhatsApp (Smart Reply & in-chat search).
- **Task 2:** Three ways NLP makes AI smarter (Customer Support Chatbots, Gmail Spam Filtering, Predictive Autocomplete).
- **Task 3:** Tokenization using `sentence.split()` on `'I love ordering pizza from Zomato!'`.
- **Task 4:** Pure-Python BookMyShow movie review sentiment detector (`Positive`, `Negative`, `Neutral`).
- **Task 5:** Email spam filter function `is_spam(email_text)` checking `'win'` or `'free'` keywords.

### [Session 2: Text Cleaning & Preprocessing](./Session-02_Text_Cleaning_and_Preprocessing/)
- **Task 1:** Instagram bio lowercase converter function.
- **Task 2:** Function to remove punctuation marks from Flipkart product reviews using `string.punctuation`.
- **Task 3:** Word tokenization of sample Zomato food descriptions.
- **Task 4:** WhatsApp chat message stopword filter using NLTK `stopwords.words('english')`.
- **Task 5:** Stemming trending Twitter hashtags side-by-side using NLTK `PorterStemmer`.

### [Session 3: Feature Engineering for Text](./Session-03_Feature_Engineering_for_Text/)
- **Task 1:** Vocabulary list extraction from three Instagram captions (ignoring case & punctuation).
- **Task 2:** Bag of Words (BoW) document-term matrix using scikit-learn `CountVectorizer` on Zomato reviews.
- **Task 3:** Word frequency dictionary calculator across WhatsApp messages with stopword exclusion.
- **Task 4:** Scikit-learn `TfidfVectorizer` feature transformation on Flipkart reviews with `'quality'` TF-IDF score inspection.
- **Task 5:** Theoretical and architectural comparison of BoW vs TF-IDF with eCommerce search application.

### [Session 4: Text Classification Models](./Session-04_Text_Classification_Models/)
- **Task 1:** Loading and inspecting IPL cricket match tweets CSV using pandas (`head(10)`).
- **Task 2:** Multinomial Naive Bayes classifier on 24 Zomato reviews for sentiment prediction.
- **Task 3:** Logistic Regression WhatsApp spam detection trained on 16 messages and evaluated on 3 new messages.
- **Task 4:** Train-test split accuracy comparison between Naive Bayes and Logistic Regression.
- **Task 5:** AI-assisted (ChatGPT/Copilot) prompt-engineered review classifier with verification.

### [Session 5: Deep Learning for NLP](./Session-05_Deep_Learning_for_NLP/)
- **Task 1:** Version verification of `torch` (`2.14.0+cpu`) and `transformers` (`5.17.0`).
- **Task 2:** Critical analysis of traditional BoW/TF-IDF limitations on conversational chat text (negation & sarcasm).
- **Task 3:** Hugging Face `pipeline("sentiment-analysis")` live inference on Zomato/Swiggy reviews using DistilBERT.
- **Task 4:** Transformer architecture comparison: Meta's RoBERTa vs Google's BERT.
- **Task 5:** Scaled Dot-Product Attention mechanism implementation in Python with mathematical explanation of attention weights.

---

## ⚡ Execution Instructions

### Run Entire Suite (All 5 Sessions):
```bash
python run_all_sessions.py
```

### Run Individual Sessions:
```bash
# Session 1
python Session-01_Introduction_to_NLP_and_Use_Cases/session_01_tasks.py

# Session 2
python Session-02_Text_Cleaning_and_Preprocessing/session_02_tasks.py

# Session 3
python Session-03_Feature_Engineering_for_Text/session_03_tasks.py

# Session 4
python Session-04_Text_Classification_Models/session_04_tasks.py

# Session 5
python Session-05_Deep_Learning_for_NLP/session_05_tasks.py
```
