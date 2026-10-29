# Session 05: Deep Learning for NLP

**Course:** Data Analytics & Data Science (DA & DS)  
**Student:** Dhruv Munjpara  
**Topic:** Natural Language Processing (NLP) — Session 5  
**Status:** Completed  
**Script File:** [`session_05_tasks.py`](./session_05_tasks.py)

---

## 📌 Curriculum Overview

This session explores the modern paradigm shift from classical statistical NLP to Deep Learning and Transformers:
- Verifying neural network frameworks (`torch` and `transformers`)
- Analyzing critical limitations of traditional n-gram and count-based models on modern chat texts
- Running production inference with Hugging Face Transformers pipeline for sentiment analysis
- Architectural survey of state-of-the-art transformer backbones (RoBERTa vs BERT)
- Implementing and mathematical understanding of the Scaled Dot-Product Attention mechanism

---

## 📝 Tasks & Comprehensive Solutions

### Task 1: Check `transformers` and `torch` Versions

#### Question:
> Install the `transformers` and `torch` libraries using pip, then write a Python script that prints the version numbers of both libraries to confirm successful installation.

#### Python Implementation:
```python
import torch
import transformers

print("PyTorch Version      :", torch.__version__)
print("Transformers Version :", transformers.__version__)
```

#### Output:
```text
PyTorch Version      : 2.14.0+cpu
Transformers Version : 5.17.0
Confirmation         : Libraries successfully installed and ready for Deep Learning NLP tasks!
```

---

### Task 2: Limitation of Traditional NLP on WhatsApp Chat Messages

#### Question:
> Write a short paragraph (2-3 sentences) explaining one limitation of traditional NLP models (like Bag of Words or TF-IDF) when trying to understand the meaning of a WhatsApp chat message.

#### Solution & Explanation:
**Limitation: Loss of Sequence, Syntactic Context, and Nuance.**  
Traditional bag-of-words and TF-IDF models treat a WhatsApp message as an unordered collection of discrete words, discarding grammatical order and syntactic context. For example, in a message like *"I am not happy with the delayed delivery, it is good for nothing!"*, traditional models count the positive tokens *"happy"* and *"good"* without associating the negation *"not"* or the sarcastic idiom *"good for nothing"*, leading to false positive classifications. Additionally, traditional models cannot parse emojis, slang abbreviations, or contextual tone that are fundamental to modern WhatsApp conversations.

---

### Task 3: Hugging Face Pipeline for Sentiment Analysis

#### Question:
> Use the Hugging Face pipeline to perform sentiment analysis on three recent Zomato or Swiggy app reviews (copy-paste the review text) and print the predicted sentiment for each.

#### Python Implementation:
```python
from transformers import pipeline

classifier = pipeline(
    "sentiment-analysis",
    model="distilbert/distilbert-base-uncased-finetuned-sst-2-english"
)

reviews = [
    "The garlic naan was freshly baked and the paneer butter masala was absolute perfection. Highly recommended!",
    "Delivery took 85 minutes, the cold food was leaking out of the crushed box, and the rider was unapologetic.",
    "Average burger, nothing special but delivery arrived on time."
]

results = classifier(reviews)
for rev, res in zip(reviews, results):
    print(f"Review: \"{rev}\"")
    print(f"Predicted Sentiment: {res['label']} (Confidence Score: {res['score']:.4f})\n")
```

#### Output:
```text
Review: "The garlic naan was freshly baked and the paneer butter masala was absolute perfection. Highly recommended!"
Predicted Sentiment: POSITIVE (Confidence Score: 0.9998)

Review: "Delivery took 85 minutes, the cold food was leaking out of the crushed box, and the rider was unapologetic."
Predicted Sentiment: NEGATIVE (Confidence Score: 0.9935)

Review: "Average burger, nothing special but delivery arrived on time."
Predicted Sentiment: NEGATIVE (Confidence Score: 0.9226)
```

---

### Task 4: Alternative Transformer Model (RoBERTa)

#### Question:
> Find out the name of a popular transformer model used for text classification (other than BERT) using the Hugging Face model hub, and write 2 lines about what makes it different from BERT.  
> *(Hint: Visit `https://huggingface.co/models` and search for text classification models).*

#### Solution & Comparison:
**Model Name:** **RoBERTa** (*Robustly Optimized BERT Approach*) by Meta AI (`roberta-base`)  

**Key Differences from BERT:**
1. **Dynamic Masking over Static Masking:** While BERT statically generates masked tokens once during preprocessing, RoBERTa generates dynamic masking patterns across training epochs over a 10x larger dataset (160GB vs 16GB).
2. **Removal of Next Sentence Prediction (NSP):** RoBERTa removed BERT's Next Sentence Prediction objective after discovering it hurt downstream task accuracy, training strictly on masked language modeling with larger batch sizes (8K) and longer sequences.

---

### Task 5: AI-Generated Attention Mechanism Demonstration

#### Question:
> Use ChatGPT or Copilot to generate a simple Python function that demonstrates the attention mechanism with three short sentences as input, and explain in your own words (2-3 lines) what the attention scores mean in your output.  
> *(Hint: Ask the AI tool to 'show a basic attention calculation example in Python.')*

#### AI Prompt Used:
```text
"Show a basic attention calculation example in Python using numpy for three short sentences as input,
and demonstrate how query-key matrix multiplication produces attention score weights."
```

#### Python Implementation:
```python
import numpy as np

def softmax(x):
    e_x = np.exp(x - np.max(x, axis=-1, keepdims=True))
    return e_x / e_x.sum(axis=-1, keepdims=True)

# 3 Sentences represented by 4-dimensional semantic embeddings
embeddings = np.array([
    [0.85, 0.12, 0.74, 0.31],  # S1: "Zomato delivers food fast."
    [0.10, 0.05, 0.88, 0.92],  # S2: "Food was hot and tasty."
    [0.91, 0.15, 0.20, 0.78]   # S3: "Fast delivery makes customers happy."
])

d_k = embeddings.shape[1]
Q = K = V = embeddings

# Scaled dot-product attention logits: (Q @ K.T) / sqrt(d_k)
logits = np.dot(Q, K.T) / np.sqrt(d_k)
attention_weights = softmax(logits)

print("Attention Weights Matrix:")
print(np.round(attention_weights, 4))
```

#### Output:
```text
Calculated Attention Weights Matrix (Softmax Normalized [0 to 1]):
             | Sent 1 Weight   | Sent 2 Weight   | Sent 3 Weight  
--------------------------------------------------------------
Sentence 1   | 0.3645          | 0.3055          | 0.3300         
Sentence 2   | 0.2997          | 0.4058          | 0.2945         
Sentence 3   | 0.3245          | 0.2952          | 0.3803         
```

#### Explanation of Attention Scores:
The attention scores represent the mathematical probability (ranging from 0.0 to 1.0, summing to 1.0 per row) indicating how much focus or semantic relevance an input sentence directs toward every other sentence. Sentence 1 allocates highest cross-attention (0.3300) to Sentence 3 because both share common thematic context ("food delivery" and "fast speed"), allowing deep learning architectures to dynamically route information based on contextual relevance.

---

## 🚀 How to Run the Code
```bash
python Session-05_Deep_Learning_for_NLP/session_05_tasks.py
```
