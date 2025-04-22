# 🇧🇩 Bangla Sentiment Analysis

<div align="center">
  <img src="https://via.placeholder.com/800x200/5D44B3/ffffff?text=Bangla+Sentiment+Analysis" alt="Bangla Sentiment Analysis Banner"/>
  <p>A deep learning approach for sentiment analysis on Bangla text using Word2Vec embeddings and LSTM networks</p>
</div>

## 📊 Overview

This project implements a binary sentiment classifier for Bangla text that can classify text as either "positive" or "negative". The model uses Word2Vec for word embeddings and LSTM (Long Short-Term Memory) for sequence learning.

<div align="center">
  <img src="https://via.placeholder.com/600x300/f0f0f0/000000?text=Positive+vs+Negative+Classification" alt="Positive vs Negative Classification"/>
</div>

## 🧠 Model Architecture

<div align="center">
  <img src="https://via.placeholder.com/800x400/e6f7ff/000000?text=Model+Architecture" alt="Model Architecture Diagram"/>
</div>

The sentiment analysis model consists of several key components:

### Word2Vec Embeddings
```
┌─────────────────────────┐
│ Word2Vec Configuration  │
├─────────────┬───────────┤
│ Dimensions  │ 100       │
│ Window Size │ 3         │
│ Algorithm   │ Skip-gram │
│ Min Count   │ 3         │
│ Workers     │ 10        │
│ Iterations  │ 100       │
│ Epochs      │ 40        │
└─────────────┴───────────┘
```

### Neural Network Architecture
```
┌───────────────────────────────────┐
│ INPUT                             │
│  Bangla Text Sequence (50 tokens) │
└───────────────┬───────────────────┘
                ▼
┌───────────────────────────────────┐
│ EMBEDDING LAYER                   │
│  Pre-trained Word2Vec (100 dim)   │
└───────────────┬───────────────────┘
                ▼
┌───────────────────────────────────┐
│ DROPOUT LAYER (0.5)               │
└───────────────┬───────────────────┘
                ▼
┌───────────────────────────────────┐
│ LSTM LAYER                        │
│  100 units, dropout=0.2,          │
│  recurrent_dropout=0.2            │
└───────────────┬───────────────────┘
                ▼
┌───────────────────────────────────┐
│ DENSE LAYER                       │
│  1 unit, sigmoid activation       │
└───────────────┬───────────────────┘
                ▼
┌───────────────────────────────────┐
│ OUTPUT                            │
│  Binary Sentiment (0-1)           │
└───────────────────────────────────┘
```

## 📈 Training Progress

<div align="center">
  <table>
    <tr>
      <td><img src="https://via.placeholder.com/400x300/e6f7ff/000000?text=Accuracy+Graph" alt="Accuracy Graph"/></td>
      <td><img src="https://via.placeholder.com/400x300/e6f7ff/000000?text=Loss+Graph" alt="Loss Graph"/></td>
    </tr>
    <tr>
      <td align="center">Model Accuracy</td>
      <td align="center">Model Loss</td>
    </tr>
  </table>
</div>

## 📁 Dataset

<div align="center">
  <img src="https://via.placeholder.com/600x300/f0f0f0/000000?text=Dataset+Distribution" alt="Dataset Distribution"/>
</div>

The model is trained on a custom Bangla sentiment dataset:
- Binary classes: "positive" and "negative"
- Training/Testing split: 80%/20%
- Text sequences padded to length 50

## 📋 Requirements

```
📦 Python 3.x
├── 📦 pandas
├── 📦 sklearn
├── 📦 gensim
├── 📦 keras
├── 📦 tensorflow
├── 📦 matplotlib
└── 📦 numpy
```

## 🚀 Usage

### Training Pipeline

<div align="center">
  <img src="https://via.placeholder.com/800x200/f0f0f0/000000?text=Training+Pipeline" alt="Training Pipeline"/>
</div>

```python
# Load and prepare data
df = pd.read_csv("your_dataset.csv", encoding="utf8", names=["bangla_text", "target"])

# Split data
df_train, df_test = train_test_split(df, test_size=0.2, random_state=42, shuffle=True)

# Create Word2Vec model
documents = [text.split() for text in df.bangla_text]
w2v_model = Word2Vec(size=100, window=3, min_count=3, workers=10, sg=1, iter=100)
w2v_model.build_vocab(documents)
w2v_model.train(documents, total_examples=len(documents), epochs=40)

# Prepare sequences
tokenizer = Tokenizer()
tokenizer.fit_on_texts(df_train.bangla_text)
x_train = pad_sequences(tokenizer.texts_to_sequences(df_train.bangla_text), maxlen=50)
x_test = pad_sequences(tokenizer.texts_to_sequences(df_test.bangla_text), maxlen=50)

# Train model
history = model.fit(x_train, y_train, batch_size=1024, epochs=20, validation_split=0.1)
```

### Inference

<div align="center">
  <img src="https://via.placeholder.com/600x200/f0f0f0/000000?text=Prediction+Pipeline" alt="Prediction Pipeline"/>
</div>

```python
def predict_sentiment(text):
    # Tokenize and pad the input text
    x_test = pad_sequences(tokenizer.texts_to_sequences([text]), maxlen=50)
    
    # Get prediction score
    score = model.predict([x_test])[0]
    
    # Classify based on threshold
    label = "negative" if score < 0.5 else "positive"
    
    return {"label": label, "score": float(score)}

# Example usage
result = predict_sentiment("আমি তাকে সব বলি সবসময়")
print(result)
```



## 🔍 Examples

<div align="center">
  <table>
    <tr>
      <th>Bangla Text</th>
      <th>Prediction</th>
      <th>Score</th>
    </tr>
    <tr>
      <td>সে খুব নিষ্পাপ মানুষ তাই তাকে এত ভালোবাসি</td>
      <td>🟢 Positive</td>
      <td>0.92</td>
    </tr>
    <tr>
      <td>সে খারাপ ছেলে</td>
      <td>🔴 Negative</td>
      <td>0.12</td>
    </tr>
    <tr>
      <td>তিনি কাজ করতে চান না</td>
      <td>🔴 Negative</td>
      <td>0.31</td>
    </tr>
    <tr>
      <td>সে আমাকে খুব পছন্দ করে</td>
      <td>🟢 Positive</td>
      <td>0.87</td>
    </tr>
  </table>
</div>

## 🚀 Future Improvements

<div align="center">
  <img src="https://via.placeholder.com/800x200/f0f0f0/000000?text=Future+Development+Roadmap" alt="Future Development Roadmap"/>
</div>

- 📈 Experiment with different embedding dimensions
- 🔄 Try different architectures (BiLSTM, Transformer-based models)
- 🏷️ Add support for multi-class classification
- ❌ Improve handling of negation in Bangla text
- 📚 Expand the dataset with more examples


---

<div align="center">
  <p>Bangla NLP Research</p>
</div>
