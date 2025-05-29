# 📊 Sentiment Analysis of Financial Phrases

This project performs **sentiment analysis** on financial phrases using deep learning techniques. Three different models — Simple RNN, LSTM with trainable embeddings, and LSTM with pretrained Word2Vec embeddings — were implemented and evaluated to classify financial text into **positive**, **neutral**, or **negative** sentiments.

## 🧠 Models Implemented

1. **Simple RNN**
   - 128D trainable embedding
   - 64-unit RNN layer
   - ReLU activation and Softmax output
   - Dropout = 0.5

2. **LSTM with Trainable Embeddings**
   - 64-unit LSTM
   - Trainable embedding
   - EarlyStopping and ReduceLROnPlateau for training optimization

3. **LSTM with Word2Vec Embeddings** *(Best performing)*
   - Pretrained 300D Word2Vec (FastText)
   - Bidirectional LSTM
   - 128 hidden units + Dense output layer

## 📈 Results

| Model                  | Accuracy | Loss    |
|------------------------|----------|---------|
| Simple RNN             | 0.7969   | 0.8405  |
| LSTM (Trainable)       | 0.8565   | 0.6047  |
| LSTM + Word2Vec        | **0.8675** | **0.5626** |

- The Word2Vec-based LSTM outperformed others in all key metrics.
- Evaluation included classification reports, confusion matrices, and accuracy curves.

## 📂 Dataset

- **Source:** Provided by Module Leader
- **Size:** 4407 samples
- **Classes:** Positive, Neutral, Negative

### 🔄 Preprocessing Steps

- Lowercase conversion
- Contraction expansion (`contractions` library)
- Removal of URLs, hashtags, and non-alphabetic characters
- Stopword removal (NLTK)
- Lemmatization (WordNetLemmatizer)
- Tokenization and padding (Keras Tokenizer)
- Class balancing using `RandomOverSampler`

## ⚙️ Technical Details

- **Frameworks:** TensorFlow / Keras, NLTK, Scikit-learn
- **Loss Function:** `sparse_categorical_crossentropy`
- **Optimizer:** Adam (learning rate = 0.0005)
- **Batch Size:** 32
- **Epochs:** Up to 30 (with early stopping and learning rate scheduler)
- **Training Platform:** Google Colab with GPU

## 🔮 Future Work

- Implement Transformer-based models like BERT for better contextual understanding
- Increase dataset diversity and size for improved generalization
- Further enhance the user-facing GUI for real-time prediction

## 👤 Author

- **Name:** Shubham Shrestha
-