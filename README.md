# 🎬 IMDB Sentiment Analysis with LSTM

This project demonstrates how to build and evaluate a sentiment analysis model using the **IMDB movie reviews dataset**. The model is implemented using a **Long Short-Term Memory (LSTM)** network, a type of recurrent neural network (RNN) known for its ability to capture long-term dependencies in text data.  

Through this tutorial, you will learn:
- How to preprocess and clean text data for sentiment analysis.
- How to build and train an LSTM model using Keras.
- How to evaluate the model using metrics such as accuracy, precision, recall, F1-score, and ROC-AUC.
- How to visualize results using plots and word clouds.

---

## 🚀 **Objective**
The goal of this project is to:
1. Classify IMDB movie reviews as **positive** or **negative**.
2. Analyze model performance using a range of evaluation metrics.
3. Provide insights into the strengths and weaknesses of the model through comprehensive visualizations.

---

## 📂 **Dataset Overview**
- **Dataset**: IMDB Movie Reviews (from TensorFlow Keras Datasets)  
- **Training Samples**: 25,000  
- **Testing Samples**: 25,000  
- **Classes**:
  - 0 → Negative Sentiment  
  - 1 → Positive Sentiment  

The dataset contains movie reviews in text format with binary labels (positive/negative).

---

## 🏗️ **Model Architecture**
The model is implemented using TensorFlow's Keras API with the following layers:
1. **Embedding Layer** – Converts text tokens to dense vectors.
2. **LSTM Layer** – Captures long-term dependencies.
3. **Dropout Layer** – Prevents overfitting.
4. **Dense Layer** – Intermediate transformation.
5. **Output Layer** – Sigmoid activation for binary classification.

### **Hyperparameters**
| Parameter | Value | Description |
|-----------|-------|-------------|
| `embedding_dim` | 128 | Dimension of word embeddings |
| `LSTM units` | 64 | Number of LSTM hidden units |
| `dropout_rate` | 0.5 | Fraction of neurons dropped during training |
| `batch_size` | 64 | Number of samples per batch |
| `epochs` | 10 | Number of training epochs |
| `learning_rate` | 0.001 | Adam optimizer learning rate |

---

## 🔎 **Performance Metrics**
| Metric | Value |
|--------|-------|
| **Accuracy** | 0.50 (50%) |
| **Precision** | 0.50 |
| **Recall** | 1.00 |
| **F1-Score** | 0.67 |
| **ROC AUC** | 0.510 |

---

## 📊 **Visualizations**
### 1. **ROC Curve**
- The model achieves an AUC score of **0.510**, which is close to random guessing.
  

---

### 2. **Learning Curve (EMA)**
- Training loss decreases steadily, but validation loss increases → **Overfitting**.

---

### 3. **Confusion Matrix**
- Model predicts positive sentiment for most reviews, leading to poor negative class recall.

---

### 4. **Word Cloud**
- **Positive and Negative Word Clouds** show overlapping terms in both classes, highlighting model confusion.

---

## 📥 **Setup and Installation**
### **1. Clone the Repository**
```bash
git clone https://github.com/MuhammadAdnan6/MLNN.git
