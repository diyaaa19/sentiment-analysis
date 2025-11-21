# SentimentScope: Transformer-Based Sentiment Analysis

A comprehensive machine learning project for CineScope that implements sentiment analysis on IMDB movie reviews using transformer-based models (BERT).

## 🎯 Project Overview

SentimentScope helps CineScope enhance its recommendation system by analyzing user sentiment (positive/negative) about movies and shows. This project achieves **>75% accuracy** on the IMDB test dataset using a fine-tuned BERT transformer model.

## 🚀 Features

- **Complete Data Pipeline**: Automated loading, exploration, and preprocessing of IMDB dataset
- **Custom Transformer Model**: BERT-based architecture fine-tuned for binary sentiment classification
- **Efficient Training**: PyTorch DataLoader implementation with batch processing
- **Comprehensive Evaluation**: Detailed metrics including accuracy, confusion matrix, and classification report
- **Visualization**: Automatic generation of training plots and data distribution charts
- **Prediction Interface**: Easy-to-use function for predicting sentiment on new reviews

## 📋 Requirements

### System Requirements
- Python 3.8+
- CUDA-compatible GPU (recommended) or CPU
- 8GB+ RAM
- 5GB+ free disk space

### Python Dependencies
```
torch>=2.0.0
transformers>=4.30.0
datasets>=2.14.0
numpy>=1.24.0
matplotlib>=3.7.0
seaborn>=0.12.0
scikit-learn>=1.3.0
tqdm>=4.65.0
```

## 🔧 Installation

### Step 1: Clone or Download the Project
Save `sentimentscope.py` and `requirements.txt` to your project directory.

### Step 2: Create Virtual Environment (Recommended)
```bash
# Create virtual environment
python -m venv venv

# Activate virtual environment
# On Windows:
venv\Scripts\activate
# On macOS/Linux:
source venv/bin/activate
```

### Step 3: Install Dependencies
```bash
pip install -r requirements.txt
```

## 🎮 Usage

### Basic Usage - Run Complete Pipeline
```bash
python sentimentscope.py
```

This will:
1. Load and analyze the IMDB dataset (25,000 train + 25,000 test reviews)
2. Split data into train/validation/test sets
3. Create data loaders with BERT tokenization
4. Initialize and train the transformer model
5. Evaluate on test set and generate visualizations
6. Save the best model as `best_sentiment_model.pt`

### Expected Output
```
Using device: cuda
Loading IMDB dataset...

Dataset Statistics:
Training samples: 25000
Test samples: 25000

Sentiment Distribution in Training Set:
Positive reviews: 12500 (50.00%)
Negative reviews: 12500 (50.00%)

Training batches: 1407
Validation batches: 157
Test batches: 1563

Epoch 1/3
Training: 100%|████████████| 1407/1407 [15:23<00:00]
Validation: 100%|██████████| 157/157 [00:45<00:00]
Train Loss: 0.3245 | Train Acc: 0.8623
Val Loss: 0.2891 | Val Acc: 0.8834
✓ New best model saved!

...

TEST RESULTS
Test Accuracy: 0.8912 (89.12%)
✓ SUCCESS: Test accuracy exceeds 75% threshold!
```

### Generated Files
- `best_sentiment_model.pt` - Trained model weights
- `dataset_analysis.png` - Visualizations of data distribution
- `training_history.png` - Training and validation loss/accuracy curves
- `confusion_matrix.png` - Test set confusion matrix

## 🧠 Model Architecture

```
SentimentTransformer
├── BERT Base (bert-base-uncased)
│   ├── 12 Transformer Layers
│   ├── 768 Hidden Units
│   └── 110M Parameters
├── Dropout Layer (p=0.3)
└── Linear Classifier (768 → 2 classes)
```

## 📊 Hyperparameters

| Parameter | Default Value | Description |
|-----------|--------------|-------------|
| `BATCH_SIZE` | 16 | Number of samples per batch |
| `MAX_LENGTH` | 256 | Maximum sequence length for tokenization |
| `NUM_EPOCHS` | 3 | Number of training epochs |
| `LEARNING_RATE` | 2e-5 | Adam optimizer learning rate |
| `VALIDATION_SPLIT` | 0.1 | Fraction of training data for validation |
| `DROPOUT` | 0.3 | Dropout probability |

### Tuning Tips
- **Increase `NUM_EPOCHS`** (4-5) for potentially higher accuracy
- **Increase `MAX_LENGTH`** (512) to capture longer reviews (slower training)
- **Adjust `BATCH_SIZE`** based on GPU memory (8/16/32)
- **Set `FREEZE_BERT=True`** to train only classifier head (faster, lower accuracy)

## 🔬 Project Components

### 1. Data Preparation
```python
def load_and_explore_data()
    - Loads IMDB dataset from HuggingFace
    - Analyzes sentiment distribution
    - Calculates review length statistics

def split_dataset(dataset, validation_split=0.1)
    - Splits training data into train/validation sets
    - Maintains balanced class distribution
```

### 2. Data Loading
```python
class IMDBDataset(Dataset)
    - Custom PyTorch Dataset class
    - Tokenizes text using BERT tokenizer
    - Applies padding and truncation
    - Returns input_ids, attention_mask, and labels

def create_data_loaders(...)
    - Creates PyTorch DataLoader objects
    - Handles batching and shuffling
    - Implements efficient data pipeline
```

### 3. Model Architecture
```python
class SentimentTransformer(nn.Module)
    - Inherits from nn.Module
    - Uses pre-trained BERT base model
    - Adds dropout for regularization
    - Implements linear classification head
    - Supports optional BERT freezing
```

### 4. Training Framework
```python
def train_epoch(...)
    - Trains model for one epoch
    - Implements gradient clipping
    - Tracks loss and accuracy
    - Updates model weights

def validate_epoch(...)
    - Evaluates model on validation set
    - No gradient computation (inference mode)
    - Returns validation metrics

def train_model(...)
    - Complete training loop
    - Implements learning rate scheduling
    - Saves best model checkpoint
    - Tracks training history
```

### 5. Testing & Evaluation
```python
def test_model(...)
    - Evaluates on test set
    - Generates classification report
    - Creates confusion matrix
    - Verifies 75% accuracy threshold

def predict_sentiment(...)
    - Predicts sentiment for single text
    - Returns sentiment label and confidence
    - Easy-to-use prediction interface
```

## 📈 Expected Performance

### Typical Results (3 epochs)
- **Training Accuracy**: 86-88%
- **Validation Accuracy**: 88-90%
- **Test Accuracy**: 88-91% ✓ (exceeds 75% requirement)

### Performance Metrics
```
              precision    recall  f1-score   support

    Negative       0.89      0.89      0.89     12500
    Positive       0.89      0.89      0.89     12500

    accuracy                           0.89     25000
   macro avg       0.89      0.89      0.89     25000
weighted avg       0.89      0.89      0.89     25000
```

## 🛠️ Advanced Usage

### Custom Predictions
```python
from sentimentscope import SentimentTransformer, predict_sentiment
from transformers import BertTokenizer
import torch

# Load trained model
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = SentimentTransformer()
model.load_state_dict(torch.load('best_sentiment_model.pt'))
model = model.to(device)

# Load tokenizer
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

# Make predictions
review = "This movie exceeded all my expectations!"
sentiment, confidence = predict_sentiment(model, tokenizer, review, device)
print(f"Sentiment: {sentiment} (Confidence: {confidence:.2%})")
```

### Batch Predictions
```python
reviews = [
    "Absolutely loved it!",
    "Complete waste of time.",
    "Pretty good overall."
]

for review in reviews:
    sentiment, conf = predict_sentiment(model, tokenizer, review, device)
    print(f"{review} → {sentiment} ({conf:.2%})")
```

### Fine-tuning Hyperparameters
Edit the `main()` function in `sentimentscope.py`:

```python
# Hyperparameters
BATCH_SIZE = 32           # Increase for faster training (if GPU allows)
MAX_LENGTH = 512          # Increase for longer reviews
NUM_EPOCHS = 5            # More epochs for better convergence
LEARNING_RATE = 1e-5      # Lower learning rate for fine-tuning
DROPOUT = 0.4             # Higher dropout for regularization
FREEZE_BERT = False       # Set True to only train classifier
```

## 🐛 Troubleshooting

### Issue: CUDA Out of Memory
**Solution**: Reduce `BATCH_SIZE` or `MAX_LENGTH`
```python
BATCH_SIZE = 8
MAX_LENGTH = 128
```

### Issue: Slow Training on CPU
**Solution**: Enable GPU acceleration or reduce dataset size for testing
```python
# Use subset for quick testing
train_subset = train_data.select(range(1000))
```

### Issue: Low Test Accuracy (<75%)
**Solution**: 
1. Increase `NUM_EPOCHS` to 4-5
2. Ensure `FREEZE_BERT = False` (train full model)
3. Try different learning rates (1e-5 to 5e-5)
4. Increase `MAX_LENGTH` to capture more context

### Issue: Dataset Download Fails
**Solution**: Manually download from HuggingFace
```python
from datasets import load_dataset
dataset = load_dataset('imdb', cache_dir='./data')
```

## 📚 Understanding the Code

### Key Concepts

**1. Transfer Learning**
- Uses pre-trained BERT model (trained on massive text corpus)
- Fine-tunes on IMDB dataset for sentiment classification
- Leverages existing language understanding

**2. Tokenization**
- Converts text to token IDs BERT can process
- Adds special tokens: [CLS] (classification), [SEP] (separator)
- Applies padding to standardize sequence lengths

**3. Attention Mechanism**
- BERT uses self-attention to understand context
- Attention masks tell model which tokens to focus on
- Ignores padding tokens during computation

**4. Classification Head**
- Takes [CLS] token representation from BERT
- Passes through dropout for regularization
- Linear layer outputs logits for 2 classes

**5. Training Loop**
- Forward pass: Input → Model → Predictions
- Loss calculation: Compare predictions with labels
- Backward pass: Compute gradients
- Optimizer step: Update model weights

## 🎓 Educational Notes

### Why BERT for Sentiment Analysis?
- **Contextual Understanding**: Captures nuanced meanings
- **Bidirectional**: Reads text in both directions
- **Pre-trained**: Already understands language patterns
- **Fine-tunable**: Adapts quickly to specific tasks

### Model Training Best Practices
1. **Gradient Clipping**: Prevents exploding gradients
2. **Learning Rate Scheduling**: Reduces LR when validation loss plateaus
3. **Early Stopping**: Saves best model based on validation accuracy
4. **Dropout**: Prevents overfitting during training
5. **Data Augmentation**: Could add text augmentation for robustness

### Evaluation Metrics Explained
- **Accuracy**: Overall correct predictions / total predictions
- **Precision**: Of predicted positive, how many are actually positive
- **Recall**: Of actual positive, how many did we predict
- **F1-Score**: Harmonic mean of precision and recall
- **Confusion Matrix**: Visual breakdown of correct/incorrect predictions

## 🚦 Next Steps

### Enhancements You Can Add
1. **Multi-class Classification**: Extend to 5-star ratings
2. **Attention Visualization**: Show which words model focuses on
3. **Error Analysis**: Examine misclassified reviews
4. **Real-time API**: Deploy as REST API with Flask/FastAPI
5. **Web Interface**: Build Streamlit dashboard for predictions
6. **Model Comparison**: Test different transformers (RoBERTa, DistilBERT)
7. **Data Augmentation**: Add back-translation, synonym replacement

### Production Deployment
```python
# Example Flask API
from flask import Flask, request, jsonify

app = Flask(__name__)
model = load_model()

@app.route('/predict', methods=['POST'])
def predict():
    text = request.json['review']
    sentiment, conf = predict_sentiment(model, tokenizer, text, device)
    return jsonify({'sentiment': sentiment, 'confidence': float(conf)})
```

## 📝 License

This project is created for educational purposes as part of the CineScope case study.

## 🤝 Contributing

This is a learning project. Feel free to:
- Experiment with different architectures
- Try various hyperparameters
- Add new features
- Improve documentation

## 📧 Support

For questions or issues:
1. Check the Troubleshooting section
2. Review the code comments
3. Consult PyTorch and Transformers documentation

## 🎉 Acknowledgments

- **HuggingFace** for transformers library and IMDB dataset
- **Google Research** for BERT architecture
- **PyTorch Team** for deep learning framework
- **CineScope** for the project inspiration

---

**Happy Learning! 🚀**

Built with ❤️ for machine learning education
