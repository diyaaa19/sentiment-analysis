# SentimentScope: Transformer-Based Sentiment Analysis for CineScope
# A complete implementation for IMDB movie review sentiment classification

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split
from transformers import BertTokenizer, BertModel
from datasets import load_dataset
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report

# Set random seeds for reproducibility
torch.manual_seed(42)
np.random.seed(42)

# Check for GPU availability
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")


# ============================================================================
# STEP 1: LOAD AND EXPLORE THE DATASET
# ============================================================================

def load_and_explore_data():
    """Load IMDB dataset and perform exploratory data analysis"""
    print("Loading IMDB dataset...")
    dataset = load_dataset('imdb')
    
    train_data = dataset['train']
    test_data = dataset['test']
    
    print(f"\nDataset Statistics:")
    print(f"Training samples: {len(train_data)}")
    print(f"Test samples: {len(test_data)}")
    
    # Analyze sentiment distribution
    train_labels = [sample['label'] for sample in train_data]
    positive_count = sum(train_labels)
    negative_count = len(train_labels) - positive_count
    
    print(f"\nSentiment Distribution in Training Set:")
    print(f"Positive reviews: {positive_count} ({positive_count/len(train_labels)*100:.2f}%)")
    print(f"Negative reviews: {negative_count} ({negative_count/len(train_labels)*100:.2f}%)")
    
    # Analyze review lengths
    review_lengths = [len(sample['text'].split()) for sample in train_data[:1000]]
    print(f"\nReview Length Statistics (sample of 1000):")
    print(f"Average length: {np.mean(review_lengths):.2f} words")
    print(f"Median length: {np.median(review_lengths):.2f} words")
    print(f"Max length: {max(review_lengths)} words")
    
    return dataset


def visualize_data_distribution(dataset):
    """Visualize dataset characteristics"""
    train_data = dataset['train']
    
    # Sample for length analysis
    sample_size = 2000
    review_lengths = [len(sample['text'].split()) for sample in train_data[:sample_size]]
    labels = [sample['label'] for sample in train_data[:sample_size]]
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Plot 1: Review length distribution
    axes[0].hist(review_lengths, bins=50, edgecolor='black', alpha=0.7)
    axes[0].set_xlabel('Review Length (words)')
    axes[0].set_ylabel('Frequency')
    axes[0].set_title('Distribution of Review Lengths')
    axes[0].axvline(np.mean(review_lengths), color='red', linestyle='--', label=f'Mean: {np.mean(review_lengths):.0f}')
    axes[0].legend()
    
    # Plot 2: Sentiment distribution
    sentiment_counts = [labels.count(0), labels.count(1)]
    axes[1].bar(['Negative', 'Positive'], sentiment_counts, color=['#ff6b6b', '#51cf66'])
    axes[1].set_ylabel('Count')
    axes[1].set_title('Sentiment Distribution')
    axes[1].set_ylim(0, max(sentiment_counts) * 1.1)
    
    for i, v in enumerate(sentiment_counts):
        axes[1].text(i, v + 20, str(v), ha='center', fontweight='bold')
    
    plt.tight_layout()
    plt.savefig('dataset_analysis.png', dpi=300, bbox_inches='tight')
    print("\nVisualization saved as 'dataset_analysis.png'")
    plt.close()


def split_dataset(dataset, validation_split=0.1):
    """Split training data into train and validation sets"""
    train_data = dataset['train']
    test_data = dataset['test']
    
    # Calculate split sizes
    total_train = len(train_data)
    val_size = int(total_train * validation_split)
    train_size = total_train - val_size
    
    print(f"\nSplitting dataset:")
    print(f"Training set: {train_size} samples")
    print(f"Validation set: {val_size} samples")
    print(f"Test set: {len(test_data)} samples")
    
    return {
        'train': train_data,
        'validation': val_size,
        'test': test_data,
        'train_size': train_size
    }


# ============================================================================
# STEP 2: IMPLEMENT DATA LOADER
# ============================================================================

class IMDBDataset(Dataset):
    """Custom Dataset class for IMDB reviews"""
    
    def __init__(self, data, tokenizer, max_length=512):
        """
        Args:
            data: Dataset containing 'text' and 'label' fields
            tokenizer: BERT tokenizer for text preprocessing
            max_length: Maximum sequence length for padding/truncation
        """
        self.data = data
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        """Get a single sample from the dataset"""
        text = self.data[idx]['text']
        label = self.data[idx]['label']
        
        # Tokenize the text
        encoding = self.tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt'
        )
        
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'label': torch.tensor(label, dtype=torch.long)
        }


def create_data_loaders(dataset_splits, batch_size=16, max_length=512):
    """Create DataLoader objects for training, validation, and testing"""
    
    print("\nInitializing BERT tokenizer...")
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    
    # Get full training data
    full_train_data = dataset_splits['train']
    
    # Split into train and validation
    train_size = dataset_splits['train_size']
    val_size = dataset_splits['validation']
    
    # Create indices for splitting
    indices = list(range(len(full_train_data)))
    np.random.shuffle(indices)
    
    train_indices = indices[:train_size]
    val_indices = indices[train_size:]
    
    # Create subsets
    train_subset = full_train_data.select(train_indices)
    val_subset = full_train_data.select(val_indices)
    test_data = dataset_splits['test']
    
    print("\nCreating datasets...")
    train_dataset = IMDBDataset(train_subset, tokenizer, max_length)
    val_dataset = IMDBDataset(val_subset, tokenizer, max_length)
    test_dataset = IMDBDataset(test_data, tokenizer, max_length)
    
    print("Creating data loaders...")
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=2
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=2
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=2
    )
    
    print(f"\nData loaders created successfully!")
    print(f"Training batches: {len(train_loader)}")
    print(f"Validation batches: {len(val_loader)}")
    print(f"Test batches: {len(test_loader)}")
    
    return train_loader, val_loader, test_loader, tokenizer


# ============================================================================
# STEP 3: CUSTOMIZE TRANSFORMER MODEL
# ============================================================================

class SentimentTransformer(nn.Module):
    """Transformer-based model for binary sentiment classification"""
    
    def __init__(self, num_classes=2, dropout=0.3, freeze_bert=False):
        """
        Args:
            num_classes: Number of output classes (2 for binary classification)
            dropout: Dropout probability for regularization
            freeze_bert: Whether to freeze BERT parameters during training
        """
        super(SentimentTransformer, self).__init__()
        
        # Load pre-trained BERT model
        self.bert = BertModel.from_pretrained('bert-base-uncased')
        
        # Optionally freeze BERT parameters
        if freeze_bert:
            for param in self.bert.parameters():
                param.requires_grad = False
        
        # Get BERT hidden size
        self.bert_hidden_size = self.bert.config.hidden_size  # 768 for bert-base
        
        # Classification head
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(self.bert_hidden_size, num_classes)
        
    def forward(self, input_ids, attention_mask):
        """
        Forward pass through the model
        
        Args:
            input_ids: Token IDs from tokenizer
            attention_mask: Attention mask to ignore padding tokens
            
        Returns:
            Logits for each class
        """
        # Get BERT outputs
        outputs = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask
        )
        
        # Extract [CLS] token representation (first token)
        pooled_output = outputs.pooler_output
        
        # Apply dropout for regularization
        pooled_output = self.dropout(pooled_output)
        
        # Get classification logits
        logits = self.classifier(pooled_output)
        
        return logits


def initialize_model(num_classes=2, dropout=0.3, freeze_bert=False):
    """Initialize the sentiment analysis model"""
    print("\nInitializing SentimentTransformer model...")
    model = SentimentTransformer(
        num_classes=num_classes,
        dropout=dropout,
        freeze_bert=freeze_bert
    )
    
    model = model.to(device)
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    
    return model


# ============================================================================
# STEP 4: TRAIN THE MODEL
# ============================================================================

def train_epoch(model, data_loader, criterion, optimizer, device):
    """Train the model for one epoch"""
    model.train()
    total_loss = 0
    correct_predictions = 0
    total_samples = 0
    
    progress_bar = tqdm(data_loader, desc='Training')
    
    for batch in progress_bar:
        # Move batch to device
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['label'].to(device)
        
        # Zero gradients
        optimizer.zero_grad()
        
        # Forward pass
        logits = model(input_ids, attention_mask)
        
        # Calculate loss
        loss = criterion(logits, labels)
        
        # Backward pass
        loss.backward()
        
        # Gradient clipping to prevent exploding gradients
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        # Update weights
        optimizer.step()
        
        # Calculate accuracy
        predictions = torch.argmax(logits, dim=1)
        correct_predictions += (predictions == labels).sum().item()
        total_samples += labels.size(0)
        total_loss += loss.item()
        
        # Update progress bar
        progress_bar.set_postfix({
            'loss': loss.item(),
            'acc': correct_predictions / total_samples
        })
    
    avg_loss = total_loss / len(data_loader)
    accuracy = correct_predictions / total_samples
    
    return avg_loss, accuracy


def validate_epoch(model, data_loader, criterion, device):
    """Validate the model on the validation set"""
    model.eval()
    total_loss = 0
    correct_predictions = 0
    total_samples = 0
    
    with torch.no_grad():
        progress_bar = tqdm(data_loader, desc='Validation')
        
        for batch in progress_bar:
            # Move batch to device
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['label'].to(device)
            
            # Forward pass
            logits = model(input_ids, attention_mask)
            
            # Calculate loss
            loss = criterion(logits, labels)
            
            # Calculate accuracy
            predictions = torch.argmax(logits, dim=1)
            correct_predictions += (predictions == labels).sum().item()
            total_samples += labels.size(0)
            total_loss += loss.item()
            
            # Update progress bar
            progress_bar.set_postfix({
                'loss': loss.item(),
                'acc': correct_predictions / total_samples
            })
    
    avg_loss = total_loss / len(data_loader)
    accuracy = correct_predictions / total_samples
    
    return avg_loss, accuracy


def train_model(model, train_loader, val_loader, num_epochs=3, learning_rate=2e-5):
    """Complete training loop with validation"""
    
    # Define loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    
    # Learning rate scheduler
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=1, verbose=True
    )
    
    # Track metrics
    history = {
        'train_loss': [],
        'train_acc': [],
        'val_loss': [],
        'val_acc': []
    }
    
    best_val_acc = 0
    
    print(f"\nStarting training for {num_epochs} epochs...")
    print(f"Learning rate: {learning_rate}")
    
    for epoch in range(num_epochs):
        print(f"\n{'='*60}")
        print(f"Epoch {epoch + 1}/{num_epochs}")
        print(f"{'='*60}")
        
        # Train
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device)
        
        # Validate
        val_loss, val_acc = validate_epoch(model, val_loader, criterion, device)
        
        # Update learning rate
        scheduler.step(val_loss)
        
        # Save metrics
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)
        
        # Print epoch summary
        print(f"\nEpoch {epoch + 1} Summary:")
        print(f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f}")
        print(f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f}")
        
        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), 'best_sentiment_model.pt')
            print(f"✓ New best model saved! (Val Acc: {val_acc:.4f})")
    
    print(f"\n{'='*60}")
    print(f"Training completed!")
    print(f"Best validation accuracy: {best_val_acc:.4f}")
    print(f"{'='*60}")
    
    return history


def plot_training_history(history):
    """Visualize training and validation metrics"""
    epochs = range(1, len(history['train_loss']) + 1)
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Plot loss
    axes[0].plot(epochs, history['train_loss'], 'b-o', label='Training Loss')
    axes[0].plot(epochs, history['val_loss'], 'r-o', label='Validation Loss')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss')
    axes[0].set_title('Training and Validation Loss')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # Plot accuracy
    axes[1].plot(epochs, history['train_acc'], 'b-o', label='Training Accuracy')
    axes[1].plot(epochs, history['val_acc'], 'r-o', label='Validation Accuracy')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Accuracy')
    axes[1].set_title('Training and Validation Accuracy')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('training_history.png', dpi=300, bbox_inches='tight')
    print("\nTraining history saved as 'training_history.png'")
    plt.close()


# ============================================================================
# STEP 5: TEST AND EVALUATE THE MODEL
# ============================================================================

def test_model(model, test_loader, device):
    """Test the model on the test set"""
    model.eval()
    correct_predictions = 0
    total_samples = 0
    all_predictions = []
    all_labels = []
    
    print("\nEvaluating on test set...")
    
    with torch.no_grad():
        progress_bar = tqdm(test_loader, desc='Testing')
        
        for batch in progress_bar:
            # Move batch to device
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['label'].to(device)
            
            # Forward pass
            logits = model(input_ids, attention_mask)
            
            # Get predictions
            predictions = torch.argmax(logits, dim=1)
            
            # Store predictions and labels
            all_predictions.extend(predictions.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            
            # Calculate accuracy
            correct_predictions += (predictions == labels).sum().item()
            total_samples += labels.size(0)
            
            # Update progress bar
            progress_bar.set_postfix({
                'acc': correct_predictions / total_samples
            })
    
    test_accuracy = correct_predictions / total_samples
    
    print(f"\n{'='*60}")
    print(f"TEST RESULTS")
    print(f"{'='*60}")
    print(f"Test Accuracy: {test_accuracy:.4f} ({test_accuracy*100:.2f}%)")
    print(f"Correct Predictions: {correct_predictions}/{total_samples}")
    
    # Generate classification report
    print(f"\n{classification_report(all_labels, all_predictions, target_names=['Negative', 'Positive'])}")
    
    # Create confusion matrix
    cm = confusion_matrix(all_labels, all_predictions)
    
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['Negative', 'Positive'],
                yticklabels=['Negative', 'Positive'])
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    plt.savefig('confusion_matrix.png', dpi=300, bbox_inches='tight')
    print("\nConfusion matrix saved as 'confusion_matrix.png'")
    plt.close()
    
    return test_accuracy


def predict_sentiment(model, tokenizer, text, device, max_length=512):
    """Predict sentiment for a single text"""
    model.eval()
    
    # Tokenize input
    encoding = tokenizer.encode_plus(
        text,
        add_special_tokens=True,
        max_length=max_length,
        padding='max_length',
        truncation=True,
        return_attention_mask=True,
        return_tensors='pt'
    )
    
    input_ids = encoding['input_ids'].to(device)
    attention_mask = encoding['attention_mask'].to(device)
    
    with torch.no_grad():
        logits = model(input_ids, attention_mask)
        probabilities = torch.softmax(logits, dim=1)
        prediction = torch.argmax(probabilities, dim=1).item()
    
    sentiment = 'Positive' if prediction == 1 else 'Negative'
    confidence = probabilities[0][prediction].item()
    
    return sentiment, confidence


# ============================================================================
# MAIN EXECUTION PIPELINE
# ============================================================================

def main():
    """Main execution pipeline for SentimentScope project"""
    
    print("="*60)
    print("SENTIMENTSCOPE: Transformer-Based Sentiment Analysis")
    print("="*60)
    
    # Hyperparameters
    BATCH_SIZE = 16
    MAX_LENGTH = 256  # Reduced for faster training
    NUM_EPOCHS = 3
    LEARNING_RATE = 2e-5
    VALIDATION_SPLIT = 0.1
    DROPOUT = 0.3
    FREEZE_BERT = False
    
    # Step 1: Load and explore data
    dataset = load_and_explore_data()
    visualize_data_distribution(dataset)
    dataset_splits = split_dataset(dataset, validation_split=VALIDATION_SPLIT)
    
    # Step 2: Create data loaders
    train_loader, val_loader, test_loader, tokenizer = create_data_loaders(
        dataset_splits, 
        batch_size=BATCH_SIZE,
        max_length=MAX_LENGTH
    )
    
    # Step 3: Initialize model
    model = initialize_model(
        num_classes=2,
        dropout=DROPOUT,
        freeze_bert=FREEZE_BERT
    )
    
    # Step 4: Train model
    history = train_model(
        model,
        train_loader,
        val_loader,
        num_epochs=NUM_EPOCHS,
        learning_rate=LEARNING_RATE
    )
    
    # Plot training history
    plot_training_history(history)
    
    # Load best model
    print("\nLoading best model for testing...")
    model.load_state_dict(torch.load('best_sentiment_model.pt'))
    
    # Step 5: Test model
    test_accuracy = test_model(model, test_loader, device)
    
    # Verify 75% threshold
    if test_accuracy > 0.75:
        print(f"\n✓ SUCCESS: Test accuracy ({test_accuracy:.4f}) exceeds 75% threshold!")
    else:
        print(f"\n✗ WARNING: Test accuracy ({test_accuracy:.4f}) is below 75% threshold.")
        print("Consider training for more epochs or adjusting hyperparameters.")
    
    # Test with sample predictions
    print("\n" + "="*60)
    print("SAMPLE PREDICTIONS")
    print("="*60)
    
    sample_reviews = [
        "This movie was absolutely fantastic! I loved every minute of it.",
        "Terrible waste of time. The plot made no sense and the acting was awful.",
        "Pretty decent movie with some good moments, though not perfect.",
        "I was bored throughout the entire film. Would not recommend."
    ]
    
    for i, review in enumerate(sample_reviews, 1):
        sentiment, confidence = predict_sentiment(model, tokenizer, review, device, MAX_LENGTH)
        print(f"\nReview {i}: \"{review[:80]}...\"")
        print(f"Prediction: {sentiment} (Confidence: {confidence:.4f})")
    
    print("\n" + "="*60)
    print("PROJECT COMPLETED SUCCESSFULLY!")
    print("="*60)
    print("\nGenerated files:")
    print("- best_sentiment_model.pt (trained model)")
    print("- dataset_analysis.png (data visualization)")
    print("- training_history.png (training metrics)")
    print("- confusion_matrix.png (test results)")


if __name__ == "__main__":
    main()
