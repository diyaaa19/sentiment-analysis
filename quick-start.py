"""
Quick Start Script for SentimentScope
Run this for a faster demo with reduced dataset size
"""

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer, BertModel
from datasets import load_dataset
from tqdm import tqdm

# Use subset for quick testing
SUBSET_SIZE = 2000  # Use smaller dataset for quick demo
BATCH_SIZE = 16
MAX_LENGTH = 128
NUM_EPOCHS = 2
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

print(f"Quick Start Mode - Using {SUBSET_SIZE} samples")
print(f"Device: {device}")


class IMDBDataset(Dataset):
    def __init__(self, data, tokenizer, max_length=128):
        self.data = data
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        text = self.data[idx]['text']
        label = self.data[idx]['label']
        
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


class SentimentTransformer(nn.Module):
    def __init__(self, num_classes=2, dropout=0.3):
        super(SentimentTransformer, self).__init__()
        self.bert = BertModel.from_pretrained('bert-base-uncased')
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(self.bert.config.hidden_size, num_classes)
        
    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        pooled_output = outputs.pooler_output
        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)
        return logits


def train_epoch(model, data_loader, criterion, optimizer):
    model.train()
    total_loss = 0
    correct = 0
    total = 0
    
    for batch in tqdm(data_loader, desc='Training'):
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['label'].to(device)
        
        optimizer.zero_grad()
        logits = model(input_ids, attention_mask)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()
        
        predictions = torch.argmax(logits, dim=1)
        correct += (predictions == labels).sum().item()
        total += labels.size(0)
        total_loss += loss.item()
    
    return total_loss / len(data_loader), correct / total


def validate(model, data_loader, criterion):
    model.eval()
    total_loss = 0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for batch in tqdm(data_loader, desc='Validation'):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['label'].to(device)
            
            logits = model(input_ids, attention_mask)
            loss = criterion(logits, labels)
            
            predictions = torch.argmax(logits, dim=1)
            correct += (predictions == labels).sum().item()
            total += labels.size(0)
            total_loss += loss.item()
    
    return total_loss / len(data_loader), correct / total


def main():
    # Load subset of dataset
    print("\nLoading IMDB dataset subset...")
    dataset = load_dataset('imdb')
    
    # Create small train/val/test splits
    train_data = dataset['train'].select(range(SUBSET_SIZE))
    val_data = dataset['train'].select(range(SUBSET_SIZE, SUBSET_SIZE + 500))
    test_data = dataset['test'].select(range(1000))
    
    print(f"Train: {len(train_data)}, Val: {len(val_data)}, Test: {len(test_data)}")
    
    # Initialize tokenizer
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    
    # Create datasets and loaders
    train_dataset = IMDBDataset(train_data, tokenizer, MAX_LENGTH)
    val_dataset = IMDBDataset(val_data, tokenizer, MAX_LENGTH)
    test_dataset = IMDBDataset(test_data, tokenizer, MAX_LENGTH)
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE)
    
    # Initialize model
    print("\nInitializing model...")
    model = SentimentTransformer().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5)
    
    # Training loop
    print(f"\nTraining for {NUM_EPOCHS} epochs...")
    best_val_acc = 0
    
    for epoch in range(NUM_EPOCHS):
        print(f"\nEpoch {epoch + 1}/{NUM_EPOCHS}")
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer)
        val_loss, val_acc = validate(model, val_loader, criterion)
        
        print(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}")
        print(f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")
        
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), 'quick_model.pt')
            print(f"✓ Best model saved!")
    
    # Test
    print("\nTesting...")
    model.load_state_dict(torch.load('quick_model.pt'))
    test_loss, test_acc = validate(model, test_loader, criterion)
    print(f"\nTest Accuracy: {test_acc:.4f} ({test_acc*100:.2f}%)")
    
    if test_acc > 0.75:
        print("✓ Exceeds 75% threshold!")
    
    # Demo predictions
    print("\n" + "="*60)
    print("SAMPLE PREDICTIONS")
    print("="*60)
    
    samples = [
        "This movie was absolutely amazing! Best film I've seen this year.",
        "Terrible movie. Complete waste of time and money.",
        "It was okay, nothing special but watchable."
    ]
    
    model.eval()
    for text in samples:
        encoding = tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            max_length=MAX_LENGTH,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt'
        )
        
        with torch.no_grad():
            input_ids = encoding['input_ids'].to(device)
            attention_mask = encoding['attention_mask'].to(device)
            logits = model(input_ids, attention_mask)
            probs = torch.softmax(logits, dim=1)
            pred = torch.argmax(probs, dim=1).item()
            conf = probs[0][pred].item()
        
        sentiment = 'Positive' if pred == 1 else 'Negative'
        print(f"\n\"{text}\"")
        print(f"→ {sentiment} (confidence: {conf:.2%})")
    
    print("\n" + "="*60)
    print("Quick demo completed!")
    print("For full training, run: python sentimentscope.py")
    print("="*60)


if __name__ == "__main__":
    main()
