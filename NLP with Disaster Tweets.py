import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import DistilBertTokenizer, DistilBertModel, get_linear_schedule_with_warmup
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score
import warnings
warnings.filterwarnings('ignore')

# Set random seeds for reproducibility
SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# ============================================================================
# 2. HYPERPARAMETERS
# ============================================================================

MAX_LEN = 160  # Max sequence length for tweets
BATCH_SIZE = 16
EPOCHS = 4
LEARNING_RATE = 2e-5
MODEL_NAME = 'distilbert-base-uncased'

# ============================================================================
# 3. LOAD DATA
# ============================================================================

# Load the data from Kaggle
# Make sure you have train.csv, test.csv in your working directory
train_df = pd.read_csv(r'D:\Users\Documents\Python\nlp-getting-started\train.csv')
test_df = pd.read_csv(r'D:\Users\Documents\Python\nlp-getting-started\test.csv')
sample_submission = pd.read_csv(r'D:\Users\Documents\Python\nlp-getting-started\sample_submission.csv')

print(f"Training set shape: {train_df.shape}")
print(f"Test set shape: {test_df.shape}")
print(f"\nSample tweets:")
print(train_df.head())

# ============================================================================
# 4. DATA PREPROCESSING
# ============================================================================

def preprocess_text(text):
    """Basic text preprocessing"""
    if pd.isna(text):
        return ""
    text = str(text)
    text = text.lower()
    return text

# Apply preprocessing
train_df['text'] = train_df['text'].apply(preprocess_text)
test_df['text'] = test_df['text'].apply(preprocess_text)

# Split train data into train and validation
train_texts, val_texts, train_labels, val_labels = train_test_split(
    train_df['text'].values,
    train_df['target'].values,
    test_size=0.1,
    random_state=SEED,
    stratify=train_df['target'].values
)

print(f"\nTraining samples: {len(train_texts)}")
print(f"Validation samples: {len(val_texts)}")
print(f"Test samples: {len(test_df)}")

# ============================================================================
# 5. TOKENIZER
# ============================================================================

tokenizer = DistilBertTokenizer.from_pretrained(MODEL_NAME)

# Test tokenization
sample = train_texts[0]
print(f"\nSample text: {sample}")
print(f"Tokenized: {tokenizer.tokenize(sample)}")
print(f"Token IDs: {tokenizer.encode(sample)}")

# ============================================================================
# 6. DATASET CLASS
# ============================================================================

class DisasterTweetDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_len):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_len = max_len
    
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        text = str(self.texts[idx])
        label = self.labels[idx] if self.labels is not None else 0
        
        # Tokenize and encode
        encoding = self.tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            max_length=self.max_len,
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

# Create datasets
train_dataset = DisasterTweetDataset(train_texts, train_labels, tokenizer, MAX_LEN)
val_dataset = DisasterTweetDataset(val_texts, val_labels, tokenizer, MAX_LEN)
test_dataset = DisasterTweetDataset(test_df['text'].values, None, tokenizer, MAX_LEN)

# Create dataloaders
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

# ============================================================================
# 7. MODEL DEFINITION
# ============================================================================

class DisasterTweetClassifier(nn.Module):
    def __init__(self, n_classes=2, dropout=0.3):
        super().__init__()
        self.bert = DistilBertModel.from_pretrained(MODEL_NAME)
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(self.bert.config.hidden_size, n_classes)
    
    def forward(self, input_ids, attention_mask):
        outputs = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask
        )
        
        # Use [CLS] token representation
        pooled_output = outputs.last_hidden_state[:, 0]
        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)
        
        return logits

# Initialize model
model = DisasterTweetClassifier(n_classes=2)
model = model.to(device)

print(f"\nModel architecture:")
print(model)

# ============================================================================
# 8. TRAINING SETUP
# ============================================================================

# Loss function
criterion = nn.CrossEntropyLoss()

# Optimizer
optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)

# Learning rate scheduler
total_steps = len(train_loader) * EPOCHS
scheduler = get_linear_schedule_with_warmup(
    optimizer,
    num_warmup_steps=0,
    num_training_steps=total_steps
)

# ============================================================================
# 9. TRAINING & EVALUATION FUNCTIONS
# ============================================================================

def train_epoch(model, data_loader, criterion, optimizer, device, scheduler):
    model.train()
    losses = []
    correct_predictions = 0
    
    for batch in data_loader:
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['label'].to(device)
        
        # Forward pass
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        loss = criterion(outputs, labels)
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        
        # Gradient clipping
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        optimizer.step()
        scheduler.step()
        
        # Track metrics
        _, preds = torch.max(outputs, dim=1)
        correct_predictions += torch.sum(preds == labels)
        losses.append(loss.item())
    
    return correct_predictions.double() / len(data_loader.dataset), np.mean(losses)

def eval_model(model, data_loader, criterion, device):
    model.eval()
    losses = []
    correct_predictions = 0
    predictions = []
    true_labels = []
    
    with torch.no_grad():
        for batch in data_loader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['label'].to(device)
            
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            loss = criterion(outputs, labels)
            
            _, preds = torch.max(outputs, dim=1)
            
            predictions.extend(preds.cpu().numpy())
            true_labels.extend(labels.cpu().numpy())
            correct_predictions += torch.sum(preds == labels)
            losses.append(loss.item())
    
    # Calculate F1 score
    f1 = f1_score(true_labels, predictions)
    
    return correct_predictions.double() / len(data_loader.dataset), np.mean(losses), f1

# ============================================================================
# 10. TRAINING LOOP
# ============================================================================

print("\n" + "="*50)
print("TRAINING STARTED")
print("="*50)

best_f1 = 0

for epoch in range(EPOCHS):
    print(f'\nEpoch {epoch + 1}/{EPOCHS}')
    print('-' * 30)
    
    # Train
    train_acc, train_loss = train_epoch(
        model, train_loader, criterion, optimizer, device, scheduler
    )
    
    print(f'Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f}')
    
    # Validate
    val_acc, val_loss, val_f1 = eval_model(
        model, val_loader, criterion, device
    )
    
    print(f'Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f} | Val F1: {val_f1:.4f}')
    
    # Save best model
    if val_f1 > best_f1:
        best_f1 = val_f1
        torch.save(model.state_dict(), r'D:\Users\Documents\Python\nlp-getting-started\best_model.pt')
        print(f'✓ Best model saved with F1: {best_f1:.4f}')

print("\n" + "="*50)
print("TRAINING COMPLETED")
print("="*50)

# ============================================================================
# 11. LOAD BEST MODEL & MAKE PREDICTIONS
# ============================================================================

# Load best model
model.load_state_dict(torch.load(r'D:\Users\Documents\Python\nlp-getting-started\best_model.pt'))
model.eval()

print("\n" + "="*50)
print("GENERATING PREDICTIONS")
print("="*50)

# Get predictions on test set
test_predictions = []

with torch.no_grad():
    for batch in test_loader:
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        _, preds = torch.max(outputs, dim=1)
        
        test_predictions.extend(preds.cpu().numpy())

# ============================================================================
# 12. CREATE SUBMISSION FILE
# ============================================================================

# Create submission dataframe
submission = pd.DataFrame({
    'id': test_df['id'],
    'target': test_predictions
})

# Save to CSV
submission.to_csv('submission.csv', index=False)

print(f"\n✓ Submission file created: submission.csv")
print(f"✓ Number of predictions: {len(submission)}")
print(f"\nPrediction distribution:")
print(submission['target'].value_counts())
print(f"\nFirst few predictions:")
print(submission.head(10))

print("\n" + "="*50)
print("DONE! Upload submission.csv to Kaggle")
print("="*50)