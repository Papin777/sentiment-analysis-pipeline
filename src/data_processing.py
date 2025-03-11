#2. Data Processing (Student 1 & Student 2)

import pandas as pd
import numpy as np
import re
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from transformers import BertTokenizer
from torch.utils.data import Dataset, DataLoader
import torch

# Set random seed for reproducibility
RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)

# Set the model name
MODEL_NAME = 'bert-base-cased'

# Load the tokenizer
tokenizer = BertTokenizer.from_pretrained(MODEL_NAME)

# Print some common BERT tokens
print(f"Sep Token: {tokenizer.sep_token}, ID: {tokenizer.sep_token_id}")
print(f"Cls Token: {tokenizer.cls_token}, ID: {tokenizer.cls_token_id}")
print(f"Pad Token: {tokenizer.pad_token}, ID: {tokenizer.pad_token_id}")
print(f"Unk Token: {tokenizer.unk_token}, ID: {tokenizer.unk_token_id}")

# Text cleaning and preprocessing
def clean_text(text):
    """
    Cleans and preprocesses the input text.
    - Removes unnecessary characters.
    - Converts text to lowercase.
    - Normalizes whitespace.
    """
    # Remove special characters and numbers
    text = re.sub(r"[^a-zA-Z\s]", "", text)
    # Convert to lowercase
    text = text.lower()
    # Normalize whitespace
    text = re.sub(r"\s+", " ", text).strip()
    return text
     
# Custom Dataset class for reviews
class GPReviewDataset(Dataset):
    def __init__(self, reviews, targets, tokenizer, max_len):
        self.reviews = reviews
        self.targets = targets
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.reviews)
  
    def __getitem__(self, item):
        review = str(self.reviews[item])
        target = self.targets[item]

        # Tokenize the review
        encoding = self.tokenizer.encode_plus(
            review,
            add_special_tokens=True,
            max_length=self.max_len,
            return_token_type_ids=False,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt',
        )

        return {
            'review_text': review,
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'targets': torch.tensor(target, dtype=torch.long)
        }

# Function to create a DataLoader
def create_data_loader(df, tokenizer, max_len, batch_size):
    """
    Creates a DataLoader for the given DataFrame.
    """
    ds = GPReviewDataset(
        reviews=df.content.to_numpy(),
        targets=df.sentiment.to_numpy(),
        tokenizer=tokenizer,
        max_len=max_len
    )
    return DataLoader(
        ds,
        batch_size=batch_size,
        num_workers=0
    )

# Main function for data processing
def process_data(df):
    df['content'] = df['content'].apply(clean_text)

    # Vérifier la taille avant de diviser
    if len(df) < 5:
        raise ValueError("Le dataset est trop petit pour être divisé en train/val/test.")

    df_train, df_test = train_test_split(df, test_size=0.2, random_state=RANDOM_SEED)

    # Vérifier si df_test est assez grand pour être divisé
    if len(df_test) > 1:
        df_val, df_test = train_test_split(df_test, test_size=0.5, random_state=RANDOM_SEED)
    else:
        df_val, df_test = df_test, df_test  # Si pas assez d'échantillons, on garde tout en test

    print(f"Training set size: {df_train.shape}")
    print(f"Validation set size: {df_val.shape}")
    print(f"Test set size: {df_test.shape}")

    max_len = 128  
    batch_size = 16

    train_data_loader = create_data_loader(df_train, tokenizer, max_len, batch_size)
    val_data_loader = create_data_loader(df_val, tokenizer, max_len, batch_size)
    test_data_loader = create_data_loader(df_test, tokenizer, max_len, batch_size)
    
    return train_data_loader, val_data_loader, test_data_loader
