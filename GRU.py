#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar  1 17:12:39 2024

@author: archie
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from sklearn.preprocessing import StandardScaler
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score
from sklearn.metrics import mean_squared_error
import random

random.seed(1)
# Load data
df = pd.read_csv('/Users/archie/Downloads/KeyData_sorted.csv')

# Sort the dataframe first by Player Name and then by Date Completed in ascending order to maintain the time series element
df_sorted = df.sort_values(by=['Player Name', 'Date Completed'], ascending=[True, True])

# Re-extract Rory McIlroy's data to ensure 'Course Name' column is present
rory_mcilroy_data_correct = df_sorted[df_sorted['Player Name'] == 'McIlroy, Rory']

# Apply one-hot encoding directly to the 'Course Name' column
rory_mcilroy_data_one_hot_names = pd.get_dummies(rory_mcilroy_data_correct, columns=['Course Name'])

# Define non-feature columns
non_feature_columns = ['Player Name', 'Year Played', 'Date Completed', 'Course Par', 'Round Score']

# Dynamically create the list of feature columns by excluding non-feature columns
feature_columns = [col for col in rory_mcilroy_data_one_hot_names.columns if col not in non_feature_columns]

# Define the target label column
target_column = 'Round Score'

# Split the data into features (X) and target label (y)
X = rory_mcilroy_data_one_hot_names[feature_columns]
y = rory_mcilroy_data_one_hot_names[target_column]

X.columns = [col.replace('Course Name_', '') for col in X.columns]


def create_sequences(X, y, sequence_length=5):
    X_seq, y_seq = [], []
    for i in range(len(X) - sequence_length + 1):
        X_seq.append(X.iloc[i:i+sequence_length].values)  # Select consecutive rounds
        y_seq.append(y.iloc[i+sequence_length-1])  # Target is the score of the last round in the sequence
    return np.array(X_seq), np.array(y_seq)

# Assuming `X` is a DataFrame with your features and `y` is a Series with your scores
sequence_length = 5  # Define how many rounds you want to consider in each sequence
X_seq, y_seq = create_sequences(X, y, sequence_length)

# Scaling the features (not the target in this case)
scaler = StandardScaler()
X_seq_scaled = np.array([scaler.fit_transform(x) for x in X_seq])

# Splitting into training+validation and testing sets first
X_train_val, X_test, y_train_val, y_test = train_test_split(X_seq_scaled, y_seq, test_size=0.2, random_state=42)

# Then splitting the training+validation set into separate training and validation sets
X_train, X_val, y_train, y_val = train_test_split(X_train_val, y_train_val, test_size=0.25, random_state=42)  # 0.25 x 0.8 = 0.2 of the original data

# Convert to PyTorch tensors
X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train, dtype=torch.float32).view(-1, 1)
X_val_tensor = torch.tensor(X_val, dtype=torch.float32)
y_val_tensor = torch.tensor(y_val, dtype=torch.float32).view(-1, 1)
X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
y_test_tensor = torch.tensor(y_test, dtype=torch.float32).view(-1, 1)

# Define the GRU model as before
class GRUNet(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim=1, num_layers=2):
        super(GRUNet, self).__init__()
        self.gru = nn.GRU(input_dim, hidden_dim, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)
    
    def forward(self, x):
        x, _ = self.gru(x)
        x = self.fc(x[:, -1, :])  # Output from the last sequence step
        return x

model = GRUNet(input_dim=X_train_tensor.shape[2], hidden_dim=64)

criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.1)

# Updated training loop with validation loss tracking
epochs = 250
train_losses = []
val_losses = []

for epoch in range(epochs):
    model.train()
    optimizer.zero_grad()
    outputs = model(X_train_tensor)
    loss = criterion(outputs, y_train_tensor)
    loss.backward()
    optimizer.step()
    
    train_losses.append(loss.item())
    
    # Validation phase
    model.eval()
    with torch.no_grad():
        val_outputs = model(X_val_tensor)
        val_loss = criterion(val_outputs, y_val_tensor)
        val_losses.append(val_loss.item())
    
    if epoch % 10 == 0:
        print(f'Epoch [{epoch+1}/{epochs}], Train Loss: {loss.item():.4f}, Val Loss: {val_loss.item():.4f}')

# Plot training and validation loss curves
plt.figure(figsize=(10, 5))
plt.plot(train_losses, label='Training Loss')
plt.plot(val_losses, label='Validation Loss')
plt.title('Training and Validation Losses')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()

model.eval()
with torch.no_grad():
    predictions = model(X_test_tensor).view(-1).numpy()
    # No need for inverse scaling if y wasn't scaled

# Calculate performance metric, e.g., RMSE
test_rmse = np.sqrt(mean_squared_error(y_test, predictions))
print(f'Test RMSE: {test_rmse}')

plt.figure(figsize=(10, 6))
plt.plot(y_test, label='Actual Scores', marker='o')
plt.plot(predictions, label='Predicted Scores', marker='x')
plt.xlabel('Test Sample')
plt.ylabel('Golf Score')
plt.title('Actual vs. Predicted Golf Scores')
plt.legend()
plt.savefig('GRU_predictions.png')
plt.show()

