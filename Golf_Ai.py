import pandas as pd
import torch
import torch.nn as nn
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
df = pd.read_excel('/Users/archie/Downloads/Masters_2021.xlsx', skiprows=1)

# Split the data into training set and test set
train_df, test_df = train_test_split(df, test_size=0.2)

columns_to_keep = ['sg_putt', 
       'sg_arg', 'sg_app', 'sg_ott', 'sg_t2g', 'sg_total', 'driving_dist',
       'driving_acc', 'gir', 'scrambling', 'prox_rgh', 'prox_fw',
       'great_shots', 'poor_shots','round_score']

# List of columns to drop
columns_to_drop = [col for col in df.columns if col not in columns_to_keep]

# Drop the columns for 
train_data = train_df.drop(columns_to_drop, axis=1)
test_data = test_df.drop(columns_to_drop, axis=1)

# For the training data
train_data.fillna(train_data.mean(), inplace=True)

# For the testing data
test_data.fillna(test_data.mean(), inplace=True)

# Select the first column as train_Y and the rest as train_X
train_Y = train_data.iloc[:, 0]
train_X = train_data.iloc[:, 1:]

# Do the same for test_data
test_Y = test_data.iloc[:, 0]
test_X = test_data.iloc[:, 1:]

train_data.to_csv('train_data.csv', index=False)
test_data.to_csv('test_data.csv', index=False)

# Fit the scaler to the training and test data and transform it
train_X_scaled = scaler.fit_transform(train_X)
test_X_scaled = scaler.transform(test_X)
train_Y_scaled = scaler.fit_transform(train_Y.values.reshape(-1, 1))

# Assuming train_X, train_Y, test_X, test_Y are pandas DataFrames or Series
train_X_tensor = torch.tensor(train_X_scaled, dtype=torch.float32)
train_Y_tensor = torch.tensor(train_Y_scaled, dtype=torch.float32)  # Use torch.long if this is for classification

test_X_tensor = torch.tensor(test_X_scaled, dtype=torch.float32)
test_Y_tensor = torch.tensor(test_Y.values, dtype=torch.float32)  # Use torch.long if this is for classification


class SimpleRegressionNet(nn.Module):
    def __init__(self, input_size):
        super(SimpleRegressionNet, self).__init__()
        self.fc1 = nn.Linear(input_size, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 1)  # Output a single value

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)  # Direct output for regression, predicting the mean
        return x

    
# Assuming train_X.shape[1] gives the number of features
net = SimpleRegressionNet(input_size=train_X.shape[1])

criterion = nn.MSELoss()  # Mean Squared Error Loss
optimizer = torch.optim.Adam(net.parameters(), lr=0.001)


epochs = 100
for epoch in range(epochs):
    optimizer.zero_grad()
    outputs = net(train_X_tensor)
    loss = criterion(outputs, train_Y_tensor.view(-1, 1))  # Ensure correct shape for target
    loss.backward()
    optimizer.step()

    if epoch % 10 == 0:
        print(f'Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.4f}')



with torch.no_grad():
    predicted_means = net(test_X_tensor).cpu().numpy().flatten()

predicted_means = scaler.inverse_transform(predicted_means.reshape(-1, 1)).flatten()

# Compare predicted and actual values
for predicted, actual in zip(predicted_means, test_Y.to_numpy()):
    print(f'Predicted: {predicted}, Actual: {actual}')