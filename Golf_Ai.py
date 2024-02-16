import pandas as pd
import torch
import torch.nn as nn
import numpy as np
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import seaborn as sns
from sklearn.metrics import accuracy_score


#Output disribution not a number  

# Set the random seed
#np.random.seed(0)

df = pd.read_excel('Masters_2021.xlsx', skiprows=1)

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

# Assuming train_X, train_Y, test_X, test_Y are pandas DataFrames or Series
train_X_tensor = torch.tensor(train_X.values, dtype=torch.float32)
train_Y_tensor = torch.tensor(train_Y.values, dtype=torch.float32)  # Use torch.long if this is for classification

test_X_tensor = torch.tensor(test_X.values, dtype=torch.float32)
test_Y_tensor = torch.tensor(test_Y.values, dtype=torch.float32)  # Use torch.long if this is for classification


class RegressionNet(nn.Module):
    def __init__(self, input_size):
        super(RegressionNet, self).__init__()
        self.fc1 = nn.Linear(input_size, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 1)  # Single output for regression

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)  # Direct output for regression
        return x

net = RegressionNet(input_size=train_X.shape[1])
criterion = nn.MSELoss()  # Mean Squared Error Loss
optimizer = torch.optim.Adam(net.parameters(), lr=0.001)

epochs = 100
for epoch in range(epochs):
    optimizer.zero_grad()
    outputs = net(train_X_tensor)
    loss = criterion(outputs, train_Y_tensor.view(-1, 1))  # Ensure correct shape
    loss.backward()
    optimizer.step()

    if epoch % 10 == 0:
        print(f'Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.4f}')


with torch.no_grad():  # No need to track gradients here
    predicted = net(test_X_tensor)
    predicted_rounded = torch.round(predicted).int()  # Round predictions to nearest integer


# Converting the Pandas Series to a numpy array
test_Y_np = test_Y.to_numpy()
# Assuming `predicted_rounded` is your tensor of predictions
predicted_rounded_np = predicted_rounded.cpu().numpy()

# Printing predicted and actual values side by side
for predicted, actual in zip(predicted_rounded_np.flatten(), test_Y_np):
    print(f'Predicted: {predicted}, Actual: {actual}')

# Assuming `test_Y_tensor` is your true labels and `predicted_rounded` is your predictions rounded to the nearest integer
accuracy = accuracy_score(test_Y_tensor.cpu().numpy(), predicted_rounded.cpu().numpy())
print(f'Accuracy: {accuracy * 100:.2f}%')