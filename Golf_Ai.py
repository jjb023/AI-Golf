import pandas as pd
import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

scaler = StandardScaler()
df = pd.read_csv('/Users/archie/Downloads/KeyData_sorted.csv')

# Split the data into training set and test set
train_df, test_df = train_test_split(df, test_size=0.2)

columns_to_keep = ['Course Name', 
       'Round Score', 'Round SG Total', 'Round SG App',
       'Round SG Arg', 'Round SG Putt', 'Round Driving Distance',
       'Round Driving Accuracy']

# List of columns to drop
columns_to_drop = [col for col in df.columns if col not in columns_to_keep]

# Drop the columns for 
train_data = train_df.drop(columns_to_drop, axis=1)
test_data = test_df.drop(columns_to_drop, axis=1)

# For the training data cleaning
train_data.fillna(train_data.mean(), inplace=True)
test_data.fillna(test_data.mean(), inplace=True)

# Perform one-hot encoding on the 'Course Name' column for both training and test datasets
train_data_encoded = pd.get_dummies(train_data, columns=['Course Name'])
test_data_encoded = pd.get_dummies(test_data, columns=['Course Name'])

# Adjusting column names for train_data_encoded
train_data_encoded.columns = [col.replace('Course Name_', '') for col in train_data_encoded.columns]

# Adjusting column names for test_data_encoded
test_data_encoded.columns = [col.replace('Course Name_', '') for col in test_data_encoded.columns]


# This step aligns the test data columns with the train data columns, adding missing columns with default value 0
test_data_encoded = test_data_encoded.reindex(columns=train_data_encoded.columns, fill_value=0)

# Further split train_data_encoded into training and validation sets
train_data_encoded, val_data_encoded = train_test_split(train_data_encoded, test_size=0.2, random_state=42)

# Select the first column as train_Y and the rest as train_X
train_Y = train_data_encoded.iloc[:, 0]
train_X = train_data_encoded.iloc[:, 1:]

# Selecting the target and features for the validation set
val_Y = val_data_encoded.iloc[:, 0]
val_X = val_data_encoded.iloc[:, 1:]

# Do the same for test_data
test_Y = test_data_encoded.iloc[:, 0]
test_X = test_data_encoded.iloc[:, 1:]

train_data_encoded.to_csv('train_data.csv', index=False)
test_data_encoded.to_csv('test_data.csv', index=False)


# Fit the scaler to the training data only
scaler.fit(train_X)

# Now transform the training, validation, and test data with the fitted scaler
train_X_scaled = scaler.transform(train_X)
val_X_scaled = scaler.transform(val_X)
test_X_scaled = scaler.transform(test_X)

# Assuming train_X, train_Y, test_X, test_Y are pandas DataFrames or Series
train_X_tensor = torch.tensor(train_X_scaled, dtype=torch.float32)
train_Y_tensor = torch.tensor(train_Y.values, dtype=torch.float32)  #Use torch.long if this is for classification

test_X_tensor = torch.tensor(test_X_scaled, dtype=torch.float32)
test_Y_tensor = torch.tensor(test_Y.values, dtype=torch.float32)  #Use torch.long if this is for classification

# Converting validation set to tensors
val_X_tensor = torch.tensor(val_X_scaled, dtype=torch.float32)
val_Y_tensor = torch.tensor(val_Y.values, dtype=torch.float32).view(-1, 1)

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
optimizer = torch.optim.Adam(net.parameters(), lr=0.1)


# Training loop with validation loss tracking
epochs = 100
train_losses = []
val_losses = []

for epoch in range(epochs):
    net.train()
    optimizer.zero_grad()
    outputs = net(train_X_tensor)
    loss = criterion(outputs, train_Y_tensor.view(-1, 1))
    loss.backward()
    optimizer.step()
    train_losses.append(loss.item())

    # Validation
    net.eval()
    val_outputs = net(val_X_tensor)
    val_loss = criterion(val_outputs, val_Y_tensor)
    val_losses.append(val_loss.item())

    if epoch % 10 == 0:
        print(f'Epoch [{epoch+1}/{epochs}], Train Loss: {loss.item():.4f}, Val Loss: {val_loss.item():.4f}')

# Plotting training and validation loss
plt.figure(figsize=(10, 5))
plt.plot(train_losses, label='Training Loss')
plt.plot(val_losses, label='Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training and Validation Loss')
plt.legend()
plt.show()


with torch.no_grad():
    predicted_means = net(test_X_tensor).cpu().numpy().flatten()
# Print a comparison of predicted vs. actual values for a subset of the test set
print("\nPredicted vs Actual Scores:")
for predicted, actual in zip(predicted_means[:10], test_Y.to_numpy()[:50]):  # Displaying first 10 for brevity
    print(f'Predicted: {predicted:.2f}, Actual: {actual}')
    
# Plotting training and validation loss
plt.figure(figsize=(10, 5))
plt.plot(predicted_means[:50], label='predicted', marker='x')
plt.plot(test_Y.to_numpy()[:50], label='actual', marker='o')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training and Validation Loss')
plt.legend()
plt.savefig('RegressionNet.png')
plt.show()