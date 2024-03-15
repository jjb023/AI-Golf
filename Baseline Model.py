import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error
import numpy as np
import time
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv('KeyData_sorted.csv')
test_df = pd.read_csv('10_test_players.csv')
Results = pd.DataFrame()
Results['Player Name'] = test_df.iloc[:, 0]

df = df.drop(df.columns[[0, 1, 2, 3]], axis=1)
test_df = test_df.drop(test_df.columns[[0, 1, 2, 3]], axis=1)

X_train = df.drop("Round Score", axis=1)
Y_train = df["Round Score"]
X_test = test_df.drop("Round Score", axis=1)
Y_test = test_df["Round Score"]

# Train the linear regression model
regression_model = LinearRegression()
start_LR = time.time()
regression_model.fit(X_train, Y_train)
end_LR = time.time()
LR_time = end_LR - start_LR

print("Linear Regression Training Time:", LR_time)

# Train the decision tree model
decision_tree_model = DecisionTreeRegressor()
start_DT = time.time()
decision_tree_model.fit(X_train, Y_train)
end_DT = time.time()
DT_time = end_DT - start_DT

print("Decision Tree Training Time:", DT_time)

# Predict the target variable for X_test
LR_Y_pred = regression_model.predict(X_test)
DT_Y_pred = decision_tree_model.predict(X_test)

# Calculate the error and add to result dataframe
Results['Y_Test'] = Y_test
Results['LR_Y_Pred'] = LR_Y_pred
Results['DT_Y_Pred'] = DT_Y_pred
Results['LR_Error'] = Results['LR_Y_Pred'] - Results['Y_Test']
Results['DT_Error'] = Results['DT_Y_Pred'] - Results['Y_Test']
Results['RNN_Error'] = [1.102440, 1.952888, 0.693192, -0.310905, -1.287437, -0.894043, -1.262550, -0.966667, -1.546059, -0.561317]
Results['LSTM_Error'] = [0.973,-2.745,-0.537,-2.054,-2.552,-2.733,-2.179,-3.286,-1.853,-3.558]
Results['GRU_Error'] = [-0.117,0.945,0.095,-1.628,0.02,0.700,-4.144,1.12,-3.04,-2.815]

# Calculate the mean squared error for each predictor
LR_RMSE = np.sqrt(mean_squared_error(Y_test, LR_Y_pred))
DT_MSE = np.sqrt(mean_squared_error(Y_test, DT_Y_pred))

print("Linear Regression RMSE:", LR_RMSE)
print("Decision Tree RMSE:", DT_MSE)
print(Results)


fig, axs = plt.subplots(2, 2, figsize=(8, 8))  # 2 rows, 3 columns


# Violin plot for LR_Error
sns.violinplot(data=Results, y='LR_Error', color='seagreen', label='LR Error', ax=axs[0, 0])
mean_error = np.mean(Results['LR_Error'])
quartiles = np.percentile(Results['LR_Error'], [25, 75])
min_error = np.min(Results['LR_Error'])
max_error = np.max(Results['LR_Error'])
axs[0, 0].text(0.05, 0.95, f"Mean: {mean_error:.2f}", transform=axs[0, 0].transAxes, ha='left', va='top')
axs[0, 0].text(0.05, 0.85, f"IQR: {quartiles[0]:.2f} - {quartiles[1]:.2f}", transform=axs[0, 0].transAxes, ha='left', va='top')
axs[0, 0].text(0.05, 0.75, f"Min: {min_error:.2f}", transform=axs[0, 0].transAxes, ha='left', va='top')
axs[0, 0].text(0.05, 0.65, f"Max: {max_error:.2f}", transform=axs[0, 0].transAxes, ha='left', va='top')
axs[0, 0].set_title("Linear Regression")
axs[0, 0].set_xlabel("")
axs[0, 0].set_ylabel("Error")
axs[0, 0].set_ylim([-7, 7])

# Violin plot for RNN_Error
sns.violinplot(data=Results, y='RNN_Error', color='seagreen', label='RNN Error', ax=axs[0, 1])
mean_error = np.mean(Results['RNN_Error'])
quartiles = np.percentile(Results['RNN_Error'], [25, 75])
min_error = np.min(Results['RNN_Error'])
max_error = np.max(Results['RNN_Error'])
axs[0, 1].text(0.05, 0.95, f"Mean: {mean_error:.2f}", transform=axs[0, 1].transAxes, ha='left', va='top')
axs[0, 1].text(0.05, 0.85, f"IQR: {quartiles[0]:.2f} - {quartiles[1]:.2f}", transform=axs[0, 1].transAxes, ha='left', va='top')
axs[0, 1].text(0.05, 0.75, f"Min: {min_error:.2f}", transform=axs[0, 1].transAxes, ha='left', va='top')
axs[0, 1].text(0.05, 0.65, f"Max: {max_error:.2f}", transform=axs[0, 1].transAxes, ha='left', va='top')
axs[0, 1].set_title("Feed Foward NN")
axs[0, 1].set_xlabel("")
axs[0, 1].set_ylabel("Error")
axs[0, 1].set_ylim([-7, 7])

# Violin plot for LSTM_Error
sns.violinplot(data=Results, y='LSTM_Error', color='seagreen', label='LSTM Error', ax=axs[1, 0])
mean_error = np.mean(Results['LSTM_Error'])
quartiles = np.percentile(Results['LSTM_Error'], [25, 75])
min_error = np.min(Results['LSTM_Error'])
max_error = np.max(Results['LSTM_Error'])
axs[1, 0].text(0.05, 0.95, f"Mean: {mean_error:.2f}", transform=axs[1, 0].transAxes, ha='left', va='top')
axs[1, 0].text(0.05, 0.85, f"IQR: {quartiles[0]:.2f} - {quartiles[1]:.2f}", transform=axs[1, 0].transAxes, ha='left', va='top')
axs[1, 0].text(0.05, 0.75, f"Min: {min_error:.2f}", transform=axs[1, 0].transAxes, ha='left', va='top')
axs[1, 0].text(0.05, 0.65, f"Max: {max_error:.2f}", transform=axs[1, 0].transAxes, ha='left', va='top')
axs[1, 0].set_title("LSTM")
axs[1, 0].set_xlabel("")
axs[1, 0].set_ylabel("Error")
axs[1, 0].set_ylim([-7, 7])

# Violin plot for GRU_Error
sns.violinplot(data=Results, y='GRU_Error', color='seagreen', label='GRU Error', ax=axs[1, 1])
mean_error = np.mean(Results['GRU_Error'])
quartiles = np.percentile(Results['GRU_Error'], [25, 75])
min_error = np.min(Results['GRU_Error'])
max_error = np.max(Results['GRU_Error'])
axs[1, 1].text(0.05, 0.95, f"Mean: {mean_error:.2f}", transform=axs[1, 1].transAxes, ha='left', va='top')
axs[1, 1].text(0.05, 0.85, f"IQR: {quartiles[0]:.2f} - {quartiles[1]:.2f}", transform=axs[1, 1].transAxes, ha='left', va='top')
axs[1, 1].text(0.05, 0.75, f"Min: {min_error:.2f}", transform=axs[1, 1].transAxes, ha='left', va='top')
axs[1, 1].text(0.05, 0.65, f"Max: {max_error:.2f}", transform=axs[1, 1].transAxes, ha='left', va='top')
axs[1, 1].set_title("GRU")
axs[1, 1].set_xlabel("")
axs[1, 1].set_ylabel("Error")
axs[1, 1].set_ylim([-7, 7])

# Adjust the spacing between subplots
plt.tight_layout()
import psutil

def print_memory_usage():
    process = psutil.Process()
    memory_info = process.memory_info()
    print(f"Memory Usage: {memory_info.rss / (1024 * 1024):.2f} MB")

print_memory_usage()
# Display the plot
plt.show()


