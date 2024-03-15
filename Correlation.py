import pandas as pd

df = pd.read_csv("KeyDataCleaned.csv")

# Select the columns of interest
columns_of_interest = df.columns[5:12]

# Calculate Pearson correlation coefficient
pearson_corr = df[columns_of_interest].corrwith(df.iloc[:, 5], method='pearson')

# Calculate Spearman correlation coefficient
spearman_corr = df[columns_of_interest].corrwith(df.iloc[:, 5], method='spearman')

print("Pearson correlation coefficients:")
print(pearson_corr)

print("Spearman correlation coefficients:")
print(spearman_corr)

