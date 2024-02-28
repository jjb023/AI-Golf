import pandas as pd

# Read the CSV file into a DataFrame
df = pd.read_csv('KeyDataCleaned/KeyDataCleaned.csv')

# Parse 'Date Completed' column as datetime objects
df['Date Completed'] = pd.to_datetime(df['Date Completed'])

# Sort DataFrame based on 'Date Completed' column
df_sorted = df.sort_values(by='Date Completed')

# Rewrite the sorted DataFrame back to the CSV file
df_sorted.to_csv('KeyDataCleaned/KeyData_sorted.csv', index=False)

print("CSV file sorted and created successfully!")
