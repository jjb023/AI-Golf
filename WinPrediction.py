import pandas as pd
import numpy as np

file_path = "/Users/Home/Library/CloudStorage/OneDrive-UniversityofBristol/3rd Year/Intro to AI/player_mean_std_numApps1.xlsx"
df1 = pd.read_excel(file_path)

df2 = df1.drop(df1.columns[[0, 2, 4]], axis=1)

df2.insert(2, "Predicted Score", [69.352440, 71.452888, 71.693192, 71.439095, 70.712563, 71.105957, 70.987450, 71.53333, 70.703941, 71.938683])

Rank = pd.DataFrame()
Rank["Player Names"] = df2["Player Name"]

num_trials = 1500

for i in range(num_trials):
    df2["Normal Distribution"] = df2["STD"].apply(lambda x: np.random.normal(0, x))
    df2["Final Score"] = df2["Predicted Score"] + df2["Normal Distribution"]
    df2["Rank"] = df2["Final Score"].rank(ascending=True, method="min")
    Rank[f"Iteration_{i+1}"] = df2["Rank"]



Rank["Mean Pos."] = Rank.iloc[:, 1:].mean(axis=1)
Rank["Count of 1st"] = Rank.apply(lambda row: row.value_counts().get(1.0, 0), axis=1)

Rank["Win Percentage"] = (Rank["Count of 1st"] / num_trials) * 100
Rank["DataGolfWP"] = [11.2, 0.75, 2.3, 6.4, 5.1, 2.6, 3.3, 3.7, 2.1, 0.4]

Final = Rank[["Player Names", "Win Percentage", "DataGolfWP"]]
Final = Final.sort_values(by="Win Percentage", ascending=False)

data_golf_sum = Final["DataGolfWP"].sum()

Final["DataGolf Adjusted"] = Final["DataGolfWP"] * 100 / data_golf_sum

print(Final)

