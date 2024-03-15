'''
File to calculate the std for each player and store it in a new file
'''

import pandas as pd
import numpy as np

df = pd.read_excel('SortedScore.xlsx')

names = df['Player Name'].unique().tolist()

player_mean_std = pd.DataFrame(columns=['Player Name', 'Mean', 'STD', 'Num App'])

for player in names:
    player_rows = df[df['Player Name'] == player]
    round_par_mean = player_rows['Round Par'].mean()
    round_par_std = player_rows['Round Par'].std()
    num_app = len(player_rows)
    
    player_mean_std = player_mean_std.append({'Player Name': player, 'Mean': round_par_mean, 'STD': round_par_std, 'Num App': num_app}, ignore_index=True)
    

# Export player_mean_std as an Excel spreadsheet
player_mean_std.to_excel('player_mean_std.xlsx', index=True)


