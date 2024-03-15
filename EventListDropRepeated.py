'''
Code to read event list and drop repeated events to get a list of unique event names and their event code

'''


import pandas as pd

df = pd.read_csv('EventList.csv')

df_dropped = df.drop_duplicates(subset='Event Name')

df_dropped.to_csv('EventList2.csv', index=False)

unique_entries = df['Event Name'].nunique()
print("Number of unique entries in 'Event Name' column:", unique_entries)
unique_event_ids = df['Event ID'].nunique()
print("Number of unique entries in 'Event ID' column:", unique_event_ids)

print("Size of df:", df.shape)
print("Size of df_dropped:", df_dropped.shape)
