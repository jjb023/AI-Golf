import requests
import json
import csv
from requests.api import head


api_key = "api"
api_url = "https://feeds.datagolf.com/historical-raw-data/rounds?tour=pga"

parameters = {
    "event_id": "5",
    "year": "2021",
    "file_format": "json",
    "key": api_key
}

response = requests.get(api_url, params=parameters)
myjson = response.json()
ourdata = []

for player in myjson['scores']:
    listing = [player['player_name'], player['fin_text']]
    ourdata.append(listing)
    
with open('Scores.csv', 'w', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(['Player', 'Finish'])
    writer.writerows(ourdata)