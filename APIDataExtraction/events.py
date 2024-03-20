import requests
import json
import csv

url = 'https://feeds.datagolf.com/historical-raw-data/event-list?file_format=[ file_format ]&key=api'

r = requests.get(url)
data = json.loads(r.text)

eventlist = []

for event in data:
    eventid = event['event_id']
    eventname = event['event_name']

    event_item = {
        eventname,
        eventid,
    }   
    eventlist.append(event_item)




with open('EventList.csv', 'w', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(['Event Name', 'Event ID'])
    writer.writerows(eventlist)

print(eventlist)