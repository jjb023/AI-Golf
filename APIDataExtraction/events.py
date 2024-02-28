import requests
import json
import csv

url = 'https://feeds.datagolf.com/historical-raw-data/event-list?file_format=[ file_format ]&key=d6b6280403a3d0f3b7917387aed7'

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