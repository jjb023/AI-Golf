import requests
import csv

def fetch_data(api_url):
    response = requests.get(api_url)
    if response.status_code == 200:
        return response.json()
    else:
        print("Error fetching data from API.")
        return None

def extract_data(api_data):
    if api_data is None:
        return []

    extracted_data = []
    for player_data in api_data['players']:
        player_name = player_data['player_name']
        sg_putt = player_data['sg_putt']
        sg_arg = player_data['sg_arg']
        sg_app = player_data['sg_app']
        sg_ott = player_data['sg_ott']
        sg_total = player_data['sg_total']
        driving_acc = player_data['driving_acc']
        driving_dist = player_data['driving_dist']
        
        extracted_data.append([player_name, sg_putt, sg_arg, sg_app, sg_ott, sg_total, driving_acc, driving_dist])
    
    return extracted_data

def main(file_format, key):
    api_url = f"https://feeds.datagolf.com/preds/skill-ratings?display=value&file_format={file_format}&key={key}"
    api_data = fetch_data(api_url)
    if api_data:
        extracted_data = extract_data(api_data)
        with open('PlayersStats/playerstats.csv', mode='w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(['Player Name', 'SG Putt', 'SG Arg', 'SG App', 'SG Ott', 'SG Total', 'Driving Acc', 'Driving Dist'])
            for data_row in extracted_data:
                writer.writerow(data_row)
        print("CSV file created successfully!")

# Set variables for file format and API token
file_format = "json"
key = "d6b6280403a3d0f3b7917387aed7"  # Replace 'API_TOKEN' with your actual API token

# Call main function
main(file_format, key)
