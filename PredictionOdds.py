# File to extract Bet365 odds from Datagolf.com API for the Arnold Palmer Invitational for 2024. 
import csv
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
    for model in api_data:
        for player_data in api_data[model]:
            if isinstance(player_data, dict):  # Check if player_data is a dictionary
                player_name = player_data.get('player_name')
                win = player_data.get('win')
                make_cut = player_data.get('make_cut')
                top_10 = player_data.get('top_10')
                top_15 = player_data.get('top_15')
                top_20 = player_data.get('top_20')
                top_30 = player_data.get('top_30')
                top_5 = player_data.get('top_5')

                
                extracted_data.append([player_name, win, top_5, top_10, top_15, top_20, top_30, make_cut])
            else:
                print("Unexpected data format:", player_data)
    
    return extracted_data

def main(tour, add_position, odds_format, file_format, key):
    api_url = f"https://feeds.datagolf.com/preds/pre-tournament?tour={tour}&add_position={add_position}&odds_format={odds_format}&file_format={file_format}&key={key}"
    api_data = fetch_data(api_url)
    if api_data:
        extracted_data = extract_data(api_data)
        with open('DGArnoldPalmerPrediction.csv', mode='w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(['Player Name', 'Win', 'Top 5', 'Top 10', 'Top 15', 'Top 20', 'Top 30', 'Make Cut'])
            for data_row in extracted_data:
                writer.writerow(data_row)
        print("CSV file created successfully!")

# Set variables for each part of the URL
tour = "pga"
add_position = "15,30"
odds_format = "decimal"
file_format = "json"
key = "d6b6280403a3d0f3b7917387aed7"  # Replace 'API_TOKEN' with your actual API token

# Call main function with the URL parts
main(tour, add_position, odds_format, file_format, key)
