import csv
import requests

# Function to fetch data from the API
def fetch_data(event_id, year, api_token):
    url = f"https://feeds.datagolf.com/historical-raw-data/rounds?tour=pga&event_id={event_id}&year={year}&file_format=json&key={api_token}"
    response = requests.get(url)
    if response.status_code == 200:
        return response.json()
    else:
        print("Error fetching data from API.")
        return None

# Function to extract desired data from API response
# Function to extract desired data from API response
def extract_data(api_data):
    extracted_data = []
    for score in api_data['scores']:
        player_name = score['player_name']
        course_name = score['round_1']['course_name']
        year_played = api_data['year']
        date_completed = api_data['event_completed']
        
        scores = []
        sg_total = []
        sg_app = []
        sg_arg = []
        sg_putt = []
        driving_distances = []
        driving_accuracies = []
        
        if 'round_1' in score:
            scores.append(score['round_1'].get('score'))
            sg_total.append(score['round_1'].get('sg_total'))
            sg_app.append(score['round_1'].get('sg_app'))
            sg_arg.append(score['round_1'].get('sg_arg'))
            sg_putt.append(score['round_1'].get('sg_putt'))
            driving_distances.append(score['round_1'].get('driving_dist'))
            driving_accuracies.append(score['round_1'].get('driving_acc'))
        else:
            scores.append(None)
            sg_total.append(None)
            sg_app.append(None)
            sg_arg.append(None)
            sg_putt.append(None)
            driving_distances.append(None)
            driving_accuracies.append(None)
        
        if 'round_2' in score:
            scores.append(score['round_2'].get('score'))
            sg_total.append(score['round_2'].get('sg_total'))
            sg_app.append(score['round_2'].get('sg_app'))
            sg_arg.append(score['round_2'].get('sg_arg'))
            sg_putt.append(score['round_2'].get('sg_putt'))
            driving_distances.append(score['round_2'].get('driving_dist'))
            driving_accuracies.append(score['round_2'].get('driving_acc'))
        else:
            scores.append(None)
            sg_total.append(None)
            sg_app.append(None)
            sg_arg.append(None)
            sg_putt.append(None)
            driving_distances.append(None)
            driving_accuracies.append(None)
        
        if 'round_3' in score:
            scores.append(score['round_3'].get('score'))
            sg_total.append(score['round_3'].get('sg_total'))
            sg_app.append(score['round_3'].get('sg_app'))
            sg_arg.append(score['round_3'].get('sg_arg'))
            sg_putt.append(score['round_3'].get('sg_putt'))
            driving_distances.append(score['round_3'].get('driving_dist'))
            driving_accuracies.append(score['round_3'].get('driving_acc'))
        else:
            scores.append(None)
            sg_total.append(None)
            sg_app.append(None)
            sg_arg.append(None)
            sg_putt.append(None)
            driving_distances.append(None)
            driving_accuracies.append(None)
        
        if 'round_4' in score:
            scores.append(score['round_4'].get('score'))
            sg_total.append(score['round_4'].get('sg_total'))
            sg_app.append(score['round_4'].get('sg_app'))
            sg_arg.append(score['round_4'].get('sg_arg'))
            sg_putt.append(score['round_4'].get('sg_putt'))
            driving_distances.append(score['round_4'].get('driving_dist'))
            driving_accuracies.append(score['round_4'].get('driving_acc'))
        else:
            scores.append(None)
            sg_total.append(None)
            sg_app.append(None)
            sg_arg.append(None)
            sg_putt.append(None)
            driving_distances.append(None)
            driving_accuracies.append(None)
        
        extracted_data.append([player_name, course_name, year_played, date_completed] + scores + sg_total + sg_app + sg_arg + sg_putt +
                              driving_distances + driving_accuracies)
    return extracted_data



# Main function to fetch data, extract, and write to CSV
def main(event_id, year, api_token):
    api_data = fetch_data(event_id, year, api_token)
    if api_data:
        extracted_data = extract_data(api_data)
        with open('dataextractiontest2.csv', mode='w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(['Player Name', 'Course Name', 'Year Played', 'Date Completed',
                             'Round 1 Score', 'Round 2 Score', 'Round 3 Score', 'Round 4 Score', 
                             'Round 1 SG Total', 'Round 2 SG Total', 'Round 3 SG Total', 'Round 4 SG Total', 
                             'Round 1 SG App', 'Round 2 SG App', 'Round 3 SG App', 'Round 4 SG App', 
                             'Round 1 SG Arg', 'Round 2 SG Arg', 'Round 3 SG Arg', 'Round 4 SG Arg',
                             'Round 1 SG Putt', 'Round 2 SG Putt', 'Round 3 SG Putt', 'Round 4 SG Putt',
                             'Round 1 Driving Distance', 'Round 2 Driving Distance', 'Round 3 Driving Distance', 'Round 4 Driving Distance', 
                             'Round 1 Driving Accuracy', 'Round 2 Driving Accuracy', 'Round 3 Driving Accuracy', 'Round 4 Driving Accuracy'])
            for data_row in extracted_data:
                writer.writerow(data_row)
        print("CSV file created successfully!")

# Call main function with event_id, year, and API token
event_id = "535"
year = "2021"
API_TOKEN = "d6b6280403a3d0f3b7917387aed7"  # Replace this with your actual API token
main(event_id, year, API_TOKEN)
