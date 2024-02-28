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
def extract_data(api_data):
    extracted_data = []
    for score in api_data['scores']:
        player_name = score['player_name']
        year_played = api_data['year']
        date_completed = api_data['event_completed']
        for round_num in range(1, 5):
            round_data = score.get(f'round_{round_num}')
            if round_data:
                course_par = round_data['course_par']
                course_name = round_data['course_name']
                score_val = round_data.get('score')
                sg_total_val = round_data.get('sg_total')
                sg_app_val = round_data.get('sg_app')
                sg_arg_val = round_data.get('sg_arg')
                sg_putt_val = round_data.get('sg_putt')
                driving_dist_val = round_data.get('driving_dist')
                driving_acc_val = round_data.get('driving_acc')
            else:
                course_par = None
                course_name = None
                score_val = None
                sg_total_val = None
                sg_app_val = None
                sg_arg_val = None
                sg_putt_val = None
                driving_dist_val = None
                driving_acc_val = None
            
            extracted_data.append([player_name, course_name, year_played, date_completed, course_par, score_val,
                                   sg_total_val, sg_app_val, sg_arg_val, sg_putt_val,
                                   driving_dist_val, driving_acc_val])
    return extracted_data

# Main function to fetch data, extract, and write to CSV
def main(event_id, year, api_token):
    api_data = fetch_data(event_id, year, api_token)
    if api_data:
        extracted_data = extract_data(api_data)
        with open('Dataextractiontesting/dataextractiontest3.csv', mode='w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(['Player Name', 'Course Name', 'Year Played', 'Date Completed', 'Course Par',
                             'Round Score', 'Round SG Total', 'Round SG App', 'Round SG Arg', 'Round SG Putt',
                             'Round Driving Distance', 'Round Driving Accuracy'])
            for data_row in extracted_data:
                writer.writerow(data_row)
        print("CSV file created successfully!")

# Call main function with event_id, year, and API token
event_id = "535"
year = "2021"
API_TOKEN = "d6b6280403a3d0f3b7917387aed7"  # Replace this with your actual API token
main(event_id, year, API_TOKEN)
