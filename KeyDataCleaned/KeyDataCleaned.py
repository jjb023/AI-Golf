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
                
                # Append data only if all required fields are present
                if all([player_name, course_name, year_played, date_completed, course_par,
                        score_val, sg_total_val, sg_app_val, sg_arg_val, sg_putt_val,
                        driving_dist_val, driving_acc_val]):
                    extracted_data.append([player_name, course_name, year_played, date_completed, course_par,
                                           score_val, sg_total_val, sg_app_val, sg_arg_val, sg_putt_val,
                                           driving_dist_val, driving_acc_val])
            else:
                # Skip this round if round_data is missing
                continue
    
    return extracted_data


# Main function to fetch data, extract, and write to CSV for multiple event_ids
def main(event_ids, start_year, end_year, api_token):
    with open('KeyDataCleaned/KeyDataCleaned.csv', mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['Player Name', 'Course Name', 'Year Played', 'Date Completed', 'Course Par',
                         'Round Score', 'Round SG Total', 'Round SG App', 'Round SG Arg', 'Round SG Putt',
                         'Round Driving Distance', 'Round Driving Accuracy'])
        for year in range(start_year, end_year + 1):
            for event_id in event_ids:
                api_data = fetch_data(event_id, str(year), api_token)
                if api_data:
                    extracted_data = extract_data(api_data)
                    for data_row in extracted_data:
                        writer.writerow(data_row)

    print("CSV file created successfully!")

# Call main function with event_ids, start year, end year and API token
event_ids = [2,3,4,5,6,7,9,10,11,12,13,14,16,18,19,21,23,26,27,
             28,30,32,33,34,41,47,54,60,88,100,117,457,464,472,
             475,478,480,483,493,518,522,524,525,527,528,
             534,540,541,547,7795]
start_year = 2017
end_year = 2024
API_TOKEN = "api"  
main(event_ids, start_year, end_year, API_TOKEN)