import csv

# Sample data
data = {
    "event_name": "U.S. Open",
    "event_id": "535",
    "tour": "pga",
    "event_completed": "2021-06-20",
    "year": 2021,
    "season": 2021,
    "sg_categories": "yes",
    "scores": [
        {
            "dg_id": 19195,
            "fin_text": "1",
            "player_name": "Rahm, Jon",
            "round_1": {
                "course_name": "Torrey Pines (South)",
                "course_num": 744,
                "course_par": 72,
                "teetime": "3:06pm",
                "start_hole": 10,
                "score": 69,
                "sg_app": 0.35,
                "sg_arg": 0.94,
                "sg_ott": 1.92,
                "sg_putt": 1.51,
                "sg_t2g": 3.21,
                "sg_total": 4.718,
                "driving_acc": 0.714,
                "driving_dist": 311.5,
                "gir": 0.611,
                "prox_fw": 34.77,
                "prox_rgh": 40.025,
                "scrambling": 0.778
            },
            "round_2": {
                "course_name": "Torrey Pines (South)",
                "course_num": 744,
                "course_par": 72,
                "teetime": "7:51pm",
                "start_hole": 1,
                "score": 70,
                "sg_app": 1.26,
                "sg_arg": 1.3,
                "sg_ott": -0.08,
                "sg_putt": 1.33,
                "sg_t2g": 2.48,
                "sg_total": 3.787,
                "driving_acc": 0.357,
                "driving_dist": 315.7,
                "gir": 0.667,
                "prox_fw": 37.182,
                "prox_rgh": 34.145,
                "scrambling": 0.75
            },
            "round_3": {
                "course_name": "Torrey Pines (South)",
                "course_num": 744,
                "course_par": 72,
                "teetime": "1:13pm",
                "start_hole": 1,
                "score": 72,
                "sg_app": 0.77,
                "sg_arg": 0.3,
                "sg_ott": 1.04,
                "sg_putt": -1.7,
                "sg_t2g": 2.11,
                "sg_total": 0.408,
                "driving_acc": 0.429,
                "driving_dist": 316.8,
                "gir": 0.778,
                "prox_fw": 31.819,
                "prox_rgh": 26.096,
                "scrambling": 0.5
            },
            "round_4": {
                "course_name": "Torrey Pines (South)",
                "course_num": 744,
                "course_par": 72,
                "teetime": "12:22pm",
                "start_hole": 1,
                "score": 67,
                "sg_app": 2.61,
                "sg_arg": -0.27,
                "sg_ott": 1.35,
                "sg_putt": 2.48,
                "sg_t2g": 3.69,
                "sg_total": 6.169,
                "driving_acc": 0.571,
                "driving_dist": 323.1,
                "gir": 0.778,
                "prox_fw": 22.699,
                "prox_rgh": 41.028,
                "scrambling": 0.8
            }
        }
    ]
}

# Extracting data
player_name = data['scores'][0]['player_name']
course_name = data['scores'][0]['round_1']['course_name']
year_played = data['year']
scores = [data['scores'][0]['round_1']['score'], data['scores'][0]['round_2']['score'],
          data['scores'][0]['round_3']['score'], data['scores'][0]['round_4']['score']]
sg_values = [
    data['scores'][0]['round_1']['sg_total'],
    data['scores'][0]['round_2']['sg_total'],
    data['scores'][0]['round_3']['sg_total'],
    data['scores'][0]['round_4']['sg_total']
]
driving_distances = [
    data['scores'][0]['round_1']['driving_dist'],
    data['scores'][0]['round_2']['driving_dist'],
    data['scores'][0]['round_3']['driving_dist'],
    data['scores'][0]['round_4']['driving_dist']
]
driving_accuracies = [
    data['scores'][0]['round_1']['driving_acc'],
    data['scores'][0]['round_2']['driving_acc'],
    data['scores'][0]['round_3']['driving_acc'],
    data['scores'][0]['round_4']['driving_acc']
]

# Writing data to CSV file
with open('golf_data.csv', mode='w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(['Player Name', 'Course Name', 'Year Played', 'Round 1 Score', 'Round 2 Score',
                     'Round 3 Score', 'Round 4 Score', 'Round 1 SG', 'Round 2 SG', 'Round 3 SG',
                     'Round 4 SG', 'Round 1 Driving Distance', 'Round 2 Driving Distance',
                     'Round 3 Driving Distance', 'Round 4 Driving Distance', 'Round 1 Driving Accuracy',
                     'Round 2 Driving Accuracy', 'Round 3 Driving Accuracy', 'Round 4 Driving Accuracy'])
    writer.writerow([player_name, course_name, year_played] + scores + sg_values +
                    driving_distances + driving_accuracies)

print("CSV file created successfully!")
