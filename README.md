# AI-Golf

## Links 
- Data Golf (Loads of stats and data from lots of years) [DataGolf](https://datagolf.com/api-access)
- pga tour players data https://www.pgatour.com/players
- original model https://datagolfblogs.ca/a-predictive-model-of-tournament-outcomes-on-the-pga-tour/
- updated model https://datagolf.com/predictive-model-methodology/
## Parameters
* Driving Distance
* Driving Accuracy
* Strokes Gained Putting
* Strokes Gained Approach
* Strokes Gained Tee to Green
* Strokes Gained off the tee
* Greens in regulation
* Strokes gained Around the green

## ***IMPORTANT*** API Usage
The API key cannot be shared
events.py extracts the events data from the api along with their id. Its and example of how to use the api. On the https://datagolf.com/api-access website there is clearer explanations on how to use the api key, im still trying to work out exactly how to use it but if you have any questions text me. The one that i use is the historical raw data, Round Scoring, Stats & Strokes Gained section which has an example of how the data is stored. 
The apikeytes.py file is me trying to work out how to extract the individual player data. I managed to get it for certain competitions, you can change the id and year parameters to get different competitions. Ideally we will be able to extract them all and be able to out them all in a csv file which we can test and train. 
