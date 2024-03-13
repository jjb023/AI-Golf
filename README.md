# AI-Golf

## *Marking Criteria*
- clear problem you want to try and solve. You should also identify ways in which you can try to solve it (pros and cons)

- define what the **inputs** to the system are, what needs to happen with them, and what the **output** will be.

- For many projects evaluation may require collecting a dataset on which to test your system. Evaluation here may consist of metrics like accuracy, running time, memory usage, and how this compares to other approaches. More generally, you want to be able to produce some meaningful number that will convince others that your system is working as intended.

- 3 main parts
  - Data collection or acquiring a suitable existing dataset.
  - Implementing and extending an algorithm.
  - Building some sort of way of evaluating your system, and evaluating a few different versions of the system

- Lit review: Compare to other projects, why is it different (better?)
 
- **Evaluating**: a baseline might be a very simple machine learning system, such as linear regression/classification, or the first simplest network you tried.

## Questions for TA
- Ask about appendix - graphs, can we reference
- What results should we show? how should we show them
- Do we need reasoning for optimiser
- Refs, abstract, title part of page count?
- Can we send draft
- Have to use template
- Mean squared error vs absolute error for shots?
- Results section? 

## What results we want
- Sequence length: Extending sequence length against error
- Table of all computational times
- Table of all MSE

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
