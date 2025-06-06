# AI-Golf [View the REPORT](Report.pdf)

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
To Email:
- Table of parameters in the appendix
- Last bit of further work, optimal sequence length
- paragraph about RNNs in lit review - necessary
- 

From meeting:
- Ask about appendix - graphs, can we reference
  make sure everything is in the report, appendix can be used but just for extras
- What results should we show? how should we show them
  
- Do we need reasoning for optimiser
  use fully connected cos lots of tabular data, have time data obvious lstm model
- Refs, abstract, title part of page count?
  abstract yes. refs no
- Can we send draft
  Yes
- Have to use template
  no, 
- Mean squared error vs absolute error for shots?
  RMSE root mean squared error
- Results section?
  loves the error bar graph with sequence length, do one for learning rate
Use time to compute
add all models into table might as well to say we compared on and built on linear
dont need to expand lstm
chose adamw as most common
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


