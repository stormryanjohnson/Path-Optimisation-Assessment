# Path Optimisation

A company called Protea Treks wants to optimise the path of a new trail that they are planning to build in
a mountainous region of the Free State. The trail is intended to be a beginner’s trail that must not require
too much exertion on behalf of its participants.

You are given a .csv file that describes an altitude map of the region, as well as a .csv of measurements
from a sports science lab that relates the walking gradient/slope on a treadmill with the energy expended
of multiple test subjects.

Your solution should consist of the following components:

1. Ingestion
    - Read from the .csv files and extract the relevant information from the datasets.
    - Note: the altitude map is at a resolution of 10m x 10m.
    - Note: energy expenditure is measured in $J.kg^{-1} .min^{-1}$ .
    - Note: the altitude map measurements are measured in meters, with North and the Y-axis
    going up vertically.
2. Modelling
    - Use any applicable statistics/machine learning method to predict a person’s expected
    energy expenditure for a given gradient.
3. Optimisation
    - Find a path from any point on the Southern border of the map to a lodge entrance at
    x=200 and y=559, which minimises the total expected exertion (in Joules).
    - You can use any optimisation method, provided it runs in a reasonable amount of time
    (less than 10 mins on standard hardware).
    - Note: for simplicity, you can assume that trail participants have a fixed body mass and a
    fixed walking speed.
4. Simple reporting
    - First write your path solution to a .csv file with the following column headings: x_coord
    and y_coord .
    - Then write your path, overlaid on the altitude map, to a .png file.
    - Your solution coordinates should be measured from the South-Western corner of the
    map (x=0, y=0) and should be measured in points that correspond to the resolution of the
    altitude map (i.e. one point for every 10m 2 square).
    - The company also wants advice about a possible future endurance trail. Write a short
    paragraph in a .txt file explaining what other information you would like to request from
    them and how you might change your approach in future.

The above steps should run end-to-end (.csv to results) with a single call to a script. Your solution must
be written using only open source tools (e.g. python, R, ...) so we can run your solution and check your
results.
