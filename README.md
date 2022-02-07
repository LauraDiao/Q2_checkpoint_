# Final report code

## transform.json: used for the test target
## all.json: for user to modify
#     "runs": [1, 2, 5, 6], this corresponds to the (seen data) runs in the test directory of observed data to train/test the model on
#    "unseen_runs": [3, 4], this corresponds to (unseen data) runs in the test directory that the model, trained on seen data, will be tested on the run folder
#    "subset": 8 # relevant for the eda, a subset of 8 seconds to compare to visualizations generated on the whole dataset
# to configure the data: 
# the user should have data formated to have folders named "run"
# since the tool used to generate data, DANE, creates data from each configured "run"
# data generated from multilple runs would look like, nested inside of the "test" folder
# run1
# run2
# run3
# there is no requirement/limit to how many run folders there are
# additionally DANE makes data csv names in the following format: datetime_loss_latency-iperf.csv
# for example: 20211130T052220_240-30000-iperf.csv
# the loss and latency are crucial to have in the file name

# columns.json
# object_list": []
# used inside of getAllCombinations(object_list) of train.py
# object list is lst inside of train.py, the list of all potential features to make combinations from (of varying length)

# eda.json
#     "lst": [1, 2], # list of runs to compare side by side made by plottogether() inside of eda.py
#    "filen1": "combined_subset_latency.csv", - subset of the processed data to make eda
#   "filen2": "combined_t_latency.csv", - features generated from processed data
#    "filen3": "combined_all_latency.csv" - all processed data
#
