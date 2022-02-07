# Anomaly Detectives
An in depth approach to detecting significant real-time shifts in network performance indicating network degradation. Building on the features of [DANE](https://github.com/dane-tool/dane), we build a classification system that determines if there are substantial changes to packet loss ratio and degree of latency. 

<br>

## To generate data for this project:

1. Generate data using our modified fork of [DANE](https://github.com/jenna-my/modified_dane)
    - ```make```, ```docker.io```, and ```docker-compose``` are required on your machine to run modified_dane properly.
    - a recursive flag is required to properly install modified_dane: <br>```git clone https://github.com/jenna-my/modified_dane --recursive```

2. Clone this branch of the repository
   ```
   git clone https://github.com/Ben243/DSC180A_Q1/tree/justin_branch
   ```

3. Place all raw DANE csv files within the directory ```data/raw``` of this repository. If the directory has not been created, run the command ```run.py``` once to generate all relevant directories.

<br>

## To use this repository: 
Each of these targets implements a core feature of the repository within ```run.py```. All code can be executed with the run.py according to various targets specified below. <br>
Example call: ```python run.py data inference``
### Target List:
- ```data```: generates features from unseen and seen data
- ```eda```: Generates visualizations used in exploring which features to use for the model
- ```train```: prints results of model performance tested on training ("seen") data with four different models with varying architectures: decision tree, random forest, extra trees, and gradient boost
- ```inference```: prints results of model performance tested on testing ("unseen") data with the same exact models.
- ```clean```: Removes files generated by targets in commonly used output directories
- ```test```: Verifies target functionality by running the targets ```data```,```eda```, ```train```, and ```inference``` with a subset of the original model training data.
- ```all```: runs all targets except ```test```

<br><br>

Our modified version of DANE creates csv files with a naming scheme in the following format: 
> *datevalue*_*latency*-*loss*-*deterministic*-*laterlatency*-*laterloss*-iperf.csv

e.g. ```20220117T015822_200-100-true-200-10000-iperf.csv```

this format is crucial for the model to train on the proper labels.

### columns.json
- object_list": []
- used inside of getAllCombinations(object_list) of train.py
- object list is lst inside of train.py, the list of all potential features to make combinations from (of varying length)

### eda.json
- "lst": [1, 2], # list of runs to compare side by side made by plottogether() inside of eda.py
- "filen1": "combined_subset_latency.csv", - subset of the processed data to make eda
- "filen2": "combined_t_latency.csv", - features generated from processed data
- "filen3": "combined_all_latency.csv" - all processed - 
