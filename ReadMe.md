# UCR Heart ML Interface    
This repository provides a very simple interface to visualising a csv dataset training
 an ensemble classifier, evaluating the results and applying the trained model
 
 ## Quick Start Guide
 Install requirements from shell/command line using:
```bash
pip install -r requirements.txt
```

Start server with:
```bash
uvicorn api:app --reload
```

Now open http://127.0.0.1:8000/docs in browser

## API Interfaces
The easiest way to explore the api is to go to 
http://127.0.0.1:8000/docs 

The follow sections outline the different interfaces and the curl commands to interact with them.

### POST Commands
#### Model Training http://127.0.0.1:8000/train
Interface to training model. provide:
 * list of features to train on. Suggested default of
[ "time",  "ejection_fraction", "serum_creatinine"] 
 * path to file containing training data. Data should be stored in 
"data/heart_failure_clinical_records_dataset.csv"
* training and validation split
 

The following curl command will start training:
```bash
curl -d "{ \"list_of_features\":[\"time\", \"ejection_fraction\", \"serum_creatinine\"],  \"data_loc\":\"data/heart_failure_clinical_records_dataset.csv\", \"training_split\":0.6}" -X "POST" "http://127.0.0.1:8000/train" -H "Content-Type:application/json"
```

The API will respond with the statistics for the validation set during training
* "accuracy": 0.875,
*  "f1_score": 0.7826086956521738,
* "area_under_curve": 0.9194365305476416,


#### Model Prediction http://127.0.0.1:8000/predict
Once trained the model can be used to predict the likelihood of heart failure from a set of features.

Simply provide a list of values for the features which were trained on. 

The following curl command will post a job:
 ```bash
curl -d "{ \"features\":{\"time\": 23, \"ejection_fraction\": 38, \"serum_creatinine\":130}}" -X "POST" "http://127.0.0.1:8000/predict" -H "Content-Type:application/json"
```
This will return predictions for the ensemble and the individual predictions of each model


#### Model Evaluation http://127.0.0.1:8000/evaluate
This will repeat training a number of times and perform cross validation. Each run will perform a random training and 
validation split and return the results. The number of runs is controlled by the repeats parameter and the curl command 
is the following
```bash
curl -d "{ \"list_of_features\":[\"time\", \"ejection_fraction\", \"serum_creatinine\"],  \"data_loc\":\"data/heart_failure_clinical_records_dataset.csv\", \"training_split\":0.6, \"repeats\":10 }" -X "POST" "http://127.0.0.1:8000/evaluate" -H "Content-Type:application/json"
```
 
### GET commands
#### Root http://127.0.0.1:8000/
Will return json of all predictions that have been completed so far.

#### Pair plots http://127.0.0.1:8000/visualise/pair
Returns pair plot of feature visualisation. 
Note this requires training to be done before hand as this is where data is loaded.

#### Histogram plots http://127.0.0.1:8000/visualise/hist
Returns histogram of feature visualisation. 
Note this requires training to be done before hand as this is where data is loaded.

#### History of predictions http://127.0.0.1:8000/predict/history
Returns entire history of predictions 

#### Current trained model RoC curve plots http://127.0.0.1:8000/train/roc_curve
Returns roc curve of false positive rate vs true positive rate. Note this requires training to be done before hand.

#### Overall Evaluation RoC curve 'http://127.0.0.1:8000/evaluate/roc_curve'
Returns roc curve for evaluation if evaluation has been completed.
 
 ## ML architecture approach
Building on the well established work on this small dataset such as in [1], this quick proof of concept work aims to 
show that it is possible to train a large ensemble of simple machine learning classifiers on the e UCI Heart failure 
clinical records Data Set. 

Code from https://www.kaggle.com/andrewmvd/heart-failure-clinical-data/code inspired some of this work. In particular:

* https://www.kaggle.com/nayansakhiya/heart-fail-analysis-and-quick-prediction
* https://www.kaggle.com/sanchitakarmakar/heart-failure-prediction-visualization

Based of [1] ['time', 'ejection_fraction', 'serum_creatinine'] were chosen as the three main features to use although 
the interface and algorithm can use any set of features desired.

 
 ## Structure
 The repository contains the following files:
 * api.py: simple api to training algorithm visualising results and predicting on new data
 * data_utils.py: methods for loading and handling data
 * data_visualisation.py: methods for visualising data
 * ml_class.py: class for controlling machine learning training, prediction and evaluation. 
 Also contains some initial code to do hyper-paramter optimisation with optuna (although this is out of scope of this current work)
 
 ## References 
  [1]  Chicco, D., Jurman, G. Machine learning can predict survival of patients with
   heart failure from serum creatinine and ejection fraction alone. BMC Med Inform
    Decis Mak 20, 16 (2020). https://doi.org/10.1186/s12911-020-1023-5
