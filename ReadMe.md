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

Now navigate to http://127.0.0.1:8000/

## API Interfaces
The easiest way to explore the api is to go to 
http://127.0.0.1:8000/docs 

The follow sections outline the different interfaces and the curl commands to interact with them.


### Model Training http://127.0.0.1:8000/train
Interface to training model. provide:
 * list of features to train on. Suggested default of
[ "time",  "ejection_fraction", "serum_creatinine"] 
 * path to file containing training data. Data should be stored in 
"data/heart_failure_clinical_records_dataset.csv"
* training and validation split
 

 
The following curl command will start training:
```bash
curl -X 'POST' \
  'http://127.0.0.1:8000/train' \
  -H 'accept: application/json' \
  -H 'Content-Type: application/json' \
  -d '{
  "list_of_features": [
    "time",
    "ejection_fraction",
    "serum_creatinine"
  ],
  "data_loc": "data/heart_failure_clinical_records_dataset.csv"
}'
```

The API will respond with the statistics for the validation set during training
* "accuracy": 0.875,
*  "f1_score": 0.7826086956521738,
* "area_under_curve": 0.9194365305476416,


### Model Prediction http://127.0.0.1:8000/predict
Once trained the model can be used to predict the likelihood of heart failure from a set of features.

Simply provide a list of values for the features which were trained on. 

The following curl will post a job
 ```bash
curl -X 'POST' \
  'http://127.0.0.1:8000/predict' \
  -H 'accept: application/json' \
  -H 'Content-Type: application/json' \
  -d '{
  "features": {
    "time": 293,
    "ejection_fraction": 100,
    "serum_creatinine": 130
  }
}'
```

This will return predictions for the ensemble and the individual predictions of each model

 
 ## ML architecture approach
Building on the well established work on this small dataset such as in [1], this quick proof of concept work aims to 
show that it is possible to train a large ensemble of simple machine learning classifiers on the e UCI Heart failure 
clinical records Data Set. 

Based of [1] ['time', 'ejection_fraction', 'serum_creatinine']
 
 ## Structure
 The repository contains the following files:
 * api.py: simple api to training algorithm visualising results and predicting on new data
 * data_utils.py: methods for loading and handling data
 * data_visualisation.py: methods for visualising data
 * ml_class.py: class for controlling 
 
 ## References 
  [1]  Chicco, D., Jurman, G. Machine learning can predict survival of patients with
   heart failure from serum creatinine and ejection fraction alone. BMC Med Inform
    Decis Mak 20, 16 (2020). https://doi.org/10.1186/s12911-020-1023-5
