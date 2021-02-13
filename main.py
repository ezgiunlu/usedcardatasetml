from pycaret.datasets import get_data
import numpy as np
import pandas as pd
import os
import sys
from IPython.display import display

dataset = pd.read_csv('D:/Desktop/ezgi/vehicles.csv')
dataset=dataset.drop(["Unnamed: 0"],axis=1)
print(dataset.columns)
dataset.head()

dataset.describe()
dataset.info()
dataset.shape

droppeddata=dataset[(dataset['price']>50000) | (dataset['price']<1000)]
dataset=dataset.drop(droppeddata.index).reset_index(drop=True)

droppeddata_odo=dataset[((dataset['odometer']>500000))]
dataset=dataset.drop(droppeddata_odo.index).reset_index(drop=True)

data=dataset.sample(frac=0.7, random_state=786).reset_index(drop=True)
data_unseen=dataset.drop(data.index).reset_index(drop=True)

ignore=['id', 'url','region_url','image_url','posting_date','description','VIN','lat','long','model','region','paint_color','state']

data.shape
dataset.shape
droppeddata.shape
droppeddata_odo.shape

data.head()

data['cylinders'].value_counts()

from pycaret.regression import *
env=setup(data=data, target='price',session_id=123,
          ignore_features=ignore,use_gpu=True,remove_outliers=True,
         outliers_threshold=0.1,normalize=True,numeric_features=['odometer'])
         
compare_models(exclude=["lasso","en","par","huber","br","dt","omp","llar","lr","ridge","br"])

xgb=create_model('xgboost') 

tuned_xgb=tune_model(xgb)

plot_model(xgb,"error")

plot_model(tuned_xgb,"error")

evaluate_model(tuned_xgb)

predict_model(tuned_xgb)

final_xgb=finalize_model(tuned_xgb)

print(final_xgb)

unseen_predicts=predict_model(final_xgb,data=data_unseen)
unseen_predicts.head()

save_model(final_xgb,'Xgb_Model')

saved_final_xgb=load_model('Xgb_Model')
