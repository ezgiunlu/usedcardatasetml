import flask
from flask import Flask,request,jsonify
from flask_restful import Api, Resource, reqparse
from pycaret.regression import load_model, predict_model
import numpy as np
import json
import pandas as pd
import requests


app=Flask(__name__)
#@app.route('/hello/',methods=['POST'])
#def hello():
 #   json_data=flask.request.json
  #  return jsonify(json_data)

@app.route('/predict/', methods=['POST'])
def predict():
    predics=load_model('Xgb_Model')
    json_data=flask.request.json
    a=pd.DataFrame.from_dict(json_data,orient='index')
    b=pd.DataFrame.transpose(a)

    prediction=predict_model(predics,data=b)
    return str(prediction['Label'][0])

app.run(debug=True)
