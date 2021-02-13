
import flask
from flask import Flask,request,jsonify
from flask_restful import Api, Resource, reqparse
from pycaret.regression import load_model, predict_model
import requests

res = requests.post('http://127.0.0.1:5000/predict/',json={"year": "2007",
                                                          "manufacturer": "honda",
                                                          "condition": "like new",
                                                          "cylinders": "6 cylinders",
                                                          "fuel":np.nan,
                                                          "odometer":np.nan,
                                                          "title_status":"",
                                                          "transmission":"",
                                                          "drive":"4wd",
                                                          "size":np.nan,
                                                          "type":np.nan,})

print(res.text)
