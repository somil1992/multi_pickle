#!/usr/bin/env python
# coding: utf-8

# In[3]:


from flask import Flask, jsonify, request
from flask_restful import Resource, Api
import pickle
import numpy as np
import json
import pandas as pd


# In[4]:


# creating the flask app
app = Flask(__name__)
# creating an API object
api = Api(app)


# In[5]:


#Loading Models

model_time_series = pickle.load(open('model_time_series_ev_arima.pkl', 'rb'))
model_regression = pickle.load(open('model_car_sale_price_prediction_regression.pkl', 'rb'))
model_classification = pickle.load(open('model_iris_classification.pkl', 'rb'))


# In[6]:


# making a class for a particular resource
# the get, post methods correspond to get and post requests
# they are automatically mapped by flask_restful.
class Predict_Time_Series(Resource):
    def post(self):

        data = request.files["input"]
        # data = request.form['gdp']
        data = pd.read_csv(data)
        data.set_index("date", inplace = True)
        print(data)

        prediction = model_time_series.predict(n_periods = 60, exogenous = data)

        output_list = []
        for i in prediction:
            output_list.append(i)
        output = pd.DataFrame()
        output["Date"] = data.index
        output["prediction"] = output_list
        output.to_csv("prediction1.csv",index = False)        
        return output.to_json()

class Predict_Car_Sales_Regression(Resource):
    def post(self):
        
        mileage = request.args.get('mileage')
        age = request.args.get('age')
        
        prediction = model_regression.predict([[mileage,age]])
        
        return prediction[0][0]

class Predict_Iris_Classification(Resource):
    def post(self):
        
        x1 = float(request.args.get('x1'))
        x2 = float(request.args.get('x2'))
        x3 = float(request.args.get('x3'))
        x4 = float(request.args.get('x4'))


        prediction = model_classification.predict([[x1,x2,x3,x4]])
        out = x1*x2
        return jsonify(prediction.tolist())


    
    
api.add_resource(Predict_Time_Series, '/predict_time_series')
api.add_resource(Predict_Car_Sales_Regression, '/predict_regression')
api.add_resource(Predict_Iris_Classification, '/predict_classification')

# driver function
if __name__ == '__main__':
    app.run(host="0.0.0.0",port=5000,debug = True)


# In[ ]:




