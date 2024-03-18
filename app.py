import pickle 
from flask import Flask, request, app,jsonify,url_for,render_template
import numpy as np 
import pandas as pd 
import json


app = Flask(__name__)
##Load the model
regmodel = pickle.load(open('regmodel.pkl','rb'))
scalar = pickle.load(open('scaling.pk1','rb'))

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/predict_api',methods = ['POST'])
def predict_api():
    data = request.json['data']
     # Convert the dictionary values to a NumPy array
    data_array = np.array(list(data.values()))
    
    # Reshape the NumPy array to have the correct shape
    data_array_reshaped = data_array.reshape(1, -1)
    
    # Transform the data using the scalar
    new_data = scalar.transform(data_array_reshaped)
    
    # Convert the transformed data to a Python list
    new_data_list = new_data.tolist()
    
    # Use jsonify to convert the prediction output to JSON
    output = regmodel.predict(new_data_list)
    
    # Return the prediction result as JSON
    return jsonify(output[0])
    
if __name__ == '__main__':
    app.run(debug=True)