import pickle
import pandas as pd
import numpy as np
from flask import Flask,request,render_template

app = Flask(__name__)

model = pickle.load(open('model.pkl','rb'))
tf = pickle.load(open('tf.pkl','rb'))

@app.route('/')
def home():
    return render_template('index.html')


@app.route('/predict',methods = ['POST'])
def predict():
    if request.method == 'POST':
        message = request.form['message']
        data = [message]
        vect = tf.transform(data).toarray()
        my_pred = model.predict(vect)
        
    return render_template('result.html',prediction = my_pred)


if __name__ == '__main__':
    app.run(debug=True)