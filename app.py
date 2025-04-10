from flask import Flask, render_template, request
import pickle
import numpy as np
from database import *
from sklearn.preprocessing import LabelEncoder
import joblib
app = Flask(__name__,static_url_path='/static')

# Load the machine learning model
 
# Load saved model and scaler

 
 
 

@app.route('/p')
def p():
    return render_template('index.html')

@app.route('/')
def m():
    return render_template('main.html')

@app.route('/l')
def l():
    return render_template('login.html')

@app.route('/h')
def h():
    return render_template('home.html')

@app.route('/r')
def r():
    return render_template('register.html')

@app.route('/m')
def menu():
    return render_template('menu.html')



@app.route("/register",methods=['POST','GET'])
def signup():
    if request.method=='POST':
        username=request.form['username']
        email=request.form['email']
        password=request.form['password']
        status = user_reg(username,email,password)
        if status == 1:
            return render_template("/login.html")
        else:
            return render_template("/register.html",m1="failed")        
    

@app.route("/login",methods=['POST','GET'])
def login():
    if request.method=='POST':
        username=request.form['username']
        password=request.form['password']
        status = user_loginact(request.form['username'], request.form['password'])
        print(status)
        if status == 1:                                      
            return render_template("/home.html", m1="sucess")
        else:
            return render_template("/login.html", m1="Login Failed")

@app.route('/predict', methods=['POST'])
def predict():
    sr = float(request.form['sr'])  # Snoring Rate
    rr = float(request.form['rr'])  # Respiration Rate
    t = float(request.form['t'])  # Body Temperature
    lm = float(request.form['lm'])  # Limb Movement
    bo = float(request.form['bo'])  # Blood Oxygen
    rem = float(request.form['rem'])  # Eye Movement
    sr2 = int(request.form['sr.1'])  # Sleeping Hours (renamed to avoid duplicate 'sr')
    hr = int(request.form['hr'])  # Heart Rate
    # Load saved model and scaler
    model = joblib.load("best_model_stress.pkl")
    
    # 
    input_data = np.array([[sr, rr, t, lm, bo, rem, sr2, hr]])
    prediction_result = model.predict(input_data)[0]

    print(f"Predicted stress level: {prediction_result}")
            # Convert input to DataFrame
    

    if prediction_result == 0:
        op1 = "This Person is not Stressed!"        
    if prediction_result == 1:
        op1 = "This Person is  Low Stress!"
    if prediction_result == 2:
        op1 = "This Person is  Medium Stressed!"        
    if prediction_result == 3:
        op1 = "This Person is  Highly Stressed!"
    

    return render_template("result.html", op1=op1)


if __name__ == "__main__":
    app.run(debug=True, port=5112)