from flask import Flask, render_template, request
import pandas as pd
import joblib
import numpy as np
import urllib.request
import urllib.parse

app = Flask(__name__)

print("Starting Flask App...")
print("Loading model...")
model = joblib.load('litemodel.sav')
print("Model loaded")

print("Loading columns...")
columns = joblib.load("columns.pkl")
print("Columns loaded")

def sendSMS(apikey, numbers, sender, message):
        data =  urllib.parse.urlencode({'apikey': apikey, 'numbers': numbers,'message' : message, 'sender': sender})
        data = data.encode('utf-8')
        request = urllib.request.Request("https://api.textlocal.in/send/?")
        f = urllib.request.urlopen(request, data)
        fr = f.read()
        return(fr)



def cal(ip):
    input = dict(ip)

    # Step 1: create raw input dict
    data_dict = {
        "Did_Police_Officer_Attend_Scene_of_Accident": float(input.get('Did_Police_Officer_Attend', ['0'])[0]),
        "Age_of_Driver": float(input.get('age_of_driver', ['0'])[0]),
        "Vehicle_Type": input.get('vehicle_type', ['0'])[0],
        "Age_of_Vehicle": float(input.get('age_of_vehicle', ['0'])[0]),
        "Engine_Capacity_(CC)": float(input.get('engine_cc', ['0'])[0]),
        "Day_of_Week": input.get('day', ['0'])[0],
        "Weather_Conditions": input.get('weather', ['0'])[0],
        "Road_Surface_Conditions": input.get('roadsc', ['0'])[0],
        "Light_Conditions": input.get('light', ['0'])[0],
        "Sex_of_Driver": input.get('gender', ['0'])[0],
        "Speed_limit": float(input.get('speedl', ['0'])[0])
    }

    # Step 2: convert to DataFrame
    df = pd.DataFrame([data_dict])

    # Step 3: apply encoding
    df = pd.get_dummies(df)

    # Step 4: align with training columns
    df_final = pd.DataFrame(columns=columns)

    for col in df.columns:
        if col in df_final.columns:
            df_final[col] = df[col]

    # fill missing columns with 0
    df_final = df_final.fillna(0)

    try:
        result = model.predict(df_final)
        return result[0]
    except Exception as e:
        return f"Error: {e}"






@app.route('/ml')
def ml():
    return render_template('ml.html')


@app.route('/map')
def map_page():
    return render_template('map.html')


@app.route('/police')
def police():
    return render_template('police.html')


@app.route('/visualization')
def visualization():
    return render_template('visual.html')

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        prediction = cal(request.form)
        return str(prediction)
    return render_template('index.html')

@app.route('/sms/', methods=['POST'])
def sms():
    res = cal(request.form)
    try:
        msg = f"Accident Severity: {res}"
        resp = sendSMS(
            'UwYs16dD3zM-DKuzZKQYolAJkoba1j0BmRGompsNRs',
            '919893018267',
            'TXTLCL',
            msg
        )
        print(resp)
    except Exception as e:
        print(e)

    return f"Prediction: {res} | SMS Sent"


if __name__ == '__main__':
    app.run(host='0.0.0.0', debug=True, port=4000)
