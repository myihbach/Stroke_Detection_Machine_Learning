from flask import Flask, render_template, request
import pickle
import numpy as np


app = Flask(__name__)

model = pickle.load(open('Adaboost_model.pkl', 'rb'))


@app.route('/')
def index():
    return render_template('index.html')


def get_bmi(height, weight):
    bmi = weight / (height/100) ** 2
    if bmi < 23:
        return 0
    elif bmi < 29:
        return 1
    elif bmi < 36:
        return 2
    elif bmi < 47:
        return 3
    else:
        return 4


def get_message(answer):
    if answer == 0:
        return "Thanks God you are fine"
    else:
        return "You have to visit a Doctor asap"


@app.route('/diagnose', methods=['POST'])
def diagnose():
    if request.method == 'POST':
        result = request.form

    bmi = get_bmi(float(result["height"]), float(result["weight"]))
    data = [result["Age"], result["Avg_glucose_level"], result["work_type"], bmi, result["smoking_status"],
            result["Residence_type"], result["gender"], result["Ever_married"]]
    prediction = model.predict(np.array(data).reshape(1, 8))
    return render_template('index.html', result=prediction[0])


if __name__ == '__main__':
    app.run(debug=True, port=7320)
