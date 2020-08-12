import numpy as np
import pandas as pd
from flask import Flask, request, jsonify, render_template
import pickle

app = Flask(__name__)
log_model = pickle.load(open('model.pkl', 'rb'))
model_columns = pickle.load(open('model_columns.pkl', 'rb'))

@app.route('/')
@app.route('/index')
def index():
    return render_template('index.html')

@app.route('/result', methods=['POST'])
def result():

    data = pd.DataFrame(0, index=np.arange(1), columns = model_columns)
    
    data.ses = int(request.form.get('ses'))
    data.pagediff = int(request.form.get('pagediff'))

    father_drug_1 = request.form.get('father_drug_1')
    father_drug_2 = request.form.get('father_drug_2')
    father_drug_3 = request.form.get('father_drug_3')
    father_drug_4 = request.form.get('father_drug_4')
    father_drug_5 = request.form.get('father_drug_5')

    mother_drug_1 = request.form.get('mother_drug_1')
    mother_drug_2 = request.form.get('mother_drug_2')
    mother_drug_3 = request.form.get('mother_drug_3')
    mother_drug_4 = request.form.get('mother_drug_4')
    mother_drug_5 = request.form.get('mother_drug_5')

    if (father_drug_1 or  father_drug_2 or father_drug_3 or father_drug_4 or father_drug_5):
        data.missing_p = 1

    if (mother_drug_1 or  mother_drug_2 or mother_drug_3 or mother_drug_4 or mother_drug_5):
        data.missing_m = 1

    if father_drug_1:
        data[father_drug_1] = 1
    if father_drug_2:
        data[father_drug_2] = 1
    if father_drug_3:
        data[father_drug_3] = 1
    if father_drug_4:
        data[father_drug_4] = 1
    if father_drug_5:
        data[father_drug_5] = 1

    if mother_drug_1:
        data[mother_drug_1] = 1
    if mother_drug_2:
        data[mother_drug_2] = 1
    if mother_drug_3:
        data[mother_drug_3] = 1
    if mother_drug_4:
        data[mother_drug_4] = 1
    if mother_drug_5:
        data[mother_drug_5] = 1

    output = log_model.predict(data)

    if int(output)==1:
        prediction='at high risk for autism.'

    else:
        prediction='at low risk for autism.'

    return render_template("result.html", prediction = prediction)

if __name__ == "__main__":
    app.run(debug=True)