from flask import Flask,render_template,request
import pickle
import numpy as np

model = pickle.load(open('Model.pkl','rb'))
app = Flask(__name__)



@app.route('/')
def home():
    return render_template('home.html')


@app.route('/predict', methods=['POST'])
def predict():
    data1 = float(request.form['a'])
    data2 = float(request.form['b'])
    data3 = float(request.form['c'])
    data4 = float(request.form['d'])
    data5 = float(request.form['e'])
    data6 = float(request.form['f'])
    data7 = float(request.form['g'])
    new_data = np.array([[data1, data2, data3, data4, data5, data6, data7]])
    soh_prediction = model.predict(new_data)
    return render_template('after.html', data=soh_prediction)


if __name__ == "__main__":
    app.run(debug=True)
