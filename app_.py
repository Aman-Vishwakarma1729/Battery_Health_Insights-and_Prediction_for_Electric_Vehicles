from flask import Flask, render_template, request
import pickle
import h5py
import numpy as np

app = Flask(__name__)


# Load data from a PKL file
with open('base_learner1.pkl', 'rb') as file:
     base_model1 = pickle.load(file)


from tensorflow.keras.models import load_model

# Load the base_learner2 model from the HDF5 file
base_model2 = load_model('base_learner2.h5')

base_model3 = load_model('base_learner3.h5')



# Load the trained stacked model
with open('meta_model.pkl', 'rb') as file:
     meta_model = pickle.load(file)


@app.route('/')
def man():
    return render_template('home.html')

@app.route('/predict', methods=['POST'])
def home():
    data1 = float(request.form['a'])
    data2 = float(request.form['b'])
    data3 = float(request.form['c'])
    data4 = float(request.form['d'])
    data5 = float(request.form['e'])
    data6 = float(request.form['f']) 
    data7 = float(request.form['g'])
    new_data = np.array([[data1, data2, data3, data4, data5, data6, data7]])

    # Use base learners to make predictions for the new data point
    preds_base_learner1 = base_model1.predict(new_data)
    # Reshape the input data for neural network models to match the expected shape
    preds_base_learner2 = base_model2.predict(new_data.reshape(1, -1, 1))  # Reshape to (1, features, 1)
    preds_base_learner3 = base_model3.predict(new_data.reshape(1, -1, 1))  # Reshape to (1, features, 1)
    # Stack predictions from base learners
    stacked_new_data = np.column_stack((preds_base_learner1, preds_base_learner2.flatten(), preds_base_learner3.flatten()))
    # Use the meta-learner to predict the SoH for the new data point
    soh_prediction = meta_model.predict(stacked_new_data)

    return render_template('after.html', data=soh_prediction)


if __name__ == "__main__":
    app.run(debug=True)
