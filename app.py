from flask import Flask, render_template, request
from data import locations, property_type
from prediction import prediction
import pandas as pd
app = Flask(__name__)



# @app.route('/')
# def home():
#     return render_template('home.html')

@app.route('/', methods=["GET","POST"])
def index_predict():
    if request.method == "POST":
        data = request.form
        data = data.to_dict()
        data['accommodates'] = float(data['accommodates'])
        data['bedrooms'] = float(data['bedrooms'])
        data['cleaning_fee'] = float(data['cleaning_fee'])
        data['security_deposit'] = float(data['security_deposit'])
        data['guest_include'] = float(data['guest_include'])
        data['minimum_nights'] = float(data['minimum_nights'])
        hasil = prediction(data)
        return render_template('prediction.html', hasil_prediction = hasil)
    return render_template('prediction.html', data_location= sorted(locations),
    data_property=sorted(property_type))


@app.route('/visualization')
def visual():
    return render_template('visualization.html')

@app.route('/map1')
def first_map():
    return render_template('map1.html')

@app.route('/data')
def dataset():
    data = pd.read_csv('clean_listing.csv')
    return render_template('dataset.html', data=data)


if __name__ == '__main__':
    app.run(debug=True, port=1212)