from flask import Flask, render_template, request
from data import locations, property_type, room_type, bed_type, instant_bookable, cancellation_policy
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
        data['bathrooms'] = float(data['bathrooms'])
        data['bedrooms'] = float(data['bedrooms'])
        data['cleaning_fee'] = float(data['cleaning_fee'])
        data['security_deposit'] = float(data['security_deposit'])
        data['extra_people'] = float(data['extra_people'])
        data['guest_include'] = float(data['guest_include'])
        data['minimum_nights'] = float(data['minimum_nights'])
        data['maximum_nights'] = float(data['maximum_nights'])
        data['review_score_rating'] = float(data['review_score_rating'])
        data['review_score_rating'] = float(data['review_score_rating'])
        data['balcony'] = float(data['balcony'])
        data['tv'] = float(data['tv'])
        data['coffee_machine'] = float(data['coffee_machine'])
        data['cooking_basics'] = float(data['cooking_basics'])
        data['dishwasher'] = float(data['dishwasher'])
        data['elevator'] = float(data['elevator'])
        data['child_friendly'] = float(data['child_friendly'])
        data['parking'] = float(data['parking'])
        data['internet'] = float(data['internet'])
        data['long_term_stays'] = float(data['long_term_stays'])
        data['pets_allowed'] = float(data['pets_allowed'])
        data['smoking_allowed'] = float(data['smoking_allowed'])
        hasil = prediction(data)
        return render_template('prediction.html', hasil_prediction = hasil)
    return render_template('prediction.html', data_location= sorted(locations),
    data_property=sorted(property_type), data_room_type=sorted(room_type), data_bed_type=sorted(bed_type),
    data_instant_bookable=sorted(instant_bookable), data_cancellation_policy=sorted(cancellation_policy))


@app.route('/visualization')
def visual():
    return render_template('visualization.html')


@app.route('/about')
def about():
    return render_template('about.html')

@app.route('/map1')
def first_map():
    return render_template('map1.html')

@app.route('/data')
def dataset():
    data = pd.read_csv('clean_listing.csv')
    return render_template('dataset.html', data=data)


if __name__ == '__main__':
    app.run(debug=True, port=1212)