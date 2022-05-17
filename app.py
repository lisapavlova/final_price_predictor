from flask import Flask, request, abort
import joblib
import numpy
import pandas

EXTENDED_MODEL_PATH = 'mlmodels/model.pkl'
SIMPLE_MODEL_PATH = 'mlmodels/simple_model.pkl'
SCALER_X_PATH = 'mlmodels/scaler_x.pkl'
SCALER_Y_PATH = 'mlmodels/scaler_y.pkl'

app = Flask(__name__)
extended_model = joblib.load(EXTENDED_MODEL_PATH)
simple_model = joblib.load(SIMPLE_MODEL_PATH)
sc_X = joblib.load(SCALER_X_PATH)
sc_y = joblib.load(SCALER_Y_PATH)

@app.route("/predict_price", methods = ['GET'])
def predict():
    args = request.args
    model_version = args.get('model_version', type=str)
    floor = args.get('floor', type=int)
    open_plan = args.get('open_plan', type=int)
    rooms = args.get('rooms', type=int)
    studio = args.get('studio', type=int)
    area = args.get('area', type=float)
    kitchen_area = args.get('kitchen_area', type=float)
    living_area = args.get('living_area', type=float)
    renovation = args.get('renovation', type=int)
    building_id = args.get('building_id', type=int)
    year = args.get('year', type=int)

    if None in [model_version, floor, open_plan, rooms, studio, area, living_area, kitchen_area, renovation]:
        abort(400)

    if model_version == 'simple':
        x = pandas.DataFrame([{'floor': floor, 'open_plan': open_plan, 'rooms': rooms, 'studio': studio, 'area': area,
                               'kitchen_area': kitchen_area, 'living_area': living_area, 'renovation': renovation}])
        x.area = sc_X.transform(x[['area']])
        x.kitchen_area = sc_X.transform(x[['kitchen_area']])
        x.living_area = sc_X.transform(x[['living_area']])

        result = sc_y.inverse_transform(simple_model.predict(x).reshape(1, -1))
        return str(result[0][0])

    elif model_version == 'extended':
        if None in [building_id, year]:
            abort(400)

        x = pandas.DataFrame([{'floor': floor, 'open_plan': open_plan, 'rooms': rooms, 'studio': studio, 'area': area,
                               'kitchen_area': kitchen_area, 'living_area': living_area, 'renovation': renovation,
                               'building_id': building_id, 'year': year, 'av_room_area': living_area / rooms}])
        x.area = sc_X.transform(x[['area']])
        x.kitchen_area = sc_X.transform(x[['kitchen_area']])
        x.living_area = sc_X.transform(x[['living_area']])
        x.av_room_area = sc_X.transform(x[['av_room_area']])

        result = sc_y.inverse_transform(extended_model.predict(x).reshape(1, -1))
        return str(result[0][0])

    else:
        return 'Error: no such model version'


if __name__ == '__main__':
    app.run(debug=True, port=5444, host='0.0.0.0')
