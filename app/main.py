from flask import Flask, request, jsonify
from model import predict_temperature, calculate_energy_consumption
import numpy as np
import joblib
import os

app = Flask(__name__)

# Пути к файлам
model_path = 'models/temperature_model.keras'
scaler_X_path = 'models/scaler_X.save'
scaler_y_path = 'models/scaler_y.save'

# Загрузка скалеров
scaler = joblib.load(scaler_X_path)
y_scaler = joblib.load(scaler_y_path)

# Проверка, существует ли сохраненная модель
if os.path.exists(model_path):
    model = load_model(model_path)
else:
    raise Exception("Модель не найдена.")

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    outdoor_temp = data.get('outdoor_temp')
    indoor_temp = data.get('indoor_temp')
    set_temp = data.get('set_temp')

    predicted_values = predict_temperature(outdoor_temp, indoor_temp, set_temp)

    return jsonify({
        'thermostat_temp': predicted_values[0][0],
        'ac_temp': predicted_values[0][1],
        'floor_temp': predicted_values[0][2]
    })

@app.route('/calculate_energy', methods=['POST'])
def calculate_energy():
    data = request.json
    actual_temperatures = np.array(data.get('actual_temperatures'))

    predicted_values = np.array([predict_temperature(
        data.get('outdoor_temp'),
        data.get('indoor_temp'),
        data.get('set_temp')
    )[0]])

    actual_energy, predicted_energy, energy_change, percent_change = calculate_energy_consumption(actual_temperatures, predicted_values)

    return jsonify({
        'actual_energy': actual_energy,
        'predicted_energy': predicted_energy,
        'energy_change': energy_change,
        'percent_change': percent_change
    })

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)