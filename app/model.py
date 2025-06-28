import numpy as np
import joblib
from tensorflow.keras.models import load_model

# Пути к файлам
model_path = 'models/temperature_model.keras'
scaler_X_path = 'models/scaler_X.save'
scaler_y_path = 'models/scaler_y.save'

# Загрузка модели и скалеров
model = load_model(model_path)
scaler = joblib.load(scaler_X_path)
y_scaler = joblib.load(scaler_y_path)

def predict_temperature(outdoor_temp, indoor_temp, set_temp):
    """
    Функция предсказания температуры с учетом ограничений.
    
    Параметры:
    - outdoor_temp: Температура на улице (°C)
    - indoor_temp: Температура в доме (°C)
    - set_temp: Заданная температура (°C)
    
    Возвращает:
    - Массив с предсказанными температурами [терморегулятор, кондиционер, теплый пол].
    """
    input_data = np.array([[outdoor_temp, indoor_temp, set_temp]])
    input_data_scaled = scaler.transform(input_data)
    predicted_temperatures_scaled = model.predict(input_data_scaled)

    # Обратное преобразование к исходному масштабу
    predicted_temperatures = y_scaler.inverse_transform(predicted_temperatures_scaled)

    # Применение ограничения к реальным значениям
    predicted_thermostat_temp = max(predicted_temperatures[0][0], indoor_temp, set_temp)
    if predicted_thermostat_temp < 20:
        predicted_thermostat_temp = 0
    elif predicted_thermostat_temp > 45:
        predicted_thermostat_temp = 45

    predicted_floor_temp = max(predicted_temperatures[0][2], 30)
    if predicted_floor_temp < 30:
        predicted_floor_temp = 0
    elif predicted_floor_temp > 45:
        predicted_floor_temp = 45

    predicted_ac_temp = max(predicted_temperatures[0][1], 17)
    if predicted_ac_temp > 28:
        predicted_ac_temp = 28

    # Логика управления устройствами
    if indoor_temp > set_temp:
        predicted_thermostat_temp = 0
        predicted_floor_temp = 0
    elif indoor_temp < set_temp:
        if predicted_thermostat_temp < 20:
            predicted_thermostat_temp = 0
        if predicted_floor_temp < 30:
            predicted_floor_temp = 0
        predicted_ac_temp = 0
    else:
        predicted_thermostat_temp = 0
        predicted_ac_temp = 0
        predicted_floor_temp = 0

    return np.array([[predicted_thermostat_temp, predicted_ac_temp, predicted_floor_temp]])

def calculate_energy_consumption(actual_temperatures, predicted_temperatures):
    """
    Функция для расчета показателей энергопотребления.
    
    Параметры:
    - actual_temperatures: Массив фактических температур.
    - predicted_temperatures: Массив предсказанных температур.
    
    Возвращает:
    - actual_energy: Общее фактическое энергопотребление.
    - predicted_energy: Общее предсказанное энергопотребление.
    - energy_change: Изменение энергопотребления.
    - percent_change: Процентное изменение энергопотребления.
    """
    actual_energy = np.sum(actual_temperatures[:, 0]) + np.sum(actual_temperatures[:, 1]) + np.sum(actual_temperatures[:, 2])
    predicted_energy = np.sum(predicted_temperatures[:, 0]) + np.sum(predicted_temperatures[:, 1]) + np.sum(predicted_temperatures[:, 2])

    energy_change = actual_energy - predicted_energy

    if actual_energy != 0:
        percent_change = (energy_change / actual_energy) * 100
    else:
        percent_change = 0

    return actual_energy, predicted_energy, energy_change, percent_change