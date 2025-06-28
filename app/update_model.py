import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import load_model
import joblib
import os
import logging

# Настройка логирования
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Пути к файлам
model_path = 'models/temperature_model.keras'
scaler_X_path = 'models/scaler_X.save'
scaler_y_path = 'models/scaler_y.save'
data_path = 'data/temperature_data.csv'

# Загрузка модели и scaler'ов
if os.path.exists(model_path) and os.path.exists(scaler_X_path) and os.path.exists(scaler_y_path):
    model = load_model(model_path)
    scaler_X = joblib.load(scaler_X_path)
    scaler_y = joblib.load(scaler_y_path)
    logging.info("Модель и scaler'ы загружены.")
else:
    logging.error("Модель или scaler'ы не найдены. Сначала обучите модель.")
    raise FileNotFoundError("Модель или scaler'ы не найдены.")

# Загрузка новых данных
try:
    df_new = pd.read_csv(data_path)
    logging.info("Новые данные загружены.")
except FileNotFoundError:
    logging.error(f"Файл данных {data_path} не найден.")
    raise

X_new = df_new[['Температура на улице (°C)', 'Температура в доме (°C)', 'Заданная температура (°C)']]
y_new = df_new[['Температура на терморегуляторе (°C)', 'Температура кондиционера (°C)', 'Температура теплого пола (°C)']]

# Разделяем новые данные на обучающую и валидационную выборки
X_train_new, X_val_new, y_train_new, y_val_new = train_test_split(X_new, y_new, test_size=0.2, random_state=42)

# Обновляем scaler'ы на новых данных
scaler_X.fit(X_train_new)
scaler_y.fit(y_train_new)

# Применяем уже обученные scaler'ы
X_train_new_scaled = scaler_X.transform(X_train_new)
X_val_new_scaled = scaler_X.transform(X_val_new)

y_train_new_scaled = scaler_y.transform(y_train_new)
y_val_new_scaled = scaler_y.transform(y_val_new)

# Дообучение модели
history = model.fit(
    X_train_new_scaled, y_train_new_scaled,
    validation_data=(X_val_new_scaled, y_val_new_scaled),
    epochs=200,
    verbose=1
)

# Сохраняем обновленную модель
model.save(model_path)
logging.info(f"Обновленная модель сохранена в {model_path}")

# Сохраняем новые scaler'ы
joblib.dump(scaler_X, scaler_X_path)
joblib.dump(scaler_y, scaler_y_path)