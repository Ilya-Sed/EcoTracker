import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tensorflow import keras
from tensorflow.keras import layers, regularizers
import joblib
import os

# Пути к файлам
model_path = 'models/temperature_model.keras'
scaler_X_path = 'models/scaler_X.save'
scaler_y_path = 'models/scaler_y.save'

def load_data():
    """Загрузка данных из CSV файла."""
    df = pd.read_csv('data/temperature_data.csv')
    return df

def preprocess_data(df):
    """Предобработка данных: разделение на признаки и целевые переменные, стандартизация."""
    X = df[['Температура на улице (°C)', 'Температура в доме (°C)', 'Заданная температура (°C)']]
    y = df[['Температура на терморегуляторе (°C)', 'Температура кондиционера (°C)', 'Температура теплого пола (°C)']]

    # Разделяем данные на обучающую и тестовую выборки
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Стандартизируем данные
    scaler_X = StandardScaler()
    X_train = scaler_X.fit_transform(X_train)
    X_test = scaler_X.transform(X_test)

    # Сохраняем скалер для признаков
    joblib.dump(scaler_X, scaler_X_path)

    y_scaler = StandardScaler()
    y_train = y_scaler.fit_transform(y_train)
    y_test = y_scaler.transform(y_test)

    # Сохраняем скалер для целевых переменных
    joblib.dump(y_scaler, scaler_y_path)

    return X_train, X_test, y_train, y_test, y_scaler

def create_model(input_shape):
    """Создание модели регрессионной нейронной сети."""
    model = keras.Sequential([
        layers.Dense(32, activation='relu', input_shape=(input_shape,), kernel_regularizer=regularizers.l2(0.01)),
        layers.Dropout(0.3),
        layers.Dense(32, activation='relu', kernel_regularizer=regularizers.l2(0.01)),
        layers.Dropout(0.3),
        layers.Dense(3)
    ])
    optimizer = keras.optimizers.Adam(learning_rate=0.01)
    model.compile(optimizer=optimizer, loss='mean_squared_error')
    return model

def train_model(model, X_train, y_train):
    """Обучение модели."""
    history = model.fit(X_train, y_train, epochs=200, validation_split=0.2, verbose=1)
    return history

def plot_training_history(history):
    """Визуализация графиков обучения."""
    plt.figure(figsize=(12, 5))

    # График потерь
    plt.subplot(1, 2, 1)
    plt.plot(history.history['loss'], label='Обучение')
    plt.plot(history.history['val_loss'], label='Валидация')
    plt.title('Потери во время обучения')
    plt.xlabel('Эпохи')
    plt.ylabel('Потери')
    plt.legend()

    # График валидации потерь
    plt.subplot(1, 2, 2)
    plt.plot(history.history['val_loss'], label='Валидация')
    plt.title('Валидация потерь')
    plt.xlabel('Эпохи')
    plt.ylabel('Потери')
    plt.legend()
    plt.tight_layout()
    plt.show()

def evaluate_model(model, X_test, y_test, y_scaler):
    """Оценка модели на тестовых данных."""
    test_loss = model.evaluate(X_test, y_test)
    print(f'Потери на тестовых данных: {test_loss:.4f}')

    # Предсказания на тестовых данных
    y_pred = model.predict(X_test)

    # Обратная стандартизация предсказаний
    y_pred_inverse = y_scaler.inverse_transform(y_pred)
    y_test_inverse = y_scaler.inverse_transform(y_test)

    # Расчет коэффициента детерминации (R²)
    r2 = r2_score(y_test_inverse, y_pred_inverse)
    print(f'Коэффициент детерминации (R²): {r2:.4f}')

def save_model(model):
    """Сохранение обученной модели."""
    model.save(model_path)
    print(f'Обученная модель сохранена в {model_path}')

if __name__ == "__main__":
    df = load_data()
    X_train, X_test, y_train, y_test, y_scaler = preprocess_data(df)
    model = create_model(X_train.shape[1])
    history = train_model(model, X_train, y_train)
    plot_training_history(history)
    evaluate_model(model, X_test, y_test, y_scaler)
    save_model(model)