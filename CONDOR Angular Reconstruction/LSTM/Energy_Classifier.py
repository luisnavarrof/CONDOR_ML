import os
import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense

# Energías especificadas
energies = ["13E1", "5E2", "8E2", "3E2", "2E2", "1E2", "15E1"]

# Directorio base
data_dir = r"C:\Users\Froxo\OneDrive - Universidad Técnica Federico Santa María\CONDOR_Angle_2"
output_dir = r"C:\Users\Froxo\OneDrive - Universidad Técnica Federico Santa María\ML_Predictions\LSTM_Predicts_Energy"
os.makedirs(output_dir, exist_ok=True)

# Parámetros del modelo
time_steps = 100
epochs = 20
batch_size = 32

# Función para cargar y procesar datos
def load_and_process_data(directory):
    all_files = [os.path.join(directory, f) for f in os.listdir(directory) if f.endswith('.csv')]
    data = []

    for file in tqdm(all_files, desc=f"Procesando datos en {directory}"):
        df = pd.read_csv(file)
        data.append(df)

    if not data:
        return None

    combined_data = pd.concat(data, ignore_index=True)
    return combined_data

# Preparar datos para LSTM
def prepare_lstm_data(data):
    scaler = MinMaxScaler()
    features = data[["x_bin", "y_bin", "particle_count"]].values
    labels = data["energy"].values  # Esta columna debe ser añadida al cargar los datos

    features_scaled = scaler.fit_transform(features)
    X, y = [], []

    for i in range(len(features_scaled) - time_steps):
        X.append(features_scaled[i:i + time_steps])
        y.append(labels[i + time_steps - 1])

    return np.array(X), np.array(y)

# Crear modelo LSTM
def create_model(input_shape):
    model = Sequential([
        LSTM(64, return_sequences=True, input_shape=input_shape),
        LSTM(64),
        Dense(1, activation='linear')
    ])
    model.compile(optimizer='adam', loss='mse', metrics=['mae'])
    return model

# Cargar y procesar datos de todas las energías
all_data = []
for energy in energies:
    energy_dir = os.path.join(data_dir, f"CONDOR_{energy}_Proton", "binned_data")

    if not os.path.exists(energy_dir):
        print(f"Directorio no encontrado: {energy_dir}. Saltando...")
        continue

    energy_data = load_and_process_data(energy_dir)
    if energy_data is not None:
        energy_data["energy"] = float(energy.replace("E", "e"))  # Convertir energía a formato numérico
        all_data.append(energy_data)

if not all_data:
    raise ValueError("No se encontró ningún dato válido en los directorios especificados.")

# Combinar todos los datos
final_data = pd.concat(all_data, ignore_index=True)
X, y = prepare_lstm_data(final_data)

# Dividir en entrenamiento y prueba
split_idx = int(0.8 * len(X))
X_train, X_test = X[:split_idx], X[split_idx:]
y_train, y_test = y[:split_idx], y[split_idx:]

# Crear y entrenar el modelo
model = create_model(input_shape=(time_steps, X.shape[2]))
model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=epochs, batch_size=batch_size)

# Evaluar el modelo
loss, mae = model.evaluate(X_test, y_test)
print(f"MAE: {mae}, Loss: {loss}")

# Guardar resultados y modelo
results_file = os.path.join(output_dir, "LSTM_Energy_Predictions.csv")
pd.DataFrame({"True Energy": y_test, "Predicted Energy": model.predict(X_test).flatten()}).to_csv(results_file, index=False)

model_file = os.path.join(output_dir, "lstm_energy_model.h5")
model.save(model_file)
print(f"Resultados guardados en {results_file}")
print(f"Modelo guardado en {model_file}")
