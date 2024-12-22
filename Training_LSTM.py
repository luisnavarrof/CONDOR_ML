import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from tqdm import tqdm  # Importar barra de progreso

# Configuración
base_dir = r"C:\Users\Froxo\OneDrive - Universidad Técnica Federico Santa María\CONDOR_Angle_2"
output_dir = r"C:\Users\Froxo\OneDrive - Universidad Técnica Federico Santa María\ML_Predictions\Results"

# Crear carpeta de resultados si no existe
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# Inicializar variables
all_data = []
all_labels = []
scaler = MinMaxScaler()

# Función auxiliar para extraer energía del nombre del archivo
def extract_energy(filename):
    try:
        return float(filename.split("_")[3])
    except (IndexError, ValueError):
        return None

# Proceso: Extraer datos de "Binned_data/run_x"
folders = os.listdir(base_dir)
total_files = sum(
    len(files) 
    for folder in folders
    for run_folder in os.listdir(os.path.join(base_dir, folder, "Binned_data"))
    for files in os.listdir(os.path.join(base_dir, folder, "Binned_data", run_folder))
)

# Barra de progreso global
print("Iniciando procesamiento de datos...")
with tqdm(total=total_files, desc="Procesando archivos") as pbar:
    for folder in folders:
        binned_path = os.path.join(base_dir, folder, "Binned_data")
        if os.path.isdir(binned_path):
            for run_folder in os.listdir(binned_path):
                run_path = os.path.join(binned_path, run_folder)
                if os.path.isdir(run_path):
                    for file in os.listdir(run_path):
                        if file.endswith(".csv"):
                            energy = extract_energy(file)
                            if energy is None:
                                pbar.update(1)
                                continue

                            # Leer el archivo CSV
                            file_path = os.path.join(run_path, file)
                            try:
                                df = pd.read_csv(file_path)
                                if {'x_bin', 'y_bin', 'particle_count'}.issubset(df.columns):
                                    features = df[['x_bin', 'y_bin', 'particle_count']].values
                                    all_data.append(features)
                                    all_labels.append(energy)
                            except:
                                pass
                            pbar.update(1)

# Verificar si hay datos
if len(all_data) == 0 or len(all_labels) == 0:
    print("No se encontraron datos válidos. Revisa las rutas y archivos.")
    exit()

# Escalar los datos
all_data_flat = [np.vstack(run) for run in all_data]
scaler.fit(np.vstack(all_data_flat))
all_data_scaled = [scaler.transform(run) for run in all_data_flat]

# Preparar secuencias LSTM
X, y = [], []
sequence_length = 10  # Longitud de la secuencia temporal

for i, run_data in enumerate(all_data_scaled):
    for j in range(len(run_data) - sequence_length):
        X.append(run_data[j:j+sequence_length])
        y.append(all_labels[i])

X = np.array(X)
y = np.array(y)

# Separar datos de entrenamiento y validación
split = int(0.8 * len(X))
X_train, X_val = X[:split], X[split:]
y_train, y_val = y[:split], y[split:]

# Modelo LSTM
model = Sequential([
    LSTM(64, activation='tanh', input_shape=(sequence_length, 3)),
    Dense(32, activation='relu'),
    Dense(1)
])

model.compile(optimizer='adam', loss='mse', metrics=['mae'])

# Entrenamiento con barra de progreso
print("Entrenando modelo...")
history = model.fit(
    X_train, y_train, 
    validation_data=(X_val, y_val), 
    epochs=10, batch_size=32, verbose=0, 
    callbacks=[tqdm(total=10, desc="Entrenando", position=0, leave=True)]
)

# Guardar resultados
results = model.evaluate(X_val, y_val, verbose=0)
model.save(os.path.join(output_dir, "LSTM_energy_prediction_model.h5"))

stats = pd.DataFrame({
    'Loss': [results[0]],
    'MAE': [results[1]]
})
stats.to_csv(os.path.join(output_dir, "LSTM_model_results.csv"), index=False)

print("Proceso finalizado. Resultados guardados.")
