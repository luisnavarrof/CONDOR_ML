import os
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import gc

# Configurar el allocador para reducir fragmentación de memoria
os.environ['TF_GPU_ALLOCATOR'] = 'cuda_malloc_async'

# Habilitar precisión mixta para reducir el uso de memoria
tf.keras.mixed_precision.set_global_policy('mixed_float16')

# Directorios
input_base_dir = r"C:\Users\Froxo\OneDrive - Universidad Técnica Federico Santa María\CONDOR_1-5E2_1000_Proton_2\particles_data"
predicts_dir = r"C:\Users\Froxo\OneDrive - Universidad Técnica Federico Santa María\ML_Predictions\LSTM_Predicts_1-5E2_Proton_2"

# Crear carpeta para predicciones si no existe
os.makedirs(predicts_dir, exist_ok=True)

# Parámetros
time_steps = 100  # Ventana de tiempo
batch_size = 32
epochs = 40
inc_energy = "1.5E2"
inc_particle = "Proton"

# Inicializar escaladores
scaler_features = StandardScaler()
scaler_angle = StandardScaler()

# Función para generar secuencias a partir de los datos
def create_sequences(data, time_steps=100):
    sequences = []

    # Ordenar los datos por tiempo
    data = data.sort_values('t')

    # Normalizar las características (x, y, t, E)
    features_df = data[['x', 'y', 't', 'E']]
    features_scaled = scaler_features.transform(features_df)

    # Crear secuencias
    for i in range(len(data) - time_steps):
        sequences.append(features_scaled[i:i + time_steps])

    return np.array(sequences)

# Inicializar listas para acumular secuencias y ángulos
all_sequences = []
all_angles = []

# Ajustar escaladores antes de generar secuencias
all_data_features = []
all_data_angles = []

# Procesar cada carpeta de run
for run_folder in os.listdir(input_base_dir):
    run_path = os.path.join(input_base_dir, run_folder)

    if os.path.isdir(run_path) and "run_" in run_folder:
        run_number = int(run_folder.split("_")[1])

        # Usar solo las carpetas relevantes
        if 1 <= run_number <= 31:
            print(f"Procesando {run_folder}...")

            # Procesar cada archivo CSV dentro de la carpeta
            for csv_file in os.listdir(run_path):
                if csv_file.endswith(".csv"):
                    csv_path = os.path.join(run_path, csv_file)

                    # Extraer el ángulo desde el nombre del archivo
                    try:
                        angle_str = csv_file.split("_")[6]  # Ajustar índice según el formato del nombre
                        angle = float(angle_str)
                    except (IndexError, ValueError):
                        print(f"Error al extraer el ángulo del archivo {csv_file}. Saltando archivo.")
                        continue

                    # Cargar datos
                    try:
                        particle_data = pd.read_csv(csv_path)
                    except pd.errors.EmptyDataError:
                        print(f"Archivo vacío o mal formateado: {csv_file}. Saltando archivo.")
                        continue

                    # Verificar si el archivo está vacío o faltan columnas
                    required_columns = ['x', 'y', 't', 'E']
                    if particle_data.empty or not all(col in particle_data.columns for col in required_columns):
                        print(f"Archivo vacío o con columnas faltantes: {csv_file}. Saltando archivo.")
                        continue

                    # Acumular características y ángulos para ajustar los escaladores
                    all_data_features.append(particle_data[['x', 'y', 't', 'E']])
                    all_data_angles.append(angle)

# Ajustar escaladores con todos los datos
all_data_features = pd.concat(all_data_features, ignore_index=True)
scaler_features.fit(all_data_features)
scaler_angle.fit(np.array(all_data_angles).reshape(-1, 1))

# Reiniciar procesamiento para crear secuencias normalizadas
for run_folder in os.listdir(input_base_dir):
    run_path = os.path.join(input_base_dir, run_folder)

    if os.path.isdir(run_path) and "run_" in run_folder:
        run_number = int(run_folder.split("_")[1])

        if 1 <= run_number <= 31:
            print(f"Procesando secuencias de {run_folder}...")

            for csv_file in os.listdir(run_path):
                if csv_file.endswith(".csv"):
                    csv_path = os.path.join(run_path, csv_file)

                    # Extraer el ángulo
                    try:
                        angle_str = csv_file.split("_")[6]
                        angle = float(angle_str)
                        angle_scaled = scaler_angle.transform([[angle]])[0, 0]
                    except (IndexError, ValueError):
                        continue

                    try:
                        particle_data = pd.read_csv(csv_path)
                    except pd.errors.EmptyDataError:
                        continue

                    if particle_data.empty or not all(col in particle_data.columns for col in required_columns):
                        continue

                    sequences = create_sequences(particle_data, time_steps)

                    if len(sequences) > 0:
                        all_sequences.append(sequences)
                        all_angles.extend([angle_scaled] * len(sequences))

# Concatenar todas las secuencias y ángulos
all_sequences = np.concatenate(all_sequences, axis=0)
all_angles = np.array(all_angles)

# Dividir en conjuntos de entrenamiento y validación
train_sequences, val_sequences, train_angles, val_angles = train_test_split(
    all_sequences, all_angles, test_size=0.2, random_state=42
)

# Liberar memoria
gc.collect()

# Definir el modelo LSTM
input_shape = (time_steps, 4)  # time_steps=100, features=4 (x, y, t, E)

model = tf.keras.Sequential([
    layers.Input(shape=input_shape),
    layers.LSTM(64, return_sequences=True),
    layers.LSTM(32),
    layers.Dense(16, activation='relu'),
    layers.Dense(1)  # Predicción del ángulo
])

# Compilar el modelo
model.compile(optimizer='adam', loss='mse', metrics=['mae'])

# Entrenar el modelo
history = model.fit(
    train_sequences, train_angles,
    validation_data=(val_sequences, val_angles),
    epochs=epochs,
    batch_size=batch_size
)

# Evaluar el modelo
eval_results = model.evaluate(val_sequences, val_angles, verbose=1)
predictions = model.predict(val_sequences)

# Guardar estadísticas en un archivo CSV
stats_path = os.path.join(predicts_dir, f'LSTM_Predicts_{inc_energy}_{inc_particle}_{epochs}_epochs_PP_2.csv')
stats_df = pd.DataFrame({
    'True Angle': scaler_angle.inverse_transform(val_angles.reshape(-1, 1)).flatten(),
    'Predicted Angle': scaler_angle.inverse_transform(predictions).flatten()
})
stats_df['Absolute Error'] = np.abs(stats_df['True Angle'] - stats_df['Predicted Angle'])
stats_df.to_csv(stats_path, index=False)

# Imprimir resultados
print(f"Estadísticas guardadas en {stats_path}")
print(f"MAE: {eval_results[1]}, MSE: {eval_results[0]}")

# Limpiar memoria
tf.keras.backend.clear_session()
gc.collect()
