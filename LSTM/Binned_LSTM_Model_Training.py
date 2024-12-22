import os
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import gc

# Configurar el allocador para reducir fragmentación de memoria
os.environ['TF_GPU_ALLOCATOR'] = 'cuda_malloc_async'

# Habilitar precisión mixta para reducir el uso de memoria
tf.keras.mixed_precision.set_global_policy('mixed_float16')

# Directorios
input_base_dir = r"C:\Users\Froxo\OneDrive - Universidad Técnica Federico Santa María\CONDOR_1-5E2_1000_Photon_2\binned_data"
predicts_dir = r"C:\Users\Froxo\OneDrive - Universidad Técnica Federico Santa María\ML_Predictions\LSTM_Predicts_Photon_1-5E2_binned"

# Crear carpeta para predicciones si no existe
os.makedirs(predicts_dir, exist_ok=True)

inc_energy = "1-5E2"
inc_particle = "Photon"

# Parámetros
time_steps = 100  # Ventana de tiempo
batch_size = 64
epochs = 100

file_name = f'LSTM_Predicts_{inc_energy}_{inc_particle}_{epochs}_epochs_binned_2.csv'

# Función para generar secuencias a partir de los datos
def create_sequences(data, time_steps=100, scaler=None):
    sequences = []

    # Asegurar que los datos estén ordenados por tiempo
    data = data.sort_values('t_bin')

    # Extraer y normalizar características
    features = data[['x_bin', 'y_bin', 'particle_count']].values
    if scaler:
        features = scaler.transform(features)

    # Crear secuencias
    for i in range(len(data) - time_steps):
        sequences.append(features[i:i + time_steps])

    return np.array(sequences)

# Inicializar listas para acumular secuencias y ángulos
all_sequences = []
all_angles = []

# Inicializar el scaler para normalizar los datos
scaler = MinMaxScaler()

# Preescanear los datos para ajustar el scaler
all_data = []
for run_folder in os.listdir(input_base_dir):
    run_path = os.path.join(input_base_dir, run_folder)

    if os.path.isdir(run_path) and "run_" in run_folder:
        run_number = int(run_folder.split("_")[1])

        # Usar solo las carpetas de run_14 a run_26
        if 1 <= run_number <= 31:
            for csv_file in os.listdir(run_path):
                if csv_file.endswith(".csv"):
                    csv_path = os.path.join(run_path, csv_file)
                    binned_data = pd.read_csv(csv_path)
                    all_data.append(binned_data[['x_bin', 'y_bin', 'particle_count']].values)

# Ajustar el scaler con todos los datos
scaler.fit(np.vstack(all_data))

# Procesar cada carpeta de run nuevamente
for run_folder in os.listdir(input_base_dir):
    run_path = os.path.join(input_base_dir, run_folder)

    if os.path.isdir(run_path) and "run_" in run_folder:
        run_number = int(run_folder.split("_")[1])

        # Usar solo las carpetas de run_14 a run_26
        if 1 <= run_number <= 31:
            print(f"Procesando {run_folder}...")

            # Procesar cada archivo CSV dentro de la carpeta
            for csv_file in os.listdir(run_path):
                if csv_file.endswith(".csv"):
                    csv_path = os.path.join(run_path, csv_file)

                    # Extraer el ángulo desde el nombre del archivo
                    try:
                        angle_str = csv_file.split("_")[7]  # Ajustar según el formato del nombre del archivo
                        angle = float(angle_str)
                    except (IndexError, ValueError):
                        print(f"Error al extraer el ángulo del archivo {csv_file}. Saltando archivo.")
                        continue

                    # Cargar datos
                    binned_data = pd.read_csv(csv_path)

                    # Generar secuencias
                    sequences = create_sequences(binned_data, time_steps, scaler)

                    # Agregar el ángulo a cada secuencia
                    if len(sequences) > 0:
                        all_sequences.append(sequences)
                        all_angles.extend([angle] * len(sequences))  # Repetir el ángulo para cada secuencia

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
input_shape = (time_steps, 3)  # time_steps=100, features=3 (x_bin, y_bin, particle_count)

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
stats_path = os.path.join(predicts_dir, file_name)
stats_df = pd.DataFrame({
    'True Angle': val_angles,
    'Predicted Angle': predictions.flatten()
})
stats_df['Absolute Error'] = np.abs(stats_df['True Angle'] - stats_df['Predicted Angle'])
stats_df.to_csv(stats_path, index=False)

# Imprimir resultados
print(f"Estadísticas guardadas en {stats_path}")
print(f"MAE: {eval_results[1]}, MSE: {eval_results[0]}")

# Limpiar memoria
tf.keras.backend.clear_session()
gc.collect()

