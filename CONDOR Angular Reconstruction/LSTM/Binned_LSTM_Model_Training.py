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

# Configurar TensorFlow para que no preasigne toda la memoria de la GPU
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    except RuntimeError as e:
        print(e)

# Directorios
input_base_dir = r"C:\Users\Froxo\OneDrive - Universidad Técnica Federico Santa María\CONDOR_Angle_2\CONDOR_13E1_Proton\binned_data"
predicts_dir = r"C:\Users\Froxo\OneDrive - Universidad Técnica Federico Santa María\ML_Predictions\LSTM_Predicts_Proton_13E1"

# Crear carpeta para predicciones si no existe
os.makedirs(predicts_dir, exist_ok=True)

inc_energy = "13E1"
inc_particle = "Proton"

# Parámetros
time_steps = 100 # Ventana de tiempo
batch_size = 32  # Reducir tamaño del lote
epochs = 100

file_name = f'LSTM_Predicts_{inc_energy}_{inc_particle}_{epochs}_epochs_Binned.csv'
model_save_path = os.path.join(predicts_dir, f'lstm_model_{inc_energy}_{inc_particle}_{epochs}_epochs_Binned.h5')  # Ruta para guardar el modelo

# Función para generar secuencias a partir de los datos
def create_sequences(data, time_steps=time_steps, scaler=None):
    sequences = []

    # Asegurar que los datos estén ordenados por tiempo
    data = data.sort_values('t_bin')

    # Extraer y normalizar características
    features = data[['x_bin', 'y_bin', 'particle_count']].values
    if scaler:
        features = scaler.transform(features)

    # Crear secuencias
    for i in range(len(features) - time_steps):
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

        if 1 <= run_number <= 31:  # Filtrar rangos
            for csv_file in os.listdir(run_path):
                if csv_file.endswith(".csv"):
                    csv_path = os.path.join(run_path, csv_file)

                    if os.stat(csv_path).st_size == 0:  # Verificar archivos vacíos
                        print(f"Archivo vacío encontrado: {csv_file}. Saltando archivo.")
                        continue

                    binned_data = pd.read_csv(csv_path)
                    all_data.append(binned_data[['x_bin', 'y_bin', 'particle_count']].values)

# Ajustar el scaler si hay datos suficientes
if all_data:
    scaler.fit(np.vstack(all_data))

# Procesar cada carpeta de run nuevamente
for run_folder in os.listdir(input_base_dir):
    run_path = os.path.join(input_base_dir, run_folder)

    if os.path.isdir(run_path) and "run_" in run_folder:
        run_number = int(run_folder.split("_")[1])

        if 1 <= run_number <= 31:
            print(f"Procesando {run_folder}...")

            for csv_file in os.listdir(run_path):
                if csv_file.endswith(".csv"):
                    csv_path = os.path.join(run_path, csv_file)

                    if os.stat(csv_path).st_size == 0:  # Ignorar archivos vacíos
                        print(f"Archivo vacío encontrado: {csv_file}. Saltando archivo.")
                        continue

                    try:
                        angle_str = csv_file.split("_")[7]
                        angle = float(angle_str)
                    except (IndexError, ValueError):
                        print(f"Error al extraer el ángulo del archivo {csv_file}. Saltando archivo.")
                        continue

                    binned_data = pd.read_csv(csv_path)

                    # Generar secuencias
                    sequences = create_sequences(binned_data, time_steps, scaler)

                    if sequences is not None and len(sequences) > 0:
                        all_sequences.append(sequences)
                        all_angles.extend([angle] * len(sequences))

# Verificar si hay datos suficientes para el modelo
if not all_sequences:
    print("No se encontraron datos suficientes para entrenar el modelo.")
    exit()

all_sequences = np.concatenate(all_sequences, axis=0)
all_angles = np.array(all_angles)

# Dividir en conjuntos de entrenamiento y validación
train_sequences, val_sequences, train_angles, val_angles = train_test_split(
    all_sequences, all_angles, test_size=0.2, random_state=42
)

gc.collect()

# Definir el modelo LSTM
input_shape = (time_steps, 3)

model = tf.keras.Sequential([
    layers.Input(shape=input_shape),
    layers.LSTM(64, return_sequences=True),
    layers.LSTM(32),
    layers.Dense(16, activation='relu'),
    layers.Dense(1)
])

model.compile(optimizer='adam', loss='mse', metrics=['mae'])

# Entrenar el modelo
history = model.fit(
    train_sequences, train_angles,
    validation_data=(val_sequences, val_angles),
    epochs=epochs,
    batch_size=batch_size
)

model.save(model_save_path)
print(f"Modelo guardado en {model_save_path}")

eval_results = model.evaluate(val_sequences, val_angles, verbose=1)
predictions = model.predict(val_sequences)

stats_path = os.path.join(predicts_dir, file_name)
stats_df = pd.DataFrame({
    'True Angle': val_angles,
    'Predicted Angle': predictions.flatten()
})
stats_df['Absolute Error'] = np.abs(stats_df['True Angle'] - stats_df['Predicted Angle'])
stats_df.to_csv(stats_path, index=False)

print(f"Estadísticas guardadas en {stats_path}")
print(f"MAE: {eval_results[1]}, MSE: {eval_results[0]}")

tf.keras.backend.clear_session()
gc.collect()
