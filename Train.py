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
input_base_dir = r"C:\Users\Froxo\OneDrive - Universidad Técnica Federico Santa María\CONDOR_Angle_2"
predicts_dir = r"C:\Users\Froxo\OneDrive - Universidad Técnica Federico Santa María\ML_Predictions\LSTM_Predicts_Proton_Photon"

# Crear carpeta para predicciones si no existe
os.makedirs(predicts_dir, exist_ok=True)

# Parámetros
time_steps = 50  # Ventana de tiempo
batch_size = 8
epochs = 100

model_save_path = os.path.join(predicts_dir, 'lstm_model_1E2_to_2E2_combined.h5')  # Ruta para guardar el modelo

# Función para generar secuencias a partir de los datos
def create_sequences(data, time_steps=100, scaler=None, energy_label=None, particle_label=None):
    sequences = []
    energy_labels = []
    particle_labels = []

    # Asegurar que los datos estén ordenados por tiempo
    data = data.sort_values('t_bin')

    # Extraer y normalizar características
    features = data[['x_bin', 'y_bin', 'particle_count']].values
    if scaler:
        features = scaler.transform(features)

    # Crear secuencias
    for i in range(len(data) - time_steps):
        sequences.append(features[i:i + time_steps])
        energy_labels.append(energy_label)
        particle_labels.append(particle_label)

    return np.array(sequences), np.array(energy_labels), np.array(particle_labels)

# Inicializar el scaler para normalizar los datos
scaler = MinMaxScaler()

# Función para preescanear los datos y ajustar el scaler
def pre_scan_data(input_base_dir):
    all_data = []
    for folder in os.listdir(input_base_dir):
        if "2E2" in folder or "1E2" in folder or "1-5E2" in folder:
            folder_path = os.path.join(input_base_dir, folder, 'binned_data')
            if os.path.isdir(folder_path):
                for run_folder in os.listdir(folder_path):
                    run_path = os.path.join(folder_path, run_folder)
                    if os.path.isdir(run_path) and "run_" in run_folder:
                        run_number = int(run_folder.split("_")[1])
                        if 1 <= run_number <= 31:
                            for csv_file in os.listdir(run_path):
                                if csv_file.endswith(".csv"):
                                    csv_path = os.path.join(run_path, csv_file)
                                    binned_data = pd.read_csv(csv_path)
                                    all_data.append(binned_data[['x_bin', 'y_bin', 'particle_count']].values)
    # Ajustar el scaler con todos los datos
    if all_data:
        scaler.fit(np.vstack(all_data))
    else:
        raise ValueError("No se encontraron archivos CSV para ajustar el scaler.")

# Función para convertir etiquetas de energía a valores numéricos
def convert_energy_label(energy_label):
    if energy_label == "1-5E2":
        return 1.5e2
    return float(energy_label.replace("E2", "e2"))

# Función para procesar los datos y generar secuencias
def process_data(input_base_dir, time_steps, scaler):
    all_sequences = []
    all_angles = []
    all_energy_labels = []
    all_particle_labels = []
    for folder in os.listdir(input_base_dir):
        if "2E2" in folder or "1E2" in folder or "1-5E2" in folder:
            folder_path = os.path.join(input_base_dir, folder, 'binned_data')
            if os.path.isdir(folder_path):
                energy_label = convert_energy_label(folder.split("_")[1])
                particle_label = 0 if "Proton" in folder else 1
                for run_folder in os.listdir(folder_path):
                    run_path = os.path.join(folder_path, run_folder)
                    if os.path.isdir(run_path) and "run_" in run_folder:
                        run_number = int(run_folder.split("_")[1])
                        if 1 <= run_number <= 31:
                            print(f"Procesando {run_folder} en {folder}...")

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
                                    sequences, energy_labels, particle_labels = create_sequences(
                                        binned_data, time_steps, scaler, energy_label, particle_label)

                                    # Agregar el ángulo, la energía y el tipo de partícula a cada secuencia
                                    if len(sequences) > 0:
                                        all_sequences.append(sequences)
                                        all_angles.extend([angle] * len(sequences))  # Repetir el ángulo para cada secuencia
                                        all_energy_labels.extend(energy_labels)  # Repetir la energía para cada secuencia
                                        all_particle_labels.extend(particle_labels)  # Repetir el tipo de partícula para cada secuencia

    # Concatenar todas las secuencias, ángulos, etiquetas de energía y etiquetas de partículas
    all_sequences = np.concatenate(all_sequences, axis=0)
    all_angles = np.array(all_angles)
    all_energy_labels = np.array(all_energy_labels)
    all_particle_labels = np.array(all_particle_labels)

    return all_sequences, all_angles, all_energy_labels, all_particle_labels

# Preescanear los datos para ajustar el scaler
pre_scan_data(input_base_dir)

# Procesar los datos y generar secuencias
train_sequences, train_angles, train_energy_labels, train_particle_labels = process_data(input_base_dir, time_steps, scaler)

# Dividir en conjuntos de entrenamiento y validación
train_sequences, val_sequences, train_angles, val_angles, train_energy_labels, val_energy_labels, train_particle_labels, val_particle_labels = train_test_split(
    train_sequences, train_angles, train_energy_labels, train_particle_labels, test_size=0.2, random_state=42
)

# Liberar memoria
gc.collect()

# Definir el modelo LSTM con tres salidas
input_shape = (time_steps, 3)  # time_steps, features=3 (x_bin, y_bin, particle_count)

input_layer = layers.Input(shape=input_shape)
lstm_layer_1 = layers.LSTM(64, return_sequences=True)(input_layer)
lstm_layer_2 = layers.LSTM(32)(lstm_layer_1)
dense_layer = layers.Dense(16, activation='relu')(lstm_layer_2)

# Salida para predecir el ángulo
angle_output = layers.Dense(1, name='angle_output')(dense_layer)

# Salida para predecir la energía
energy_output = layers.Dense(1, activation='sigmoid', name='energy_output')(dense_layer)

# Salida para predecir el tipo de partícula
particle_output = layers.Dense(1, activation='sigmoid', name='particle_output')(dense_layer)

# Definir el modelo con tres salidas
model = tf.keras.Model(inputs=input_layer, outputs=[angle_output, energy_output, particle_output])

# Compilar el modelo
model.compile(optimizer='adam', 
              loss={'angle_output': 'mse', 'energy_output': 'binary_crossentropy', 'particle_output': 'binary_crossentropy'}, 
              metrics={'angle_output': 'mae', 'energy_output': 'accuracy', 'particle_output': 'accuracy'})

# Entrenar el modelo
history = model.fit(
    train_sequences, {'angle_output': train_angles, 'energy_output': train_energy_labels, 'particle_output': train_particle_labels},
    validation_data=(val_sequences, {'angle_output': val_angles, 'energy_output': val_energy_labels, 'particle_output': val_particle_labels}),
    epochs=epochs,
    batch_size=batch_size
)

# Guardar el modelo
model.save(model_save_path)
print(f"Modelo guardado en {model_save_path}")

# Evaluar el modelo
eval_results = model.evaluate(val_sequences, {'angle_output': val_angles, 'energy_output': val_energy_labels, 'particle_output': val_particle_labels}, verbose=1)
print(f"Resultados de evaluación: {eval_results}")

# Limpiar memoria
tf.keras.backend.clear_session()
gc.collect()