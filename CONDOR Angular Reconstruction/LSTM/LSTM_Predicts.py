import os
import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler

# Directorios
input_base_dir = r"C:\Users\Froxo\OneDrive - Universidad Técnica Federico Santa María\CONDOR_Angle_2\CONDOR_1E2_1000_Proton_2\binned_data"
predicts_dir = r"C:\Users\Froxo\OneDrive - Universidad Técnica Federico Santa María\ML_Predictions\LSTM_Predicts_Proton_1-5E2_binned"
model_save_path = os.path.join(predicts_dir, 'lstm_model_1-5E2_Proton_100_epochs_50_ts_binned.h5')  # Ruta del modelo guardado

# Parámetros
time_steps = 50  # Ventana de tiempo

# Función para generar secuencias a partir de los datos
def create_sequences(data, time_steps=50, scaler=None):
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

# Cargar el modelo guardado
model = tf.keras.models.load_model(model_save_path)
print(f"Modelo cargado desde {model_save_path}")

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

# Procesar cada carpeta de run nuevamente para generar predicciones
all_sequences = []
all_angles = []
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

# Generar predicciones
predictions = model.predict(all_sequences)

# Guardar estadísticas en un archivo CSV
file_name = 'LSTM_Predicts_Test_Set.csv'
stats_path = os.path.join(predicts_dir, file_name)
stats_df = pd.DataFrame({
    'True Angle': all_angles,
    'Predicted Angle': predictions.flatten()
})
stats_df['Absolute Error'] = np.abs(stats_df['True Angle'] - stats_df['Predicted Angle'])
stats_df.to_csv(stats_path, index=False)

# Imprimir resultados
print(f"Estadísticas guardadas en {stats_path}")