import os
import numpy as np
import pandas as pd
import time
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, Conv2D, Dropout, MaxPooling2D
from tensorflow.keras.utils import to_categorical

# Función para cargar y procesar datos
def load_and_process_data(proton_dir, photon_dir):
    def load_data_from_dir(directory, label):
        data = []
        for run_folder in os.listdir(directory):
            run_path = os.path.join(directory, run_folder)
            if os.path.isdir(run_path):
                for file in os.listdir(run_path):
                    file_path = os.path.join(run_path, file)
                    if file.endswith(".csv"):
                        df = pd.read_csv(file_path)
                        # Convertir a matriz (rellenar con ceros si es necesario)
                        matrix = df.pivot_table(index='y_bin', columns='x_bin', values='particle_count', aggfunc='sum').fillna(0).to_numpy()
                        data.append((matrix, label))
        return data
    
    print("-> Cargando datos de protones...")
    start_time = time.time()
    proton_data = load_data_from_dir(proton_dir, 0)  # 0 para protones
    print(f"   Datos de protones cargados en {time.time() - start_time:.2f} segundos.")
    
    print("-> Cargando datos de fotones...")
    start_time = time.time()
    photon_data = load_data_from_dir(photon_dir, 1)  # 1 para fotones
    print(f"   Datos de fotones cargados en {time.time() - start_time:.2f} segundos.")
    
    all_data = proton_data + photon_data
    
    # Encontrar las dimensiones máximas
    max_rows = max([matrix.shape[0] for matrix, _ in all_data])
    max_cols = max([matrix.shape[1] for matrix, _ in all_data])
    
    # Rellenar matrices a dimensiones máximas
    print("-> Normalizando dimensiones con relleno...")
    padded_data = []
    for matrix, label in all_data:
        padded_matrix = np.zeros((max_rows, max_cols))
        padded_matrix[:matrix.shape[0], :matrix.shape[1]] = matrix
        padded_data.append((padded_matrix, label))
    
    # Separar características y etiquetas
    X = np.array([item[0] for item in padded_data])  # Características
    y = np.array([item[1] for item in padded_data])  # Etiquetas
    return X, y

# Directorios de datos
proton_dir = r"C:\Users\Froxo\OneDrive - Universidad Técnica Federico Santa María\CONDOR_8E2_1000_Photon\binned_data"
photon_dir = r"C:\Users\Froxo\OneDrive - Universidad Técnica Federico Santa María\CONDOR_8E2_1000_Proton\binned_data"

# Cargar datos
print("==> Inicio del proceso de datos.")
X, y = load_and_process_data(proton_dir, photon_dir)
print(f"-> Dimensiones de X: {X.shape}")
print(f"-> Dimensiones de y: {y.shape}")

# Dividir en conjunto de entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Preparar etiquetas para clasificación categórica
y_train_cat = to_categorical(y_train, num_classes=2)
y_test_cat = to_categorical(y_test, num_classes=2)

# Crear el modelo
print("==> Construyendo el modelo...")
model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(X_train.shape[1], X_train.shape[2], 1)),
    MaxPooling2D((2, 2)),
    Dropout(0.25),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    Dropout(0.25),
    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.5),
    Dense(2, activation='softmax')  # Dos clases: protón y fotón
])

# Compilar modelo
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
print(model.summary())

# Entrenar el modelo
print("==> Entrenando el modelo...")
history = model.fit(X_train[..., np.newaxis], y_train_cat, epochs=10, batch_size=32, validation_split=0.2)

# Evaluar el modelo
print("==> Evaluando el modelo...")
test_loss, test_acc = model.evaluate(X_test[..., np.newaxis], y_test_cat)
print(f"-> Precisión en conjunto de prueba: {test_acc:.4f}")

# Realizar predicciones
print("==> Realizando predicciones...")
y_pred_prob = model.predict(X_test[..., np.newaxis])  # Probabilidades de salida
y_pred = np.argmax(y_pred_prob, axis=1)  # Clases predichas

# Guardar estadísticas
output_dir = r"C:\Users\Froxo\OneDrive - Universidad Técnica Federico Santa María\ML_Predictions\Particle_Prediction_WG"
os.makedirs(output_dir, exist_ok=True)

# Guardar estadísticas de entrenamiento
stats_path = os.path.join(output_dir, "prediction_stats.csv")
pd.DataFrame({
    "Epoch": list(range(1, len(history.history['accuracy']) + 1)),
    "Train_Accuracy": history.history['accuracy'],
    "Validation_Accuracy": history.history['val_accuracy']
}).to_csv(stats_path, index=False)
print(f"-> Estadísticas guardadas en: {stats_path}")

# Guardar predicciones reales vs predichas
predictions_path = os.path.join(output_dir, "real_vs_predicted.csv")
pd.DataFrame({
    "Real_Particle": y_test,
    "Predicted_Particle": y_pred
}).to_csv(predictions_path, index=False)
print(f"-> Predicciones guardadas en: {predictions_path}")



