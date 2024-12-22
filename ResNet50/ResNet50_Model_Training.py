import os
import gc
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.preprocessing.image import img_to_array, load_img
import pandas as pd
import numpy as np

# Configurar el allocador para reducir fragmentación de memoria
os.environ['TF_GPU_ALLOCATOR'] = 'cuda_malloc_async'

# Habilitar precisión mixta para reducir el uso de memoria
tf.keras.mixed_precision.set_global_policy('mixed_float16')

# Directorios
base_dir = r'C:\Users\Froxo\OneDrive - Universidad Técnica Federico Santa María\ML_Predictions'
images_dir = os.path.join(base_dir, 'Images')
predicts_dir = os.path.join(base_dir, 'Predicts_All')

# Crear carpeta para predicciones si no existe
os.makedirs(predicts_dir, exist_ok=True)

# Parámetros del modelo
input_shape = (224, 224, 3)
epochs = 20
batch_size = 1
max_images_per_folder = 200  # Número máximo de imágenes a cargar por carpeta

# Función para cargar imágenes y ángulos desde el directorio de cada run
def load_images_and_angles(run_path, train_ratio=0.7):
    images = []
    angles = []
    
    # Filtrar archivos PNG en el directorio
    image_files = [f for f in os.listdir(run_path) if f.endswith('.png')]
    
    # Seleccionar aleatoriamente hasta max_images_per_folder imágenes
    selected_files = np.random.choice(image_files, size=min(max_images_per_folder, len(image_files)), replace=False)
    
    for filename in selected_files:
        # Cargar imagen
        img_path = os.path.join(run_path, filename)
        img = load_img(img_path, target_size=(224, 224))
        img_array = img_to_array(img) / 255.0  # Normalizar
        images.append(img_array)
        
        # Extraer ángulo del nombre del archivo
        angle = float(filename.split('_')[7])  # Ajusta el índice si el ángulo está en otra posición
        angles.append(angle)
    
    # Convertir a arrays numpy para manipulación fácil
    images = np.array(images)
    angles = np.array(angles)

    # Mezclar aleatoriamente los datos de la carpeta
    indices = np.arange(len(images))
    np.random.shuffle(indices)
    images = images[indices]
    angles = angles[indices]

    # Dividir en entrenamiento y validación según el porcentaje especificado
    split_index = int(len(images) * train_ratio)
    train_images = images[:split_index]
    val_images = images[split_index:]
    train_angles = angles[:split_index]
    val_angles = angles[split_index:]

    return train_images, train_angles, val_images, val_angles

# Inicializar listas para almacenar todos los datos de entrenamiento y validación
all_train_images = []
all_train_angles = []
all_val_images = []
all_val_angles = []

# Iterar solo sobre las carpetas "Images_run_1" a "Images_run_14"
for i in range(1, 14):
    run_folder = f'Images_run_{i}'
    run_path = os.path.join(images_dir, run_folder)
    
    if os.path.isdir(run_path):
        print(f"Procesando {run_folder}...")

        # Cargar y dividir los datos de cada carpeta
        train_images, train_angles, val_images, val_angles = load_images_and_angles(run_path)
        
        # Acumular datos de entrenamiento y validación
        all_train_images.append(train_images)
        all_train_angles.append(train_angles)
        all_val_images.append(val_images)
        all_val_angles.append(val_angles)

# Concatenar todos los datos de entrenamiento y validación de todas las carpetas
all_train_images = np.concatenate(all_train_images, axis=0)
all_train_angles = np.concatenate(all_train_angles, axis=0)
all_val_images = np.concatenate(all_val_images, axis=0)
all_val_angles = np.concatenate(all_val_angles, axis=0)

# Definir el modelo
base_model = tf.keras.applications.ResNet50(
    input_shape=input_shape, 
    include_top=False, 
    weights='imagenet'
)
base_model.trainable = False

model = models.Sequential([
    base_model,
    layers.GlobalAveragePooling2D(),
    layers.Dense(64, activation='relu'),
    layers.Dense(1, dtype='float32')  # Salida de regresión
])

model.compile(optimizer='adam', loss='mse', metrics=['mae'])

# Entrenar el modelo
history = model.fit(
    all_train_images, all_train_angles,
    validation_data=(all_val_images, all_val_angles),
    epochs=epochs,
    batch_size=batch_size
)

# Evaluar el modelo y guardar las estadísticas
eval_results = model.evaluate(all_val_images, all_val_angles, verbose=1)
predictions = model.predict(all_val_images)

mae = np.mean(np.abs(all_val_angles - predictions.flatten()))
mse = np.mean((all_val_angles - predictions.flatten()) ** 2)

# Guardar estadísticas en CSV
stats_path = os.path.join(predicts_dir, 'ResNet50_stats_proton.csv')
stats_df = pd.DataFrame({
    'True Angle': all_val_angles,
    'Predicted Angle': predictions.flatten()
})
stats_df['Absolute Error'] = np.abs(stats_df['True Angle'] - stats_df['Predicted Angle'])
stats_df.to_csv(stats_path, index=False)

print(f"Estadísticas combinadas guardadas en {stats_path}")
print(f"MAE: {mae}, MSE: {mse}")

# Limpiar memoria después de todo el proceso
tf.keras.backend.clear_session()
gc.collect()
