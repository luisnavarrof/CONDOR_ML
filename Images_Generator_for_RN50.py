import os
import re
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Ruta base para los directorios
input_base_dir = r"C:\Users\Froxo\OneDrive - Universidad Técnica Federico Santa María\CONDOR_8E2Energy_1000_output_files\binned_data"
output_base_dir = r"C:\Users\Froxo\OneDrive - Universidad Técnica Federico Santa María\ML_Predictions\Images"

# Función para extraer el número de cada carpeta "run_x" para ordenar correctamente
def get_run_number(folder_name):
    match = re.search(r'run_(\d+)', folder_name)
    return int(match.group(1)) if match else None

# Listar carpetas en el directorio de entrada y ordenarlas numéricamente
run_folders = sorted([f for f in os.listdir(input_base_dir) if os.path.isdir(os.path.join(input_base_dir, f))], key=get_run_number)

# Iterar sobre cada carpeta 'run_x' ordenada
for run_folder in run_folders:
    run_path = os.path.join(input_base_dir, run_folder)
    
    # Crear el directorio de salida para cada run
    images_output_dir = os.path.join(output_base_dir, f"Images_{run_folder}")
    os.makedirs(images_output_dir, exist_ok=True)
    
    # Procesar cada archivo CSV en la carpeta actual
    csv_files = [f for f in os.listdir(run_path) if f.endswith(".csv")]
    if not csv_files:
        print(f"No se encontraron archivos CSV en {run_folder}")
        continue  # Saltar a la siguiente carpeta si no hay archivos CSV

    for csv_file in csv_files:
        csv_path = os.path.join(run_path, csv_file)
        
        # Cargar el archivo CSV
        binned_data = pd.read_csv(csv_path)
        
        # Crear tabla pivote para el heatmap
        heatmap_data = binned_data.pivot_table(index='y_bin', columns='x_bin', values='particle_count', fill_value=0)
        
        # Graficar el mapa de calor sin títulos, etiquetas ni barra de colores
        plt.figure(figsize=(8, 6))
        sns.heatmap(heatmap_data, cmap='mako', cbar=False)
        
        # Eliminar títulos, etiquetas y ejes
        plt.axis('off')
        
        # Guardar la imagen en la carpeta de salida
        output_image_path = os.path.join(images_output_dir, f"{csv_file.replace('.csv', '')}.png")
        plt.savefig(output_image_path, bbox_inches='tight', pad_inches=0)
        
        # Cerrar la figura para liberar memoria
        plt.close()