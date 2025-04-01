import panama as pn
import numpy as np
import pandas as pd
from pathlib import Path
import os
import gc  # Garbage collector interface

# Directorio base para fotones con el modelo Epos Urqmd
base_directory = r"C:\Users\Froxo\OneDrive - Universidad Técnica Federico Santa María\8E1_Showers_PhotonCR_2"

# Directorio de salida para fotones
output_directory = r"C:\Users\Froxo\OneDrive - Universidad Técnica Federico Santa María\CONDOR_Angle_2\CONDOR_8E1_Photon_ChPt"

# Crear las carpetas base de salida si no existen
os.makedirs(output_directory, exist_ok=True)
binned_data_dir = os.path.join(output_directory, 'binned_data')
particle_data_dir = os.path.join(output_directory, 'particle_data')
os.makedirs(binned_data_dir, exist_ok=True)
os.makedirs(particle_data_dir, exist_ok=True)

# Energías y ángulos
energies = [8e1]
angles = list(range(0, 62, 2))  # 0° to 60°, step of 2

# Procesar archivos para fotones
dat_files = sorted(Path(base_directory).glob("DAT*"))
particle_id = 1  # 1 para fotones, 14 para protones, 202 para He4, 1608 para O, 1206 para C, 5626 para Fe

# Mapear ángulos basado en grupos de 1000 archivos
for i, dat_file in enumerate(dat_files):
    try:
        angle_index = i // 1000  # Determinar el grupo de ángulos
        angle = angles[angle_index]  # Obtener el ángulo correspondiente
        seed_number = i % 1000 + 1  # Número de semilla dentro del grupo

        # Leer archivo DAT
        run_header, event_header, particles = pn.read_DAT(glob=str(dat_file), mother_columns=False)

        # Filtrar y procesar partículas
        particles['pdgid'] = particles['pdgid'].astype(int)
        particles_df = particles.query(
            "pdgid in (2212, -2212, 11, -11, 13, -13, 211, -211, 321, -321, 311, 3222, 3112, 3312, 3334, -3222, -3112, -3312, -3334)"
        )

        x = particles_df['x'] / 100
        y = particles_df['y'] / 100
        t = particles_df['t']
        En = particles_df['energy']

        # Normalizar el tiempo
        t -= t.min()

        # Preparar DataFrame de salida
        output_df = pd.DataFrame({
            'x': x,
            'y': y,
            't': t,
            'E': En
        })

        total_particles = len(output_df)

        # Filtrar por área de CONDOR
        condor_df = output_df[(output_df['x'] >= -61) & (output_df['x'] <= 61) & 
                              (output_df['y'] >= -56.5) & (output_df['y'] <= 56.5)].reset_index(drop=True)
        
        condor_df['t'] = condor_df['t'] - condor_df['t'].min()
        condor_particles = len(condor_df)
        condor_percentage = int((condor_particles / total_particles) * 100)

        # Calcular área de la shower
        x_min, x_max = condor_df['x'].min(), condor_df['x'].max()
        y_min, y_max = condor_df['y'].min(), condor_df['y'].max()
        area_shower = f"{int(x_max - x_min)} × {int(y_max - y_min)}"

        # Agregar metadatos
        condor_df['particle_id'] = particle_id
        condor_df['incidence_energy'] = energies[0]
        condor_df['incidence_angle'] = angle
        condor_df['area_shower_m2'] = area_shower
        condor_df['percentage_in_condor'] = condor_percentage
        condor_df['seed'] = seed_number
        condor_df['run'] = angle_index + 1

        new_column_order = ['x', 'y', 't', 'E', 'particle_id', 'incidence_energy', 'incidence_angle', 'area_shower_m2', 'percentage_in_condor', 'seed', 'run']
        condor_df = condor_df[new_column_order]

        # Ordenar por tiempo de llegada
        condor_df = condor_df.sort_values(by='t').reset_index(drop=True)

        # Crear subdirectorios para cada run
        run_data_directory = os.path.join(particle_data_dir, f"run_{angle_index + 1}")
        run_binned_directory = os.path.join(binned_data_dir, f"run_{angle_index + 1}")
        os.makedirs(run_data_directory, exist_ok=True)
        os.makedirs(run_binned_directory, exist_ok=True)

        # Guardar datos de partículas
        output_filename = f"data_particle_{particle_id}_energy_{energies[0]:.1E}_angle_{angle}_run_{angle_index + 1}_seed_{seed_number}.csv"
        output_filepath = os.path.join(run_data_directory, output_filename)
        condor_df.to_csv(output_filepath, index=False, header=True)

        # Crear bins
        condor_df['x_bin'] = np.floor(condor_df['x']).astype(int)
        condor_df['y_bin'] = np.floor(condor_df['y']).astype(int)
        time_bin_size = 1
        condor_df['t_bin'] = np.floor(condor_df['t'] / time_bin_size).astype(int)

        # Agrupar y guardar datos binneados
        binned_particles = condor_df.groupby(['x_bin', 'y_bin', 't_bin']).size().reset_index(name='particle_count')
        binned_output_filename = f"binned_data_particleid_{particle_id}_energy_{energies[0]:.1E}_angle_{angle}_run_{angle_index + 1}_seed_{seed_number}.csv"
        binned_output_filepath = os.path.join(run_binned_directory, binned_output_filename)
        binned_particles.to_csv(binned_output_filepath, index=False)

    except Exception as e:
        pass
