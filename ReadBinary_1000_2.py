import panama as pn
import numpy as np
import pandas as pd
from pathlib import Path
import os
import gc  # Garbage collector interface

# Directorios base para fotones y protones
base_directories = {
    "photon": r"C:\Users\Froxo\OneDrive - Universidad Técnica Federico Santa María\1-5E2_1000Showers_PhotonCR_2",
    "proton": r"C:\Users\Froxo\OneDrive - Universidad Técnica Federico Santa María\1-5E2_1000Showers_ProtonCR_2"
}

# Directorios de salida separados para cada tipo de partícula
output_directories = {
    "photon": r"C:\Users\Froxo\OneDrive - Universidad Técnica Federico Santa María\CONDOR_1-5E2_1000_Photon_2",
    "proton": r"C:\Users\Froxo\OneDrive - Universidad Técnica Federico Santa María\CONDOR_1-5E2_1000_Proton_2"
}

# Crear las carpetas base de salida si no existen
for particle_type, output_directory in output_directories.items():
    os.makedirs(output_directory, exist_ok=True)
    os.makedirs(os.path.join(output_directory, "binned_data"), exist_ok=True)
    os.makedirs(os.path.join(output_directory, "particles_data"), exist_ok=True)

# Energías y ángulos
energies = [1.5e2]
angles = list(range(0, 62, 2))  # 0° to 60°, step of 5

# Procesar archivos para cada tipo de partícula
for particle_type, input_directory in base_directories.items():
    dat_files = sorted(Path(input_directory).glob("DAT*"))
    particle_id = 1 if particle_type == "photon" else 14  # 1 para fotones, 14 para protones

    # Directorios de salida específicos
    output_directory = output_directories[particle_type]
    binned_directory = os.path.join(output_directory, "binned_data")
    data_directory = os.path.join(output_directory, "particles_data")

    # Mapear ángulos basado en grupos de 1000 archivos
    for i, dat_file in enumerate(dat_files):
        try:
            angle_index = i // 1000  # Determinar el grupo de ángulos
            angle = angles[angle_index]  # Obtener el ángulo correspondiente
            seed_number = i % 1000 + 1  # Número de semilla dentro del grupo

            # Leer archivo DAT
            run_header, event_header, particles = pn.read_DAT(glob=str(dat_file), mother_columns=False)
            print(f"Processing {dat_file} (Angle {angle}°, Seed {seed_number})")

            # Filtrar y procesar partículas
            particles['pdgid'] = particles['pdgid'].astype(int)
            particles_df = particles.query(
                "pdgid in (22, 2212, 2112, 11, -11, 13, -13, 211, -211, 111, 321, -321, 311, 3122, 3222, 3112, 3322, 3312, 3334)"
            )

            x = particles_df['x'] / 100
            y = particles_df['y'] / 100
            t = particles_df['t']
            E = particles_df['energy']

            # Normalizar el tiempo
            t -= t.min()

            # Preparar DataFrame de salida
            output_df = pd.DataFrame({
                'x': x,
                'y': y,
                't': t,
                'E': E
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
            run_data_directory = os.path.join(data_directory, f"run_{angle_index + 1}")
            run_binned_directory = os.path.join(binned_directory, f"run_{angle_index + 1}")
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

            print(f"Saved output to {output_filepath}")

        except Exception as e:
            print(f"Error processing {dat_file}: {e}")
