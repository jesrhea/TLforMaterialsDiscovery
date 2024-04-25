import pandas as pd
import pymatgen.core as mg
import numpy as np 
import os, shutil, subprocess, csv
from rdkit.Chem import rdDetermineBonds
from rdkit import Chem

# Dictionary of elements and their atomic masses (in atomic mass units, amu)
atomic_masses = {
    'H': 1.0079,
    'He': 4.0026,
    'Li': 6.941,
    'Be': 9.0122,
    'B': 10.81,
    'C': 12.011,
    'N': 14.007,
    'O': 15.999,
    'F': 18.998,
    'Ne': 20.180,
    'Na': 22.990,
    'Mg': 24.305,
    'Al': 26.982,
    'Si': 28.085,
    'P': 30.974,
    'S': 32.06,
    'Cl': 35.45,
    'K': 39.098,
    'Ar': 39.948,
    'Ca': 40.08,
    'Sc': 44.956,
    'Ti': 47.867,
    'V': 50.942,
    'Cr': 51.996,
    'Mn': 54.938,
    'Fe': 55.845,
    'Ni': 58.693,
    'Co': 58.933,
    'Cu': 63.546,
    'Zn': 65.38,
    'Ga': 69.723,
    'Ge': 72.630,
    'As': 74.922,
    'Se': 78.971,
    'Br': 79.904,
    'Kr': 83.798,
    'Rb': 85.468,
    'Sr': 87.62,
    'Y': 88.906,
    'Zr': 91.224,
    'Nb': 92.906,
    'Mo': 95.95,
    'Tc': 98.000,
    'Ru': 101.1,
    'Rh': 102.9,
    'Pd': 106.4,
    'Ag': 107.9,
    'Cd': 112.4,
    'In': 114.8,
    'Sn': 118.7,
    'Sb': 121.8,
    'Te': 127.6,
    'I': 126.9,
    'Xe': 131.3,
    'Cs': 132.9
}


import multiprocessing.pool
import functools

def timeout(max_timeout):
    """Timeout decorator, parameter in seconds."""
    def timeout_decorator(item):
        """Wrap the original function."""
        @functools.wraps(item)
        def func_wrapper(*args, **kwargs):
            """Closure for function."""
            pool = multiprocessing.pool.ThreadPool(processes=1)
            async_result = pool.apply_async(item, args, kwargs)
            # raises a TimeoutError if execution exceeds max_timeout
            return async_result.get(max_timeout)
        return func_wrapper
    return timeout_decorator

## this read_xyz function is used for all mass calculations
def read_xyz(filename):
    with open(filename, 'r') as f:
        lines = f.readlines()

    num_atoms = int(lines[0])
    atomic_symbols = []
    coordinates = []

    for line in lines[2:]:
        parts = line.split()
        atomic_symbols.append(parts[0])
        coordinates.append([float(parts[1]), float(parts[2]), float(parts[3])])

    return num_atoms, atomic_symbols, coordinates

def calculate_molecular_mass(filename):
    num_atoms, atomic_symbols, _ = read_xyz(filename)
    total_mass = 0.0

    for symbol in atomic_symbols:
        if symbol in atomic_masses:
            total_mass += atomic_masses[symbol]
        else:
            print(f"Warning: Atomic symbol '{symbol}' not found in the atomic_masses dictionary.")

    return total_mass

def find_max_length(xyz_fn):
    
    with open(xyz_fn,'r') as f:
        lines = f.readlines()

    # Assuming the first line contains the number of atoms (you may need to adapt this depending on your specific .xyz file format)
    num_atoms = int(lines[0])

    coordinates = []
    for line in lines[2:]:  # Skip the first two lines (comment and number of atoms)
        parts = line.split()
        symbol, x, y, z = parts[0], float(parts[1]), float(parts[2]), float(parts[3])
        coordinates.append((symbol, x, y, z))
    max_distance = 0.0
    a_coords=[]
    b_coords=[]
    for i in range(num_atoms):
        for j in range(i + 1, num_atoms):
            x1, y1, z1 = coordinates[i][1], coordinates[i][2], coordinates[i][3]
            x2, y2, z2 = coordinates[j][1], coordinates[j][2], coordinates[j][3]

            distance = ((x2 - x1) ** 2 + (y2 - y1) ** 2 + (z2 - z1) ** 2) ** 0.5

            if distance > max_distance:
                max_distance = distance

    return(max_distance)

# Function to read the coordinates from an .xyz file
def read_xyz(filename):
    with open(filename, 'r') as f:
        lines = f.readlines()
    
    num_atoms = int(lines[0])  # Line 1: Number of atoms
    atomic_symbols = []
    coordinates = []

    for line in lines[2:]:
        parts = line.split()
        atomic_symbols.append(parts[0])  # Atom symbol (e.g., 'C', 'H')
        coordinates.append([float(parts[1]), float(parts[2]), float(parts[3])])

    return num_atoms, atomic_symbols, np.array(coordinates)

# Function to check if a group of atoms forms a phenylene unit
def is_phenylene(coordinates, bond_length_threshold=1.4, angle_threshold=120):
    for i in range(len(coordinates)):
        if i < 6:
            continue
        for j in range(i - 1, i - 6, -1):
            if np.linalg.norm(coordinates[i] - coordinates[j]) > bond_length_threshold:
                return False

        angles = []
        for k in range(i - 5, i + 1):
            vec1 = coordinates[k - 1] - coordinates[k]
            vec2 = coordinates[k + 1] - coordinates[k]
            cos_theta = np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))
            angle = np.degrees(np.arccos(cos_theta))
            angles.append(angle)
        
        if all(angle > angle_threshold for angle in angles):
            return True

    return False

# Function to count phenylene units in the .xyz file
def count_phenylene_units(filename):
    num_atoms, atomic_symbols, coordinates = read_xyz(filename)
    num_phenylene_units = 0

    for i in range(num_atoms):
        if atomic_symbols[i] == 'C' and is_phenylene(coordinates[i:]):
            num_phenylene_units += 1

    return num_phenylene_units

def calculate_mass_of_first_metal_atom(filename):
    num_atoms, atomic_symbols, _ = read_xyz(filename)

    for i in range(num_atoms):
        symbol = atomic_symbols[i]
        if symbol in atomic_masses:
            if symbol in ['Na', 'Mg', 'Al', 'Si', 'K', 'Ca', 'Sc', 'Ti', 'V', 'Cr', 'Mn', 'Fe', 'Ni', 'Co', 'Cu', 'Zn']:
                # Check if the atom is a metal atom (add more metal symbols as needed)
                mass = atomic_masses[symbol]
                return mass
        else:
            print(f"Warning: Atomic symbol '{symbol}' not found in the atomic_masses dictionary.")

    return None  # If no metal atom is found

@timeout(10.0)
def generate_smiles(xyz_file):
    raw_mol = Chem.MolFromXYZFile(xyz_file)
    mol = Chem.Mol(raw_mol)
    try:
        rdDetermineBonds.DetermineBondOrders(raw_mol, charge=0)
    except ValueError as e:
        print(e)
        return('')
    else:
        return(Chem.CanonSmiles(Chem.MolToSmiles(mol)))
    
def move_outs_to_storage(data:list, name, storage):
    for x in data:
        shutil.move(x, os.path.join(storage, os.path.basename(x)[:-4]+'_'+name+'.xyz' ))
    
def main():    
    st_data_dir = '/rds/general/user/jkd19/home/working_dir/git_copy/cofs/npj-computmater-thermal-conductivity-dataset/git_mof_dat/regnerated_mof_cifs'
    run_dir = '/rds/general/user/jkd19/home/working_dir/mbud/mBUD'
    storage = os.path.join(run_dir, 'output_data')
    
    headings = [
        'name',
        'node_connectivity',
        'linker_length_[A]',
        'node_length_[A]',
        'number_of_phenylenes_per_linker',
        'linker_mass_[amu]',
        'node_mass_[amu]',
        'node_metal_mass_[amu]'
        #'linker_structure',
        #'node_structure'
    ]

    csv_name = 'mbud_features2.csv'

    # Read in thermal conductivity  data as pandas
    therm_data_fn = '/rds/general/user/jkd19/home/working_dir/git_copy/cofs/npj-computmater-thermal-conductivity-dataset/name_kxyz_9593MOFs.csv'
    therm_df = pd.read_csv(therm_data_fn)
    
    for i, fn in enumerate(therm_df["name"]):
        
        if i == 0:
            with open(csv_name,'a') as f:
                csv_writer = csv.writer(f)
                csv_writer.writerow(headings)
        if i > 213:
            ab_fn = os.path.join(st_data_dir,fn+'.cif')
            print('Reading file using pymatgen: ')
            print(ab_fn)
            st1 = mg.Structure.from_file(ab_fn)
            print('converting to primitive ')
            st1_primitive = st1.to_primitive()
            print('writing structure to mBUD folder')
            st1_primitive.to_file(os.path.join(run_dir, f'_primitive{fn}.cif'),'cif')

            # move file to mbud run_dir
           # print('copying to run dir')
            #shutil.copy(f'st{i}_primitive.cif', run_dir)

            print('running mbud job')
            p = subprocess.Popen([
                f'python3 {run_dir}/mBUD.py --ciffile _primitive{fn}.cif --outdir .'],
                cwd=run_dir,
                shell=True)
            p.wait()

            # calculate connectivity
            print('calculating connectivity... ')
            comp_out_node = os.path.join(run_dir, 'comp-node-0.xyz')
            try:
                with open(comp_out_node,'r') as f:
                    noof_ars = [1 for line in f.read().split('\n') if line[:2]=='Ar']
                    noof_ars = len(noof_ars)
            except FileNotFoundError:
                comp_out_node = os.path.join(run_dir, 'comp-node-1.xyz')
                with open(comp_out_node,'r') as f:
                    noof_ars = [1 for line in f.read().split('\n') if line[:2]=='Ar']
                    noof_ars = len(noof_ars)
            connectivity = noof_ars

            # linker length
            print('calculating linker length... ')
            out_linker = os.path.join(run_dir,'linker-0.xyz')
            try:
                linker_length = find_max_length(out_linker)
            except FileNotFoundError:
                out_linker = os.path.join(run_dir,'linker-1.xyz')
                linker_length = find_max_length(out_linker)

            # node length
            print('calculating node length... ')
            nc_node = os.path.join(run_dir,'node-0.xyz')
            try:
                node_length = find_max_length(nc_node)
            except FileNotFoundError:
                nc_node = os.path.join(run_dir,'node-1.xyz')
                node_length = find_max_length(nc_node)

            # noof_phenyls 
            print('calculating noof phenyls... ')
            linker_phenyls = count_phenylene_units(out_linker)

    # NEED TO RUN CODE WITH ~10 STRUCTURES AND SEE IF THIS PHENYL COUNT CODE WORKS!!!

            # linker mass
            print('calculating linker mass... ')
            l_mass=calculate_molecular_mass(out_linker)
            l_mass=round(l_mass,3)

            # node mass
            print('calculating node mass... ')
            n_mass = calculate_molecular_mass(nc_node)
            n_mass = round(n_mass, 3)

            # metal-ion node mass
            print('calculating metal ion mass... ')
            n_metal_mass=calculate_mass_of_first_metal_atom(nc_node)

            # node structure
          #  print('calculating node smiles.. ')
          #  node_st = generate_smiles(nc_node)

            # linker structure
           # print('calculating linker smiles.. ')
           # link_st = generate_smiles(out_linker)

            with open(csv_name, 'a') as f:
                csv_writer=csv.writer(f)
                csv_writer.writerow([
                    fn,
                    connectivity,
                    linker_length,
                    node_length,
                    linker_phenyls,
                    l_mass,
                    n_mass,
                    n_metal_mass
                   # link_st,
                   # node_st
                ])

            move_outs_to_storage([comp_out_node, out_linker,nc_node], fn, storage)
            
        
if __name__=="__main__":
    main()
    # out_dir needs to be absolute path 