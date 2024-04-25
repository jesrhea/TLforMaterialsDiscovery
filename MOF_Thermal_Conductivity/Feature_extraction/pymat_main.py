from pymat_functions import _read_st, number_of_vertices_edges, noof_c_h_n_o, is_3d
import pandas as pd
import multiprocessing as mp
import math
import csv 


csv_name = 'pymat_features2.csv'

def write_row(data):
    with open(csv_name,'a') as f:
        csv_writer = csv.writer(f)
        csv_writer.writerow(data)


def main1():

    headings = [
        'name',
        'topology',
        'cell_a_length_[A]',
        'cell_b_length_[A]',
        'cell_c_length_[A]',
        'alpha_[deg]',
        'beta_[deg]',
        'gamma_[deg]',
        'density',
        'volume',
        'number_of_vertices',
        'number_of_edges',
        'dimensionality',
        'number_of_carbons_per_primitive_cell',
        'number_of_hydrogens_per_primitive_cell',
        'number_of_nitrogens_per_primitive_cell',
        'number_of_oxygens_per_primitive_cell',
        'formula',

        'kx',
        'ky',
        'kz',
    ]

    therm_data_fn = '/rds/general/user/jkd19/home/working_dir/git_copy/cofs/npj-computmater-thermal-conductivity-dataset/name_kxyz_9593MOFs.csv'
    therm_df = pd.read_csv(therm_data_fn)
    with open(csv_name,'a') as f:
        csv_writer = csv.writer(f)
        csv_writer.writerow(headings)

# started on primtiive from 2700 structure 

    start = 0
    batch_num = 200
    range_ = int(10000/batch_num)
    start_from=3000
    start_range_=math.floor(start_from/batch_num)
    for z in range(start_range_, range_+1):
        start = z*batch_num
        end = z*batch_num+batch_num+1
        if z == 0:
            start = 20000
        print(f'Running structures: {start} =<  < {end}')
        names = [x for i, x in list(enumerate(therm_df['name']))if  start<= i < end]
        #names = [x for i, x in enumerate(therm_df['name'])if start <= i < 10]

        print('Reading in structures')
        with mp.Pool() as pool:
            sts = (pool.map(_read_st, names))
        
        print('Calculating vertices (takes the longest)')
        # this takes the longest to process 
        #with open('v_e.csv', 'a') as f:
        #    cw = csv.writer(f)

        with mp.Pool() as pool:
            ve = (pool.map(number_of_vertices_edges, sts))

        print('Calculating number of species... ')
        with mp.Pool() as pool:
            chno = (pool.map(noof_c_h_n_o, sts))

        print('seeing whether they are 3D')
        with mp.Pool() as pool:
            dim = (pool.map(is_3d, sts))

        print('writing csv file...')
        with open(csv_name,'a') as f:
            csv_writer = csv.writer(f)
            for i in range(len(names)):
                csv_writer.writerow([
                    names[i],
                    names[i].split('_')[1],
                    round(sts[i]._lattice.params_dict['a'],5),
                    round(sts[i]._lattice.params_dict['b'],5),
                    round(sts[i]._lattice.params_dict['c'],5),
                    round(sts[i]._lattice.params_dict['alpha'],5),
                    round(sts[i]._lattice.params_dict['beta'],5),
                    round(sts[i]._lattice.params_dict['gamma'],5),
                    round(sts[i].density,5),
                    round(sts[i].volume, 5),
                    ve[i][0],
                    ve[i][1],
                    dim[i],
                    chno[i][0],
                    chno[i][1],
                    chno[i][2],
                    chno[i][3],
                    sts[i].formula,

                    therm_df["kx"][i],
                    therm_df["ky"][i],
                    therm_df["kz"][i]     
                    ])
        


        
        
if __name__ == '__main__':
    print(__name__)
    main1()
        

   