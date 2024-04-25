import subprocess, os, csv, math
import multiprocessing as mp
import pandas as pd
from zeopp_functions import *

def main1():
    
    with open('zeopp_data_fe2.csv', 'a') as f:
        cw = csv.writer(f)
        cw.writerow((
            'name',
            'LCD_x_[A]',
            'LCD_y_[A]',
            'LCD_z_[A]',
            'GCD_x_[A]',
            'GCD_y_[A]',
            'GCD_z_[A]',
            'PLD_x_[A]',
            'PLD_y_[A]',
            'PLD_z_[A]',
            'unitcell_vol_[Aˆ3]',
            'ASA_[mˆ2/g]',
            'AV_A^3',
            'AV_Volume_fraction'
        ))
        
    therm_data_fn = '/rds/general/user/jkd19/home/working_dir/git_copy/cofs/npj-computmater-thermal-conductivity-dataset/name_kxyz_9593MOFs.csv'
    therm_df = pd.read_csv(therm_data_fn)
    st_dir = '/rds/general/user/jkd19/home/working_dir/git_copy/cofs/npj-computmater-thermal-conductivity-dataset/git_mof_dat/regnerated_mof_cifs'
    names = [os.path.join(st_dir, x+'.cif') for i, x in enumerate(therm_df['name']) if i>2470]


    start = 0
    batch_num = 100
    range_ = int(10000/batch_num)
    start_from=2477
    start_range_=int(math.floor(start_from/batch_num))
    for z in range(start_range_, range_+1):
        start = z*batch_num
        end = z*batch_num+batch_num+1
        if z == 0:
            start = 20000
        print(f'Running structures: {start} =<  < {end}')
        #names = [os.path.join(st_dir, x+'.cif') for i, x in list(enumerate(therm_df['name']))if start <= i < end]
        names = [ x+'.cif' for i, x in list(enumerate(therm_df['name']))if start <= i < end]
        # started primtiives from 2400 
        #with mp.Pool() as pool:
         #   prim_sts = (pool.map(create_primitive, names))
        prim_sts = [f'../primitives/_primitive_{n}' for n in names]

        with mp.Pool() as pool:
            _pore_diams = (pool.map(pore_diams, prim_sts))
            
        with mp.Pool() as pool:
             _sas = (pool.map(sa, prim_sts))
                
        with mp.Pool() as pool:
             _vols = (pool.map(vol, prim_sts))
            
        with open('zeopp_data_fe2.csv', 'a') as f:
            for i in range(len(names)):
                cw = csv.writer(f)
                cw.writerow((
                    os.path.basename(names[i]).split('.')[0],
                    _pore_diams[i][0][0],
                    _pore_diams[i][0][1],
                    _pore_diams[i][0][2],
                    _pore_diams[i][1][0],
                    _pore_diams[i][1][1],
                    _pore_diams[i][1][2],
                    _pore_diams[i][2][0],
                    _pore_diams[i][2][1],
                    _pore_diams[i][2][2],
                    _sas[i][0],
                    _sas[i][1],
                    _vols[i][0],
                    _vols[i][1]
                ))
        

        
if __name__ == '__main__':
    print(__name__)
    main1()
        