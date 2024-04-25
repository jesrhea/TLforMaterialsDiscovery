import os, subprocess, csv
import pymatgen.core as mg

def submit_zeopp(st_fn, job, out_type):
   # zeopp ='/rds/general/user/jkd19/home/working_dir/zeo++-0.3/network'
    #st_dir = '/rds/general/user/jkd19/home/working_dir/git_copy/cofs/npj-computmater-thermal-conductivity-dataset/git_mof_dat/regnerated_mof_cifs/'
    p = subprocess.Popen([
        f'{zeopp} -ha -{job}  {st_fn}'],
        shell=True,
    )
    p.wait()
    # Format of expected outputs 
    outs = st_fn[:-3]+out_type
    print('output = ', outs)
    return(outs)

def data_to_dict(outs):
    with open(outs, 'r') as f:
        data = f.read()
    data=data.replace('\n',' ')
    out_data ={}
    for i, x in enumerate(data.split(' ')):
        if ':' in x:
            out_data[x.split(':')[0]] = data.split(' ')[i+1]
    return(out_data)
           
def pore_diams(st_fn):
    pore_outs= submit_zeopp(st_fn, 'resex', 'res')
    try:
        with open(pore_outs, 'r') as f:
            data = f.read().split('\n')[0].split(' ')
            lcd = [data[4], data[9], data[15]]
            gcd = [data[5], data[11], data[17]]
            pld = [data[7], data[13], data[19]]
    except FileNotFoundError:
        lcd = ['', '', '']
        gcd = ['', '', '']
        pld = ['', '', '']
    return(lcd,gcd,pld)
    
def sa(st_fn):
    sa_outs= submit_zeopp(st_fn, 'sa 1.0 1.0 2000', 'sa')
    try:
        out_data = data_to_dict(sa_outs)
    except FileNotFoundError:
        unit_vol = ''
        acc_sa_perg=''
    else:
        unit_vol = out_data['Unitcell_volume']
        acc_sa_perg = out_data['ASA_m^2/g']

    return(unit_vol, acc_sa_perg)
        
def vol(st_fn):
    v_outs = submit_zeopp(st_fn, 'vol 1.2 1.2 50000', 'vol')

    try:
        out_data = data_to_dict(v_outs)
        acc_vol = out_data['AV_A^3']
        acc_vol_fr = out_data['AV_Volume_fraction']
    except FileNotFoundError:
        acc_vol=''
        acc_vol_fr=''
    return(acc_vol, acc_vol_fr)
        
def create_primitive(cif_file):
    st1 = mg.Structure.from_file(cif_file)
    st1_primitive = st1.to_primitive()
    st1_primitive.to_file(f'primitives/_primitive_{os.path.basename(cif_file)}','cif')
    return(f'primitives/_primitive_{cif_file}')




    
