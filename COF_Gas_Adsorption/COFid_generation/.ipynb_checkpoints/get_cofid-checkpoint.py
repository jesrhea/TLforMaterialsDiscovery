import pandas as pd
from rdkit import Chem
import csv

CSV_NAME = 'lowP58_repsubset_cofids.csv'

linker_data = pd.read_csv('linker_data.csv')
pd.set_option('display.max_colwidth',1000)

def main():
    
    highP65_repsubset = pd.read_csv('../mercado_ch4_highP65_repsubset_10000-2.csv', names=['st_name','gas-ads'])
    lowP65_repsubset = pd.read_csv('../mercado_ch4_lowP58_repsubset_10000-2.csv', names=['st_name','gas-ads'])
    
    data = lowP65_repsubset
    
    with open(CSV_NAME,'w') as f:
        cw = csv.writer(f) 
        cw.writerow([
            'st_name', 'cofid','gas-ads'
        ])
    
    for i , name in enumerate(data['st_name']):
        print(name)
        try:
            cofid = string_stuff(name)
        except Exception as err:
            print(err.__str__)
            cofid=f'ERROR: {err,}'
        with open(CSV_NAME,'a') as f:
            cw = csv.writer(f) 
            cw.writerow([
                name,
                cofid,
                data['gas-ads'][i]
            ])

def string_stuff(example):
    
    linkers = []
    bondgroups = []
    top = []
    cat = []
    sbu = []
    cat.append(0)
    for i, x in enumerate(example.split('_')):
        if i==0 or i==2:
            linkers.append(int(x.split('linker')[-1]))
        if i==1 or i==3:
            bondgroups.append(x)
        if i==4:
            top.append(x)
        if i==7:
            cat= [int(x)-1]          

    link_st = []
    for link, bond in zip(linkers, bondgroups):
        if bond == 'CO':
            bond = 'C(=O)'
            #print(bond)
        if bond == 'C':
            bond = '[CD1]'
        if bond == 'N':
            bond = '[ND1]'
        if bond == 'NH':
            bond = '[NH1]'
        if bond == 'CH':
            bond = '[CH1]'
        if bond == 'CH2':
            bond = '[CH2]'
        link_smiles = linker_data.loc[linker_data['linker_number']==link]['smiles'].to_string(index=False)
        link_st.append(Chem.MolFromSmiles(link_smiles)) 
        sbu.append(atom_remover3(link_smiles, bond))

    
    cofid = sbu[0] +'.' + sbu[1] + '&&' + top[0] + '.' + f'cat{cat[0]}'
    
    return(cofid)

def atom_remover3(smiles, replacement):
    start_mol = Chem.MolFromSmiles(smiles)

    mod_mol = Chem.ReplaceSubstructs(
        start_mol, 
        Chem.MolFromSmarts('[BrD1]'),
        Chem.MolFromSmarts(f'{replacement}'),
        replaceAll=True
    )
    return(Chem.MolToSmiles(mod_mol[0]))

if __name__ == '__main__':
    main()