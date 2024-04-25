import pymatgen.core as mg
from pymatgen.analysis.graphs import StructureGraph
from pymatgen.analysis.local_env import VoronoiNN
import csv, os 

#st_data_dir = '/rds/general/user/jkd19/home/working_dir/git_copy/cofs/npj-computmater-thermal-conductivity-dataset/git_mof_dat/regnerated_mof_cifs'

def _read_st(fn):
    st = mg.Structure.from_file(os.path.join(st_data_dir, fn+'.cif'))
    st_p = st.to_primitive()
    st_p.properties['name'] = fn
    return(st_p)

def number_of_vertices_edges(st_object):

    structure = st_object.to_primitive()
    # Create a StructureGraph from the structure
    try:
        structure_graph = StructureGraph.with_local_env_strategy(structure, VoronoiNN(tol=0.4))
    except Exception as err:
        return(0,0)

    # Get the vertices (nodes)
    vertices = list(structure_graph.graph.nodes)

    # Initialize an empty list to store edges
    edges = []

    # Iterate through the nodes and find connected edges
    for i in vertices:
        neighbors = list(structure_graph.graph.neighbors(i))

        for j in neighbors:
            edges.append((i, j))
    #with open('v_e_data.csv', 'a') as f:
    #    cw = csv.writer(f)
    #    cw.writerow((st_object.properties['name'], len(vertices), len(edges)))
    return(len(vertices), len(edges))

def noof_c_h_n_o(st_object):
    structure = st_object.to_primitive()

    # Create a dictionary to store the count of each atom type
    atom_counts = {}

    # Iterate through the sites in the structure
    for site in structure:
        element = site.specie.symbol
        if element in atom_counts:
            atom_counts[element] += 1
        else:
            atom_counts[element] = 1
    ele_dir ={}
    
    # Print the counts of each atom type
    for element, count in atom_counts.items():
        ele_dir[element]= count
              
    outs = []
    for x in ['C','H','N','O']:
        if x not in ele_dir.keys():
            outs += [0]
        else:
            outs += [ele_dir[x]]
    #with open('chno_data.csv', 'a') as f:
    #    cw = csv.writer(f)
    #    cw.writerow((st_object.properties['name'], outs[0],outs[1],outs[2],outs[3]))
    return(outs)


def is_3d(st_obj):
    if st_obj.is_3d_periodic == True:
        return 3
    else:
        return 2