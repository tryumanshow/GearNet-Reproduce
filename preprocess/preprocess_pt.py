import re
import os
import glob
import gzip
import random
import pickle
import argparse
import numpy as np
from tqdm import tqdm
from itertools import chain
from sklearn.preprocessing import OneHotEncoder


known_atoms = ['ALA', 'ARG', 'ASN', 'ASP', 'CYS', 
            'GLN', 'GLU', 'GLY', 'HIS', 'ILE', 
            'LEU', 'LYS', 'MET', 'PHE', 'PRO', 
            'SER', 'THR', 'TRP', 'TYR', 'VAL']
known_atoms_cp = known_atoms.copy()
known_atoms_cp.append('UNK')
strange = 0



def pickle_file_dump_individual(data, dir_name):
    """
    To decrease heavy-file reading time. 
    """
    total_length = len(data)
    os.makedirs(dir_name, exist_ok=True)
    for i in range(total_length):
        file_name = f'index{i}.pkl'
        file_name = os.path.join(dir_name, file_name)
        with open(file_name, 'wb') as f:
            pickle.dump(data[i], f)
        f.close()


def atom_info_extraction(line):
    amino_acid = line[3]
    x_axis = float(line[6])
    y_axis = float(line[7])
    z_axis = float(line[8])
    summary = [amino_acid, x_axis, y_axis, z_axis]
    return summary


def get_alpha_carbon_data(path):

    global strange

    with gzip.open(path, 'rb') as f:
        lines = f.read()
        lines = lines.decode('utf-8').split('\n')
        lines = [x for x in lines if x[:4] == 'ATOM']
        
        alpha_carbon = []
        
        for line in lines:
            line = re.split(r' {1,}', line)
            try:
                if line[2] == 'CA':
                    summary = atom_info_extraction(line)
                    if summary[0] not in known_atoms:
                        summary[0] = 'UNK' # This case does not exist in 'Swiss-Prot'
                    alpha_carbon.append(summary)     
            except:
                strange += 1
                return None
            
    return alpha_carbon


def parse_swiss_prot(args):
    
    global strange
    
    strange = 0
    whole_path = glob.glob(f'{args.data_dir}/*')

    processed_data = [get_alpha_carbon_data(x) for x in tqdm(whole_path)]
    processed_data = [x for x in processed_data if x is not None]
    
    print(f'The Number of Excluded Data: {strange} / {len(whole_path)}')

    # Not splitted data ( Made it just in case )
    os.makedirs(os.path.join(args.save_dir, 'data'), exist_ok=True)
    with open(os.path.join(args.save_dir, 'for_pretrain.pkl'), 'wb') as f:
        pickle.dump(processed_data, f)
    f.close()
        
    # Split the preprocessed data
    total_cnt_list = list(range(len(processed_data)))
    while True:
        random.seed(args.seed)
        
        for_valid = random.sample(total_cnt_list, args.valid_cnt)
        for_train = [x for x in total_cnt_list if x not in for_valid]
        
        valid_sampled = [processed_data[x] for x in for_valid]
        sampled_cat = list(chain(*valid_sampled))
        sampled_mol = [x[0] for x in sampled_cat]
        if len(list(set(sampled_mol))) < 20:
            args.seed += 1
            continue
        
        train_sampled = [processed_data[x] for x in for_train]
        
        # Save all instances as an individual example to avoid excessive file loading time at dataloader. 
        pickle_file_dump_individual(train_sampled, os.path.join(args.save_dir, 'individual_data', 'train'))
                    
        pickle_file_dump_individual(valid_sampled, os.path.join(args.save_dir, 'individual_data', 'valid'))
        
            
        break
    
    # Make one-hot encoder in advance which will be used at 'dataloader'
    known_atoms_array = np.array(known_atoms_cp).reshape(-1, 1)
    onehot = OneHotEncoder(sparse=False)
    onehot_transformed = onehot.fit_transform(known_atoms_array)
    
    mol_idx_dict = {}
    for key, value in enumerate(known_atoms_cp):
        mol_idx_dict[value] = key
    
    os.makedirs(os.path.join(args.save_dir, 'stats'), exist_ok=True)
    with open(os.path.join(args.save_dir, 'stats', 'molecule.pkl'), 'wb') as f:
        pickle.dump([onehot, onehot_transformed, mol_idx_dict], f)
    f.close()

        
if __name__ == '__main__':
    parser = argparse.ArgumentParser() 
    parser.add_argument('--data_dir', type=str, default='/home/swryu/uniprot/dataset')
    parser.add_argument('--save_dir', type=str, default='/home/swryu/uniprot/interim')
    parser.add_argument('--seed', type=int, default=2022)
    parser.add_argument('--valid_cnt', type=int, default=3000) # Arbitrary
    args = parser.parse_args()
    
    parse_swiss_prot(args)