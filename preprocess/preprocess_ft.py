import os
import re
import wget
import csv
import sys
import h5py
import gzip
import string
import argparse
import numpy as np
import pandas as pd
from tqdm import tqdm
from Bio import SeqIO
from itertools import chain
from joblib import Parallel, delayed


dirname = os.path.dirname(os.path.abspath(os.path.dirname(__file__)))
sys.path.append(dirname)

from utils.file_utils import pickle_file_dump, pickle_file_dump_individual


known_atoms = ['ALA', 'ARG', 'ASN', 'ASP', 'CYS', 
            'GLN', 'GLU', 'GLY', 'HIS', 'ILE', 
            'LEU', 'LYS', 'MET', 'PHE', 'PRO', 
            'SER', 'THR', 'TRP', 'TYR', 'VAL']

mapping = {
        'A': 'ALA',
        'C': 'CYS',
        'D': 'ASP',
        'E': 'GLU',
        'F': 'PHE',
        'G': 'GLY',
        'H': 'HIS',
        'I': 'ILE',
        'K': 'LYS',
        'L': 'LEU',
        'M': 'MET',
        'N': 'ASN',
        'P': 'PRO',
        'Q': 'GLN',
        'R': 'ARG',
        'S': 'SER',
        'T': 'THR',
        'V': 'VAL',
        'W': 'TRP',
        'Y': 'TYR',
    }

debug = {}
for key, value in mapping.items():
    debug[value] = key

alphabet = list(string.ascii_uppercase)
possible_unk = [amino for amino in alphabet if amino not in mapping.keys()]
for token in possible_unk:
    mapping[token] = 'UNK'

identity = ['train', 'valid' ,'test']


#%%
###########################################
# For Enzyme Commision (EC) Prediction &  #
#      Gene Ontology (GO) term prediction #
###########################################

def gather_splits(file_paths):
    prot_dic = {}
    
    for idx, path in enumerate(file_paths):
        temp = []
        with open(path, 'r') as f:
            lines = f.readlines()
            for line in lines:
                line = line.strip()
                line = line.split('-')[0]
                line = line.lower()
                temp.append(line)
            prot_dic[identity[idx]] = temp
            
    return prot_dic


def download_files(args, splits):
    fail = {}
    
    new_dir = os.path.join(args.data_dir, args.task, 'downloaded')
    os.makedirs(new_dir, exist_ok=True)
    
    new_dirs = []
    for id in identity:
        path_cat = os.path.join(new_dir, id)
        os.makedirs(path_cat, exist_ok=True) 
        new_dirs.append(path_cat)
    
    for i, id in enumerate(identity):
        outlier = []
        id_extracted = splits[id]
        for ie in id_extracted:
            if f'{ie}.pdb.gz' in os.listdir(new_dirs[i]):
                continue
            url = f'https://files.rcsb.org/download/{ie}.pdb.gz'
            try:
                wget.download(url, out=new_dirs[i])
            except:
                outlier.append(ie)
        fail[id] = len(outlier)
    
    print(f'The number of proteins which exist on the list of split data, but not on the pdb archive: {fail}')
        
    
def extract_info_from_split(args):
    split_path = os.path.join(args.data_dir, 'EC_GO')
    
    prot_dic = {}
    
    if args.task == 'EC':
        file_iter = [f'nrPDB-EC_2020.04_{split}.txt' for split in identity]
    else:
        file_iter = [f'nrPDB-GO_2019.06.18_{split}.txt' for split in identity]
    
    for idx, id in enumerate(identity):
        prot_dic[id] = {}
        file = file_iter[idx]
        file_path = os.path.join(split_path, file)
        with open(file_path, 'r') as f:
            lines = f.read()
            lines = lines.split('\n')
            if lines[-1] == '':
                lines = lines[:-1]
            for line in lines:
                prot, chain = line.split('-')
                prot = prot.lower()
                if prot in prot_dic[id].keys():
                    prot_dic[id][prot].append(chain)
                else:
                    prot_dic[id][prot] = [chain]
    
    return prot_dic # {'1r9w': ['A'], '3u7v': ['A'], '1ck7': ['A'], ... }
  

def index_exceptional(seq_id):
    if str.isalpha(seq_id[4]) or str.isdigit(seq_id[4]):
        return seq_id
    else:  # ex: A1570
        alphas = ''.join(x for x in seq_id[4] if x.isalpha())
        nums = ''.join(x for x in seq_id[4] if x.isdigit())
        return list(chain(seq_id[:4], [alphas, nums], seq_id[5:]))
    

def chainname_exceptional(seqres, chain):
    if seqres[4] == chain:
        pass
    else:
        chain_, order = seqres[4][:len(chain)], seqres[4][len(chain):]
        seqres[4:5] = chain_, order
    return seqres


def coordinate_exceptional(seqres):
    try:
        coord = [float(x) for x in seqres[6:9]]
    except: 
        coord = [re.split(r'(-[0-9.]+)', x) for x in seqres[6:8]]
        coord = list(chain(*coord))
        coord = [x for x in coord if x != '']
        coord = [float(x) for x in coord][:3]
        if len(coord) == 2: # '1HRZ-A' => '-7.419', '-7.697', '-17.064101.08',
            coord = None
    return coord
        

def atom_exceptional(x):
    if x[3] in known_atoms:
        output = [x[3]]
    elif x[3] == 'MSE': # MET is sometimes expressed as MSE.
        output = ['MET']
    else:
        output = ['UNK']
    return output
 

def atom_and_coordinate(prot_name, seqres, chain, fasta_dict):
    global outlier, known_atoms    
    
    seqres_ = [chainname_exceptional(x, chain) for x in seqres if chain in x[4]]
    atoms = [atom_exceptional(x) for x in seqres_]
    
    """
    Due to the 'un-observed' tail, the length of sequence in PDB is always shorter (or at most same) than FASTA.
    Therefore, in case of multi-chain, cut according to the maximum length of FASTA file.
    """
    
    max_len = len(fasta_dict[prot_name])
        
    if max_len > len(seqres_): 
        # Follows the info. of PDB as it is. 
        pass
    
    else: # Cut 
        atoms = atoms[:max_len]
        seqres_ = seqres_[:max_len]
        
    try:
        coordinates = [coordinate_exceptional(x) for x in seqres_]
        if None in coordinates:
            print(f'Strange cases exist in {prot_name} at representing 3d coordinates.')
            outlier += 1
            return None
        else:
            output = list(map(lambda x, y: x+y, atoms, coordinates)) 
            return output
    except:
        # ex) '5W0U-B': '255.193', '58.3641226.437', '1.00114.04'
        print(f'Strange cases exist in {prot_name} at representing 3d coordinates.')
        outlier += 1
        return None
                 

    
def extract_seq_info_from_FASTA(args):
    
    fasta_dict = {}
    main_path = os.path.join(args.data_dir, 'EC_GO')
    if args.task == 'EC':
        main_path += '/nrPDB-EC_2020.04_sequences.fasta'
    else:
        main_path += '/nrPDB-GO_2019.06.18_sequences.fasta'

    # Loads fasta file and returns a list of the Bio SeqIO records
    infile = open(main_path, 'rU')
    for entry in SeqIO.parse(infile, 'fasta'):
        key = str(entry.id)
        sequence = str(entry.seq)
        sequence = [mapping[x] for x in sequence]
        fasta_dict[key] = sequence
    return fasta_dict
    
    
def extract_specific_chain(prot_name, id, seqres, chains, fasta_dict):
    global outlier 
    
    chain_collection = {}
    prot_name = prot_name.upper()

    seqres = [re.split(r'[ ]+', x) for x in seqres]
    seqres = [x for x in seqres if x[4][0] in chains and x[2] == 'CA']  # 0: to treat the case such as 'A1570' as 'A'
    seqres = [index_exceptional(x) for x in seqres] # Due to the recording error in downloaded original 'PDB' file.
    
    if len(seqres) == 0:
        outlier += 1
        print(f"Strange cases exist in {prot_name} at representing type of chain.")
        print("I (arbitrarily) excluded this data from my dataset.")
        return None
    
    if len(chains) > 1: # Multi-chain
        print(f'{prot_name} has {len(chains)} separate chains on {id} dataset: {prot_name.upper()}-{chains}')
        
    for chain in chains:
        prot_name_ = prot_name + f'-{chain}'
        output = atom_and_coordinate(prot_name_, seqres, chain, fasta_dict)
        if output is None:
            continue
        chain_dict = {prot_name_: output}
        chain_collection.update(chain_dict)
    
    return chain_collection
    
    
def extract_sequence_and_coordinate(args, prot_chain_dict, fasta_dict):
    
    global outlier
    outlier = 0
    
    data_path = os.path.join(args.data_dir, args.task, 'downloaded')
    
    identity_dic = {}
    
    for id in identity: 
        id_dic = {}
        identity_dic[id] = {}
        
        reference = prot_chain_dict[id].keys()
        
        main_path = os.path.join(data_path, id)
        listdir = os.listdir(main_path)
        for ld in listdir:
            if '(' in ld: # File is sometimes downloaded twice.
                continue
            prot_name = ld.split('.')[0]
            if prot_name in reference:
                chain = prot_chain_dict[id][prot_name]
                with gzip.open(os.path.join(main_path, ld), 'rt') as f:
                    lines = f.read().split('\n')
                    seqres = [line for line in lines if line[:4] == 'ATOM' and 'REMARK' not in line] 
                    output = extract_specific_chain(prot_name, id, seqres, chain, fasta_dict)
                    if len(output) == 0:
                        continue
                    id_dic.update(output)
                f.close()
        print(f'Coordinate outlier for {id} dataset: {outlier}')
        outlier = 0
        identity_dic[id].update(id_dic)
    
    return identity_dic
                    

def load_EC_annot(data_path):
    """
    Annotation from:
    https://github.com/flatironinstitute/DeepFRI/blob/fa9409cca7dc7b475f71ab4bab0aa7b6b1091448/deepfrier/utils.py
    """
    
    filename = os.path.join(data_path, 'nrPDB-EC_2020.04_annot.tsv')
    
    prot2annot = {}
    with open(filename, mode='r') as tsvfile:
        reader = csv.reader(tsvfile, delimiter='\t')

        # molecular function
        next(reader, None)  # skip the headers
        ec_numbers = {'ec': next(reader)}
        next(reader, None)  # skip the headers
        counts = {'ec': np.zeros(len(ec_numbers['ec']), dtype=float)}
        for row in tqdm(reader):
            prot, prot_ec_numbers = row[0], row[1]
            ec_indices = [ec_numbers['ec'].index(ec_num) for ec_num in prot_ec_numbers.split(',')]
            prot2annot[prot] = {'ec': np.zeros(len(ec_numbers['ec']), dtype=np.int64)}
            prot2annot[prot]['ec'][ec_indices] = 1.0
            counts['ec'][ec_indices] += 1
    
    return prot2annot, ec_numbers, ec_numbers, counts


def load_GO_annot(data_path):
    # Load GO annotations
    
    filename = os.path.join(data_path, 'nrPDB-GO_2019.06.18_annot.tsv')
    
    onts = ['mf', 'bp', 'cc']
    prot2annot = {}
    goterms = {ont: [] for ont in onts}
    gonames = {ont: [] for ont in onts}
    with open(filename, mode='r') as tsvfile:
        reader = csv.reader(tsvfile, delimiter='\t')

        # molecular function
        next(reader, None)  # skip the headers
        goterms[onts[0]] = next(reader)
        next(reader, None)  # skip the headers
        gonames[onts[0]] = next(reader)

        # biological process
        next(reader, None)  # skip the headers
        goterms[onts[1]] = next(reader)
        next(reader, None)  # skip the headers
        gonames[onts[1]] = next(reader)

        # cellular component
        next(reader, None)  # skip the headers
        goterms[onts[2]] = next(reader)
        next(reader, None)  # skip the headers
        gonames[onts[2]] = next(reader)

        next(reader, None)  # skip the headers
        counts = {ont: np.zeros(len(goterms[ont]), dtype=float) for ont in onts}
        for row in reader:
            prot, prot_goterms = row[0], row[1:]
            prot2annot[prot] = {ont: [] for ont in onts}
            for i in range(3):
                goterm_indices = [goterms[onts[i]].index(goterm) for goterm in prot_goterms[i].split(',') if goterm != '']
                prot2annot[prot][onts[i]] = np.zeros(len(goterms[onts[i]]))
                prot2annot[prot][onts[i]][goterm_indices] = 1.0
                counts[onts[i]][goterm_indices] += 1.0
    
    return prot2annot, goterms, gonames, counts


def extract_test_split(data_path, task):
    
    csv_file_path = 'nrPDB-EC_2020.04_test.csv' if task == 'EC' else 'nrPDB-GO_2019.06.18_test.csv'
    filename = os.path.join(data_path, csv_file_path)
    
    split_pd = pd.read_csv(filename, sep=',')
    
    pdb_chain = split_pd['PDB-chain']
    
    split_30 = pdb_chain.loc[split_pd['<30%'] == 1].values.tolist()
    split_40 = pdb_chain.loc[split_pd['<40%'] == 1].values.tolist()
    split_50 = pdb_chain.loc[split_pd['<50%'] == 1].values.tolist()
    split_70 = pdb_chain.loc[split_pd['<70%'] == 1].values.tolist()
    split_95 = pdb_chain.loc[split_pd['<95%'] == 1].values.tolist()
   
    splits = {'30': split_30, '40': split_40, '50': split_50, '70': split_70, '95': split_95}
    
    return splits


def classify_test_split(mol, label, key, value):
    features = [mol[x] for x in value if x in mol.keys()]
    labels = [label[x] for x in value if x in label.keys()]
    return {key: [features, labels]}
    

def file_save(args, test_split, features, label, cnt):
    
    def save(args, prot_feat, prot_label, split):
        save_dir = os.path.join(args.save_dir, args.task, split)
        feature_dir = os.path.join(save_dir, 'feature')
        label_dir = os.path.join(save_dir, 'label')
        os.makedirs(feature_dir, exist_ok=True)
        os.makedirs(label_dir, exist_ok=True)
        pickle_file_dump_individual(prot_feat, feature_dir)
        pickle_file_dump_individual(prot_label, label_dir)
    
    for split, mol in features.items():
        prots = list(mol.keys())
        if split != 'test':
            prot_feat = [mol[x] for x in prots]
            prot_label = [label[x] for x in label]
            save(args, prot_feat, prot_label, split)
        else:
            labels = Parallel(n_jobs=5)(
                delayed(classify_test_split)(mol, label, key, value)
                for key, value in test_split.items()
            )
            for chunks in labels:
                resolution = list(chunks.keys())[0]
                prot_feat, prot_label = chunks.get(resolution)
                child = split + resolution
                save(args, prot_feat, prot_label, child)             
                        
    pickle_file_dump(cnt, os.path.join(args.save_dir, args.task, 'cnt.pkl')) # weight for loss
            

def preprocess_ec_go(args):

    global known_atoms

    raw_data_path = os.path.join(args.data_dir, 'EC_GO')
    
    selected = os.listdir(raw_data_path)
    selected = [os.path.join(raw_data_path, x) for x in selected \
                                    if args.task in x and x.split('.')[-1] == 'txt']
    
    fn_arrange = []
    for id in identity:
        fn = [x for x in selected if id in x]
        fn_arrange.append(fn[0])
    
    # Download files
    answer = input("Are you going to download dataset? It might take about a day. [yes, no]: ")
    if answer == 'yes':
        splits = gather_splits(fn_arrange)
        download_files(args, splits)
    
    fasta_dict = extract_seq_info_from_FASTA(args)
    prot_chain_dict = extract_info_from_split(args)
    features = extract_sequence_and_coordinate(args, prot_chain_dict, fasta_dict)
    test_split = extract_test_split(raw_data_path, args.task)
    
    if args.task == 'EC':
        label, _, _, cnt = load_EC_annot(raw_data_path)
    else: 
        label, _, _, cnt = load_GO_annot(raw_data_path)
    
    file_save(args, test_split, features, label, cnt)
    
    print(f'Intermediate preprocessing for {args.task} dataset is done!')


#%%

##################################
## For Fold Classification (FC) ##
##################################

# It was not enough to use the data preprocessed in advance as it is, 
# because GearNet needs coordinate information.

def feature_dic(data_path, dl_list):
    """
    : Also used at Reaction Classification preprocessing.
    : Using for-loop was faster than Parellel joblib in 128-cpu cores setting.
    """
    input_dict = {}
    for dl in tqdm(dl_list):
        with h5py.File(os.path.join(data_path, dl), 'r') as f:
            prot_name = dl.replace('.hdf5', '')
            in_interest = [list(f[x]) for x in ['atom_names', 'amino_pos', 'atom_residue_names']]
            for idx, contents in enumerate(in_interest):
                if idx == 0: # atom_names
                    contents = [x.decode() for x in contents]
                    ca_idx = [i for i, x in enumerate(contents) if x == 'CA']
                elif idx == 1: # amino_pos
                    contents = contents[0]
                    coords = np.split(contents, contents.shape[0])
                    coords = [x.squeeze(0).tolist() for x in coords]
                else: # atom_residue_names
                    res = [x.decode() for i, x in enumerate(contents) if i in ca_idx]
                    res = [x if x in known_atoms else 'UNK' for x in res]
                    res = [[x] for x in res]
        assert len(ca_idx) == len(coords) == len(res), 'Strange'
        feature = list(map(lambda x, y: x+y, res, coords))
        if prot_name in input_dict:
            raise ValueError('Original data is strange.')
        input_dict[prot_name] = feature
    return input_dict


def mapping_dic(task_path):
    mapping_dict = {}
    with open(os.path.join(task_path, 'class_map.txt'), 'r') as f:
        lines = f.readlines()
        for line in lines:
            line = line.strip()
            key, value = line.split('\t')
            if key in mapping_dict:
                raise ValueError()
            mapping_dict[key] = int(value)
    return mapping_dict


def extract_names_labels_fc(split_path):
    with open(split_path, 'r') as f:
        lines = f.readlines()
        names = [x.strip().split('\t')[0] for x in lines]
        labels = [x.strip().split('\t')[-1] for x in lines]
    f.close()
    return names, labels


def get_features_labels_fc(prot_split,  prot_label, feature_dict, func_dict):
    features = [feature_dict[x] for x in prot_split]
    labels = [func_dict[x] for x in prot_label]
    return features, labels


def preprocess_fc(args):
    file_order = ['training', 'validation', 'test_fold', 'test_superfamily', 'test_family']
    task_path = os.path.join(args.data_dir, args.task)
    data_paths = [os.path.join(task_path, x) for x in file_order]
    
    func_dict = mapping_dic(task_path)
    feat_dict_list = []
    for data_path in data_paths:
        dir_list = os.listdir(data_path)
        feature_dict = feature_dic(data_path, dir_list) 
        feat_dict_list.append(feature_dict)
    
    task_dirs = os.path.join(args.save_dir, args.task)
    os.makedirs(task_dirs, exist_ok=True)
    
    file_domain = ['train', 'valid', 'test_fold', 'test_superfamily', 'test_family']
    for idx, split in enumerate(file_order):
        split_path = os.path.join(task_path, split) + '.txt'
        prot_split, prot_label = extract_names_labels_fc(split_path)
        feature, label = get_features_labels_fc(prot_split, 
                                                prot_label,
                                                feat_dict_list[idx], 
                                                func_dict)
        os.makedirs(os.path.join(task_dirs, file_domain[idx]), exist_ok=True)
        pickle_file_dump_individual(feature, os.path.join(task_dirs, file_domain[idx], 'feature'))
        pickle_file_dump_individual(label, os.path.join(task_dirs, file_domain[idx], 'label'))

    print('Intermediate preprocessing for FC dataset is done!')



#%%

######################################
## For Reaction Classification (RC) ##
######################################

# It was not enough to use the data preprocessed in advance as it is, 
# because GearNet needs coordinate information.

def function_dic(task_path):
    func = {}
    with open(os.path.join(task_path, 'chain_functions.txt'), 'r') as f:
        lines = f.readlines()
        for line in lines:
            line = line.strip()
            protein, label = line.split(',')
            if protein in func:
                raise ValueError
            func[protein] = label
    return func


def extract_names_rc(split_path):
    with open(split_path, 'r') as f:
        lines = f.readlines()
        lines = [x.strip() for x in lines]
    f.close()
    return lines


def get_features_labels_rc(prot_split, feature_dict, func_dict):
    features = [feature_dict[x] for x in prot_split]
    labels = [int(func_dict[x]) for x in prot_split]
    return features, labels


def preprocess_rc(args):
    task_path = os.path.join(args.data_dir, args.task)
    data_path = os.path.join(task_path, 'data')
    dir_list = os.listdir(data_path)
    
    func_dict = function_dic(task_path)
    feature_dict = feature_dic(data_path, dir_list) 
    
    task_dirs = os.path.join(args.save_dir, args.task)
    os.makedirs(task_dirs, exist_ok=True)
    
    file_domain = ['train', 'valid', 'test']
    for idx, split in enumerate(['training', 'validation', 'testing']):
        split_path = os.path.join(task_path, split) + '.txt'
        prot_split = extract_names_rc(split_path)
        feature, label = get_features_labels_rc(prot_split, 
                                             feature_dict, 
                                             func_dict)
        
        os.makedirs(os.path.join(task_dirs, file_domain[idx]), exist_ok=True)
        pickle_file_dump_individual(feature, os.path.join(task_dirs, file_domain[idx], 'feature'))
        pickle_file_dump_individual(label, os.path.join(task_dirs, file_domain[idx], 'label'))
   
   
    print('Intermediate preprocessing for RC dataset is done!')     


#%%
        
if __name__ == '__main__':
    parser = argparse.ArgumentParser() 
    parser.add_argument('--task', type=str, default='FC', 
                        choices=['EC', 'GO', 'FC', 'RC'],
                        help='EC: Enzyme Commision number prediction, \
                            GO: Gene Ontology term prediction, \
                            FC: Fold Classification, \
                            RC: Reaction Classification')
    parser.add_argument('--data_dir', type=str, default='/home/swryu/downstream/dataset')
    parser.add_argument('--save_dir', type=str, default='/home/swryu/downstream/interim')
    parser.add_argument('--seed', type=int, default=2022)
    args = parser.parse_args()
    
    WHICH_FUNCTION = {
        'EC': preprocess_ec_go,
        'GO': preprocess_ec_go,
        'RC': preprocess_rc,
        'FC': preprocess_fc
    }
    
    WHICH_FUNCTION[args.task](args)
    