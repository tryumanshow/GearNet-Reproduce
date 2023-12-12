import os
import pickle

def pickle_file_dump(data, save_path):
    with open(save_path, 'wb') as f:
        pickle.dump(data, f)
    f.close()


def pickle_file_dump_individual(data, dir_name):
    """
    To decrease heavy-file reading time. 
    """
    total_length = len(data)
    os.makedirs(dir_name, exist_ok=True)
    for i in range(total_length):
        file_name = f'index{i}.pkl'
        file_name = os.path.join(dir_name, file_name)
        pickle_file_dump(data[i], file_name)


def pickle_file_load(load_path):
    with open(load_path, 'rb') as f:
        data = pickle.load(f)
    f.close()
    return data


def pickle_file_load_individual(load_path):
    output = []
    dirs = os.listdir(load_path)
    for dir_ in dirs:
        path = os.path.join(load_path, dir_)
        data = pickle_file_load(path)
        output.append(data)
    return output