import json
import os
JSON_PATH = 'index.json'

def print_index():
    with open(JSON_PATH, 'r') as file:
        data = json.load(file)
        for key in data:
            print(key)
            for subkey in data[key]:
                print(f'\t{subkey}: {data[key][subkey]}')

def get_experiment(folder):
    with open(JSON_PATH, 'r') as file:
        data = json.load(file)
        #Print the metadata and return the "subfolder" key for the experiment matching the folder name
        for key in data:
            for subkey in data[key]:
                if data[key][subkey] == folder:
                    print(data[key]['metadata'])
                    return data[key]['subfolders']

    
        
def index_folder(folder, title='Placeholder'):
    """Locates the specific folder in the Duke_Data/data folder, and indexes it by reading in the metadata.txt file and accumulating the data file names in the folder. Adds this information as a new entry in the index.json file"""
    meta_data = ''
    data_file_names = []
    print(os.getcwd())
    with open(f'data/{folder}/metadata.txt', 'r') as file:
        meta_data = file.read()
    for file in os.listdir(f'data/{folder}'):
        if file.endswith('.h5'):
            data_file_names.append(file)
    #Prompt the user for the folder name/title:
    #title = input('Enter the title of the experiment: ')
    """Formatted as:
    "title": {"folder": {folder}, "metadata": {metadata}, "subfolders": {data_file_names}}
    """
    with open(JSON_PATH, 'r') as file:
        data = json.load(file)

    # Update the data
    data[title] = {'folder': folder, 'metadata': meta_data, 'subfolders': data_file_names}

    # Open the file in write mode to write the updated data
    with open(JSON_PATH, 'w') as file:
        json.dump(data, file, indent=4)
    print('Indexing complete')
    print_index()