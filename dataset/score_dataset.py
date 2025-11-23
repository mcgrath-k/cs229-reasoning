import os
import random
import numpy as np
import pandas as pd
from tqdm import tqdm
from datasets import Dataset, DatasetDict, load_from_disk


def addition_max_length(example):

    '''
    example: row from dataset
    Ex: {'prompt': '1022086+9=', 'operand_len_a': 7, 'operand_len_b': 1, 'answer': '1022095'}

    return: difficulty = length of longest operand
    '''
    # To use in other types of scoring:
    #   operands_list = example['prompt'].replace("=", "").split('+')
    #       for max length score would be:
    #   difficulty = max([len(operand) for operand in operands_list])
    METRIC = '_max_len'
    difficulty = max([example['operand_len_a'], example['operand_len_b']])
    return difficulty
    

def score_data(data, metric):
    """
    data: an iterable with entries of the form 
    {'prompt': '1022086+9=', 'operand_len_a': 7, 'operand_len_b': 1, 'answer': '1022095'}
    metric: a function giving an integer score for difficulty of the prompt
    
    return 1D numpy array of difficulty scores
    """
    
    difficulties = np.zeros(len(data))-1
    for i in range(len(data)):
        difficulties[i] = metric(data[i])

    return difficulties 

def score_dataset(dataset, metric):
    """
    Args:
        dataset (_type_): dictionary of datasets
                            typically {'train':iterable, 'val': iterable, 'test': iterable}
        metric (_type_): takes an example and returns an integer 
    """
    print(f"metric is {metric}")

    for segment in dataset:
        # Compute difficulties and append to dataset
        difficulties = score_data(dataset[segment], metric)
        dataset[segment] = dataset[segment].add_column('difficulty', difficulties)
        print(f"Done scoring {segment}")

    return dataset 

### TO Do: make sure the path makes sense with the file name
def save_dataset_to_disk(dataset_dict, save_dir):
    os.makedirs(save_dir, exist_ok=True)
    dataset_dict.save_to_disk(save_dir)
    print("Saved Huggingface Dataset to disk")

DATASET_PATH = 'data/addition_dataset'
# METRIC is set in the call of the metric function used
METRIC = ''

if __name__== '__main__':

    dataset_dict = load_from_disk(DATASET_PATH)
    scored_dataset = score_dataset(dataset_dict, addition_max_length)
    save_dir = DATASET_PATH + '_scored' + METRIC
    save_dataset_to_disk(scored_dataset, save_dir)
