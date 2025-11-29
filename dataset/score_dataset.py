import os
import random
import numpy as np
import pandas as pd
from tqdm import tqdm
from datasets import Dataset, DatasetDict, load_from_disk


def max_len(example):

    '''
    example: row from dataset
    Ex: {'prompt': '1022086+9=', 'operand_len_a': 7, 'operand_len_b': 1, 'answer': '1022095'}

    return: difficulty = length of longest operand
    '''
    # To use in other types of scoring:
    #   operands_list = example['prompt'].replace("=", "").split('+')
    #       for max length score would be:
    #   difficulty = max([len(operand) for operand in operands_list])
    difficulty = max([example['operand_len_a'], example['operand_len_b']])
    return difficulty
    
def add_carries(example):
    operands_list = example['prompt'].replace("=", "").split('+')
    # store operands in variables longer and shorter, and reverse names if necessary
    a = operands_list[0]
    b = operands_list[1]    
    answer = example['answer']
    length = len(answer)
    # Pad the shorter operand with leading zeros
    a = '0' * (len(answer)- len(a)) + a
    b = '0' * (len(answer)- len(b)) + b
    # Get the individual digits of each operand and answer in lists
    a_as_list = [int(character) for character in a]
    b_as_list = [int(character) for character in b]
    answer_as_list = [int(character) for character in answer]

    # Compute digit sums mod 10
    digit_sum_mod_10  = [(a_as_list[i] + b_as_list[i])%10 
                                for i in range(len(answer))]
    # Compute diffs 
    diffs_mod_10 = [(answer_as_list[i] - digit_sum_mod_10[i])%10 
                                for i in range(len(answer_as_list))]
    # Number of carries is how many diffs list elements are nonzero
    carries = sum([diffs_mod_10[i] !=0
                                for i in range(len(answer_as_list))])
    
    return carries

def score_data(data, metric):
    """
    data: an iterable with entries of the form 
    {'prompt': '1022086+9=', 'operand_len_a': 7, 'operand_len_b': 1, 'answer': '1022095'}
    metric: a function giving an integer score for difficulty of the prompt
    
    return 1D numpy array of difficulty scores
    """
    
    difficulties = np.zeros(len(data)) - 1
    for i, example in tqdm(enumerate(data), total=len(data)):
        difficulties[i] = metric(example)

    return difficulties

def score_dataset(dataset, metric):
    """
    Args:
        dataset (_type_): dictionary of datasets
                            typically {'train':iterable, 'val': iterable, 'test': iterable}
        metric (_type_): takes an example and returns an integer 
    """
    print(f"metric is {metric}")
    segment_max_difficulties = []
    for segment in dataset:
        print(f"Scoring split: {segment}")
        # Compute difficulties and append to dataset
        difficulties = score_data(dataset[segment], metric)
        dataset[segment] = dataset[segment].add_column('difficulty', difficulties)

        max_diff = np.max(difficulties)
        segment_max_difficulties.append(max_diff)

        print(f"Done scoring {segment}, max unnormalized difficulty {max_diff}")

    return dataset 
 
### TO Do: make sure the path makes sense with the file name
def save_dataset_to_disk(dataset_dict, save_dir):
    os.makedirs(save_dir, exist_ok=True)
    dataset_dict.save_to_disk(save_dir)
    print("Saved Huggingface Dataset to disk")

######### SET THESE VARIABLES !!!!!!!!!!!!!! ###############
DATASET_PATH = 'data/addition_dataset_big'
METRIC = max_len

if __name__== '__main__':
    ### Set path and metric above before running
    metric_name = getattr(METRIC, "__name__", str(METRIC))
    print(f"metric name is {metric_name}")
    dataset_dict = load_from_disk(DATASET_PATH)

    scored_dataset = score_dataset(dataset_dict, METRIC)
    save_dir = DATASET_PATH + '_scored_' + metric_name
    save_dataset_to_disk(scored_dataset, save_dir)
