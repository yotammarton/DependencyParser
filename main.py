from torchtext.vocab import Vocab
from torch.utils.data.dataset import Dataset, TensorDataset
from pathlib import Path
from collections import Counter
from collections import defaultdict
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import random
from torch.utils.data.dataloader import DataLoader

## from HW1 Wet ##
from collections import OrderedDict
import re
from scipy import optimize
import pickle
import scipy.special as special
import matplotlib.pyplot as plt
from matplotlib import colors
import seaborn as sns
import time
import os

import eyal.py as eyal

## Constants ##
ROOT_TOKEN = "<root>"

def get_vocabs(list_of_paths):
    """
        create dictionary with number of appears of each word and each tag
    """
    word_dict = defaultdict(int)
    pos_dict = defaultdict(int)

    for file_path in list_of_paths:
        with open(file_path) as f:
            sentence, true_tree = [ROOT_TOKEN], list()
            for line in f:
                if line != "\n":
                    splited_values = re.split('\t', line)
                    # m = splited_values[0]
                    # h = splited_values[6]
                    word = splited_values[1]
                    pos = splited_values[3]

                    word_dict[word] += 1
                    pos_dict[pos] += 1
    return word_dict, pos_dict

def create_idx_dicts(word_dict, pos_dict):
    """
    create dictionary with index to each word. also dictionary with index to each pos.
    we should call this function  only if we create embedding vectors by ourselves.
    """
    word_idx_dict = defaultdict(int)
    pos_idx_dict = defaultdict(int)
    idx = 0
    for word in word_dict.keys():
        word_idx_dict[word] = idx
        idx += 1

    idx = 0
    for pos in pos_dict.keys():
        pos_idx_dict[pos] = idx
        idx += 1

    return word_idx_dict, pos_idx_dict

class PosDataReader:
    def __init__(self, file, word_dict, pos_dict):  ## call to readData
        self.file = file
        self.D = list()
        self.__readData__()

        self.word_dict = word_dict  # TODO need it?
        self.pos_dict = pos_dict  # TODO need it?

    def __readData__(self):
        """main reader function which also populates the class data structures"""

        with open(self.file) as f:
            sentence, true_tree = [ROOT_TOKEN], list()
            for line in f:
                if line == "\n":
                    if true_tree:
                        self.D.append((sentence, true_tree))
                    sentence, true_tree = [ROOT_TOKEN], list()
                else:
                    splited_values = re.split('\t', line)
                    m = splited_values[0]
                    h = splited_values[6]
                    word = splited_values[1]
                    pos = splited_values[3]

                    sentence.append(word)
                    true_tree.append((h, m))

    def get_num_sentences(self):
        """returns num of sentences in data"""
        return len(self.D)



# def create_dataset_D(file_path):
#     D = list()
#
#     with open(file_path) as f:
#         print("in")
#         sentence, true_tree = [ROOT_TOKEN], list()
#         for line in f:
#             if line == "\n":
#                 if true_tree:
#                     D.append((sentence, true_tree))
#                 sentence, true_tree = [ROOT_TOKEN], list()
#             else:
#                 splited_values = re.split('\t', line)
#                 m = splited_values[0]
#                 h = splited_values[6]
#                 word = splited_values[1]
#                 POS = splited_values[3]
#
#                 sentence.append(word)
#                 true_tree.append((h, m))
#     print(D)
#     return D

def main():
    word_dict, pos_dict = dict(), dict()
    # dir_path = r'mini_train.labeled'
    dir_path = r''

    # data = PosDataReader(dir_path, word_dict, pos_dict)
    # print(data.D)

    PosDataset = eyal.PosDataset(word_dict, pos_dict, dir_path, subset="train1",
                 padding=False, word_embeddings=None)


if __name__ == "__main__":
    main()