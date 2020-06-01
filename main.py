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

## Constants ##
ROOT_TOKEN = "<root>"


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
                    POS = splited_values[3]

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
    data = PosDataReader(r'mini_train.labeled', word_dict, pos_dict)
    print(data.D)


if __name__ == "__main__":
    main()
