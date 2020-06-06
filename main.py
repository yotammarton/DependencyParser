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
from chu_liu_edmonds import decode_mst

# Constants
ROOT_TOKEN = "<root>"
UNKNOWN_TOKEN = "<unk>"
SPECIAL_TOKENS = [ROOT_TOKEN, UNKNOWN_TOKEN]


# returns {'The': 15374, 'I': 1556, 'Boeing': 85....}, {'DT': 17333, 'NNP': 5371, 'VBG': 5353....}
def get_vocabs_counts(list_of_paths):
    """
        create dictionary with number of appearances (counts) of each word and each tag
    """
    word_dict = defaultdict(int)
    pos_dict = defaultdict(int)

    for file_path in list_of_paths:
        with open(file_path) as f:
            sentence, true_tree = [ROOT_TOKEN], list()
            for line in f:
                if line != "\n":
                    splited_values = re.split('\t', line)
                    # m = int(splited_values[0])
                    # h = int(splited_values[6])
                    word = splited_values[1]
                    pos = splited_values[3]

                    word_dict[word] += 1
                    pos_dict[pos] += 1
    return word_dict, pos_dict


# returns {'The': 0, 'I': 1, 'Boeing': 2....}, {'DT': 0, 'NNP': 1, 'VBG': 2....}
def create_idx_dicts(word_dict, pos_dict):
    """
    create dictionary with index to each word. also dictionary with index to each pos.
    we should call this function only if we create embedding vectors by ourselves.
    :param word_dict: a dictionary with the different words
    :param pos_dict: a dictionary with the different tags
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
    def __init__(self, file, word_dict, pos_dict):  # call to readData
        self.file = file
        self.D = list()
        self.word_dict = word_dict
        self.pos_dict = pos_dict
        self.__readData__()

    def __readData__(self):
        """main reader function which also populates the class data structures"""
        with open(self.file) as f:
            sentence, tags, heads = [ROOT_TOKEN], [], []
            for line in f:
                if line == "\n":
                    if heads:
                        self.D.append((sentence, tags, heads))
                    sentence, tags, heads = [ROOT_TOKEN], [], []
                else:
                    splited_values = re.split('\t', line)
                    m = int(splited_values[0])
                    h = int(splited_values[6])
                    word = splited_values[1]
                    pos = splited_values[3]

                    sentence.append(word)
                    tags.append(pos)
                    heads.append(h)

                    # e.g.
                    # ['<root>', 'It', 'has', 'no', 'bearing', 'on', 'our', 'work', 'force', 'today', '.'] len = 11
                    # ['PRP', 'VBZ', 'DT', 'NN', 'IN', 'PRP$', 'NN', 'NN', 'NN', '.']                      len = 10
                    # ['2', '0', '4', '2', '4', '8', '8', '5', '8', '2']                                   len = 10

    def get_num_sentences(self):
        """returns num of sentences in data"""
        return len(self.D)


class PosDataset(Dataset):
    def __init__(self, word_dict, pos_dict, path: str, padding=False, word_embeddings=None):
        super().__init__()
        self.file = path
        self.datareader = PosDataReader(self.file, word_dict, pos_dict)
        self.vocab_size = len(self.datareader.word_dict)
        # TODO
        # בחלק הזה הוא מסביר על זה שבדוגמא פה הוא עושה שימוש בגלוב עבור האמבדינג
        # ניתן לשפר את המשקולות שלו לדאטא סט שלנו ולא בהכרח להישאר עם המשקולות הקבועות
        # בנוסף, אנחנו נצטרך לעשות וורד-אמבדינג גם לטאגס
        if word_embeddings:
            self.word_idx_mappings, self.idx_word_mappings, self.word_vectors = word_embeddings
        else:  # pre-trained -- Download Glove
            self.word_idx_mappings, self.idx_word_mappings, self.word_vectors = self.init_word_embeddings(
                self.datareader.word_dict)

        # pos embeddings
        # TODO ADD POS EMBEDDING VECTORS - WE NEED THEM
        self.pos_idx_mappings, self.idx_pos_mappings = self.init_pos_vocab(self.datareader.pos_dict)
        self.pos_vectors = None  # TODO

        # במודל הזה אנחנו לא נעשה באטצ'ינג ואנחנו לא נעשה את הריפוד
        # נעשה משהו שקול לעבודה עם באטצ'ים עלידי איזשהו טריק
        # אבל בלי שימוש בבאטצ'ים באמת
        # self.pad_idx = self.word_idx_mappings.get(PAD_TOKEN)

        # עבור מילים שלא ראיתי - המודל שלי ידע להתייחס אליהן אבל אני לא מכיר אותן
        self.unknown_idx = self.word_idx_mappings.get(UNKNOWN_TOKEN)
        self.word_vector_dim = self.word_vectors.size(-1)
        # self.sentence_lens = [len(sentence) for sentence in self.datareader.sentences]
        # משפטים שארוכים מהאורך הזה ייחתכו
        # self.max_seq_len = max(self.sentence_lens)
        # ממיר את המשפטים לדאטאסט - פונקציה חשובה
        self.sentences_dataset = self.convert_sentences_to_dataset(padding)

    def __len__(self):
        return len(self.sentences_dataset)

    def __getitem__(self, index):
        word_embed_idx, pos_embed_idx, sentence_len = self.sentences_dataset[index]
        return word_embed_idx, pos_embed_idx, sentence_len

    @staticmethod
    def init_word_embeddings(word_dict):
        glove = Vocab(Counter(word_dict), vectors="glove.6B.300d", specials=SPECIAL_TOKENS)
        return glove.stoi, glove.itos, glove.vectors

    def get_word_embeddings(self):
        return self.word_idx_mappings, self.idx_word_mappings, self.word_vectors

    # return idx mapping for POS tags
    # idx_pos_mappings - ['<root>', '<unk>', '#', '$', ....]
    # pos_idx_mappings - {'<root>': 0, '<unk>': 1, '#': 2, '$': 3, ..... }
    def init_pos_vocab(self, pos_dict):
        """
        :param pos_dict: {'DT': 17333, 'NNP': 5371, 'VBG': 5353....}
        :return: index mapping for POS tags
        """
        # pay attention we changed this a little bit - if everything's ok delete this comment
        idx_pos_mappings = sorted([token for token in SPECIAL_TOKENS])
        pos_idx_mappings = {pos: idx for idx, pos in enumerate(idx_pos_mappings)}

        for i, pos in enumerate(sorted(pos_dict.keys())):
            # pos_idx_mappings[str(pos)] = int(i)
            pos_idx_mappings[str(pos)] = int(i + len(SPECIAL_TOKENS))
            idx_pos_mappings.append(str(pos))
        print("idx_pos_mappings -", idx_pos_mappings)  # TODO DEL
        print("pos_idx_mappings -", pos_idx_mappings)  # TODO DEL
        return pos_idx_mappings, idx_pos_mappings

    def get_pos_vocab(self):
        return self.pos_idx_mappings, self.idx_pos_mappings

    def convert_sentences_to_dataset(self, padding):
        """
        returns a dictionary that contains all the input train sample
        :param padding: determine if we want padding to batch or not
        :return: dictionary as described
        """
        # מחזירה מילון שמכיל דוגמאות של אינפוט ,לייבל וגודל הבאטצ' -
        # מי שלא יעבוד עם באטצ' לא צריך את האחרון מביניהם
        sentence_word_idx_list = list()
        sentence_pos_idx_list = list()
        sentence_heads_list = list()  # TODO should we index the heads somehow??
        # sentence_len_list = list()

        # sentence_len_list = list()
        for sample_idx, sample in enumerate(self.datareader.D):
            words, tags, heads = sample
            words_idx_list = [self.word_idx_mappings[word] for word in words]
            pos_idx_list = [self.pos_idx_mappings[tag] for tag in tags]

            # for word, pos in zip:
            #     words_idx_list.append(self.word_idx_mappings.get(word))
            #     pos_idx_list.append(self.pos_idx_mappings.get(pos))
            # sentence_len = len(words_idx_list)

            # if padding:
            #     while len(words_idx_list) < self.max_seq_len:
            #         words_idx_list.append(self.word_idx_mappings.get(PAD_TOKEN))
            #         pos_idx_list.append(self.pos_idx_mappings.get(PAD_TOKEN))
            sentence_word_idx_list.append(torch.tensor(words_idx_list, dtype=torch.long, requires_grad=False))
            sentence_pos_idx_list.append(torch.tensor(pos_idx_list, dtype=torch.long, requires_grad=False))
            sentence_heads_list.append(torch.tensor(heads, dtype=torch.int, requires_grad=False))
            # sentence_len_list.append(sentence_len)

        # if padding:
        #     all_sentence_word_idx = torch.tensor(sentence_word_idx_list, dtype=torch.long)
        #     all_sentence_pos_idx = torch.tensor(sentence_pos_idx_list, dtype=torch.long)
        #     all_sentence_len = torch.tensor(sentence_len_list, dtype=torch.long, requires_grad=False)
        #     return TensorDataset(all_sentence_word_idx, all_sentence_pos_idx, all_sentence_len)

        return {i: sample_tuple for i, sample_tuple in enumerate(zip(sentence_word_idx_list,
                                                                     sentence_pos_idx_list,
                                                                     sentence_heads_list))}


# בשלב האימון לא ממש צריך לייצר עצים, אפשר להסתכל ישירות על הלוס - בעצם מה המודל אמר שהוא חושב בכל שלב
# בשלב ההסקה כן צריך ליצור עצים ואז לראות כמה המודל צדק על האבלואציה
class KiperwasserDependencyParser(nn.Module):
    def __init__(self, *args):
        super(KiperwasserDependencyParser, self).__init__()
        self.word_embedding = 0  # Implement embedding layer for words (can be new or pretrained - word2vec/glove)
        self.pos_embedding = 0  # Implement embedding layer for POS tags
        self.hidden_dim = self.word_embedding.embedding_dim + self.pos_embedding.embedding_dim
        self.encoder = 0  # Implement BiLSTM module which is fed with word+pos embeddings and outputs hidden representations
        self.edge_scorer = 0  # Implement a sub-module to calculate the scores for all possible edges in sentence dependency graph
        self.decoder = decode_mst  # This is used to produce the maximum spannning tree during inference
        self.loss_function = 0  # Implement the loss function described above

    def forward(self, sentence):
        word_idx_tensor, pos_idx_tensor, true_tree_heads = sentence

        # sentence = ['<root>', 'Mr', 'zibi', 'is', 'chairman']
        # true_tree_heads = [2, 3, 0, 3]
        # (2,1+0), (3,1+1) (0,1+2), (3,1+3)

        # Pass word_idx and pos_idx through their embedding layers

        # Concat both embedding outputs

        # Get Bi-LSTM hidden representation for each word+pos in sentence

        # Get score for each possible edge in the parsing graph, construct score matrix

        # Use Chu-Liu-Edmonds to get the predicted parse tree T' given the calculated score matrix

        # Calculate the negative log likelihood loss described above

        return loss, predicted_tree


def main():
    path_train = "train.labeled"
    word_dict, pos_dict = get_vocabs_counts([path_train])
    train = PosDataset(word_dict, pos_dict, path_train, padding=False, word_embeddings=None)
    train_dataloader = DataLoader(train, shuffle=True)

    path_test = "test.labeled"

    breakpoint()


if __name__ == "__main__":
    main()
