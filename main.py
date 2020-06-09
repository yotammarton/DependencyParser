from torchtext.vocab import Vocab
from torch.utils.data.dataset import Dataset, TensorDataset
from collections import Counter
from collections import defaultdict
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data.dataloader import DataLoader
import re
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
import time
import os
from chu_liu_edmonds import decode_mst

# Constants
ROOT_TOKEN = "<root>"  # TODO GAL this is root pos or root word?
UNKNOWN_TOKEN = "<unk>"  # TODO GAL according to the forum, its represents unknown POS?
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
            for line in f:
                if line != "\n":
                    splited_values = re.split('\t', line)
                    word = splited_values[1]
                    pos = splited_values[3]

                    word_dict[word] += 1
                    pos_dict[pos] += 1
    return word_dict, pos_dict


# returns {'The': 0, 'I': 1, 'Boeing': 2....}, {'DT': 0, 'NNP': 1, 'VBG': 2....}
def create_idx_dicts(word_dict, pos_dict):
    # TODO GAL maybe sparate to pos and words, because we will call the POS function in both cases of embeddings
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


class DataReader:
    def __init__(self, file, word_dict, pos_dict):  # call to readData
        self.file = file
        self.D = list()
        self.word_dict = word_dict
        self.pos_dict = pos_dict
        self.__readData__()

    def __readData__(self):
        """main reader function which also populates the class data structures"""
        with open(self.file) as f:
            sentence, tags, heads = [ROOT_TOKEN], [ROOT_TOKEN], []
            for line in f:
                if line == "\n":
                    if heads:
                        self.D.append((sentence, tags, heads))
                    sentence, tags, heads = [ROOT_TOKEN], [ROOT_TOKEN], []
                else:
                    splited_values = re.split('\t', line)
                    # m = int(splited_values[0])
                    h = int(splited_values[6])
                    word = splited_values[1]
                    pos = splited_values[3]

                    sentence.append(word)
                    tags.append(pos)
                    heads.append(h)

                    # e.g.
                    # ['<root>', 'It', 'has', 'no', 'bearing', 'on', 'our', 'work', 'force', 'today', '.'] len = 11
                    # ['<root>', 'PRP', 'VBZ', 'DT', 'NN', 'IN', 'PRP$', 'NN', 'NN', 'NN', '.']             len = 11
                    # ['2', '0', '4', '2', '4', '8', '8', '5', '8', '2']                                   len = 10

    def get_num_sentences(self):
        """returns num of sentences in data"""
        return len(self.D)


class DependencyDataset(Dataset):
    def __init__(self, word_dict, pos_dict, path: str, word_embd_dim, pos_embd_dim,
                 padding=False, use_pre_trained=True, pre_trained_vectors_name: str = None):
        super().__init__()
        self.file = path
        self.datareader = DataReader(self.file, word_dict, pos_dict)
        self.vocab_size = len(self.datareader.word_dict)
        if use_pre_trained:  # pre-trained word embeddings
            self.word_idx_mappings, self.idx_word_mappings, self.pre_trained_word_vectors = \
                self.init_word_embeddings(self.datareader.word_dict, pre_trained_vectors_name)
        else:
            self.word_idx_mappings = create_idx_dicts(word_dict, pos_dict)[0]
            self.idx_word_mappings = list(self.word_idx_mappings.keys())
            # self.word_embedding = nn.Embedding(word_vocab_size, word_embedding_dim)
            self.word_vectors = nn.Embedding(len(self.word_idx_mappings), word_embd_dim)
            # TODO GAL its attribute that doesn't init to nothing if we are with pertained, maybe change the name?

            # TODO YOTAM: Possible change to this 'else' statement:
            # that's because:
            # 1. now we can control the SPECIAL_TOKENS and add them also when having nn.Embedding
            # 2. now we can control the min_freq
            """
            self.word_idx_mappings = create_idx_dicts(word_dict, pos_dict)[0]
            self.idx_word_mappings = list(self.word_idx_mappings.keys())
            words_embeddings_tensor = nn.Embedding(len(self.word_idx_mappings), word_embd_dim).weight.data
            vocab = Vocab(Counter(word_dict), vectors=None, specials=SPECIAL_TOKENS, min_freq=0)
            vocab.set_vectors(stoi=self.word_idx_mappings, vectors=words_embeddings_tensor, dim=word_embd_dim)
            # take all 3 attributes like in the pre-trained part 
            self.word_idx_mappings, self.idx_word_mappings, self.word_vectors = \
                vocab.stoi, vocab.itos, vocab.vectors
            """

        # pos embeddings
        self.pos_idx_mappings, self.idx_pos_mappings = self.init_pos_vocab()
        self.pos_vectors = nn.Embedding(len(self.pos_idx_mappings), pos_embd_dim)

        self.unknown_idx = self.word_idx_mappings.get(UNKNOWN_TOKEN)
        # TODO index -1 will always be index 1? because it always be 2 dimensional?
        self.word_vector_dim = self.pre_trained_word_vectors.size(-1) if use_pre_trained \
            else self.word_vectors.embedding_dim  # TODO GAL this is attribute of nn.embedding function
        self.sentences_dataset = self.convert_sentences_to_dataset()

    def __len__(self):
        return len(self.sentences_dataset)

    def __getitem__(self, index):
        word_embed_idx, pos_embed_idx, sentence_len = self.sentences_dataset[index]
        return word_embed_idx, pos_embed_idx, sentence_len

    @staticmethod
    def init_word_embeddings(word_dict, vectors: str):
        if vectors not in ['charngram.100d',
                           'fasttext.en.300d',
                           'fasttext.simple.300d',
                           'glove.42B.300d',
                           'glove.840B.300d',
                           'glove.twitter.27B.25d',
                           'glove.twitter.27B.50d',
                           'glove.twitter.27B.100d',
                           'glove.twitter.27B.200d',
                           'glove.6B.50d',
                           'glove.6B.100d',
                           'glove.6B.200d',
                           'glove.6B.300d']:
            raise ValueError("pre-trained embedding vectors not found")
        glove = Vocab(Counter(word_dict), vectors=vectors, specials=SPECIAL_TOKENS, min_freq=0)
        # TODO GAL what the glove do with the specials? what the counter(word_dict) takes?
        # TODO YOTAM: word_dict is already a counter so the Counter(word_dict) is the same dict with keys
        #  and values but different object (not sure if necessary but we can leave it like that)
        # TODO YOTAM I think we should make something similar if we train by ourselves
        return glove.stoi, glove.itos, glove.vectors

    def get_word_embeddings(self):
        return self.word_idx_mappings, self.idx_word_mappings, self.word_vectors

    # return idx mapping for POS tags
    # pos_idx_mappings - {'<root>': 0, '<unk>': 1, '#': 2, '$': 3, ..... }
    # idx_pos_mappings - ['<root>', '<unk>', '#', '$', ....]
    def init_pos_vocab(self):
        """
        :param pos_dict: {'DT': 17333, 'NNP': 5371, 'VBG': 5353....}
        :return: index mapping for POS tags
        """
        idx_pos_mappings = sorted([token for token in SPECIAL_TOKENS])
        pos_idx_mappings = {pos: idx for idx, pos in enumerate(idx_pos_mappings)}

        for i, pos in enumerate(sorted(self.datareader.pos_dict.keys())):
            # pos_idx_mappings[str(pos)] = int(i)
            pos_idx_mappings[str(pos)] = int(i + len(SPECIAL_TOKENS))
            idx_pos_mappings.append(str(pos))
        return pos_idx_mappings, idx_pos_mappings

    def get_pos_vocab(self):
        return self.pos_idx_mappings, self.idx_pos_mappings

    def convert_sentences_to_dataset(self):
        """
        returns a dictionary that contains all the input train sample
        :return: dictionary as described
        """
        sentence_word_idx_list = list()
        sentence_pos_idx_list = list()
        sentence_heads_list = list()  # TODO should we index the heads somehow??

        for sample_idx, sample in enumerate(self.datareader.D):
            words, tags, heads = sample
            words_idx_list = [self.word_idx_mappings[word] for word in words]
            pos_idx_list = [self.pos_idx_mappings[tag] for tag in tags]

            # we don't want to activate grads for the indexes because these are not parameters
            sentence_word_idx_list.append(torch.tensor(words_idx_list, dtype=torch.long, requires_grad=False))
            sentence_pos_idx_list.append(torch.tensor(pos_idx_list, dtype=torch.long, requires_grad=False))
            sentence_heads_list.append(torch.tensor(heads, dtype=torch.long, requires_grad=False))

        return {i: sample_tuple for i, sample_tuple in
                enumerate(zip(sentence_word_idx_list,  # its just indexes. next phase will be convert it to embeddings
                              sentence_pos_idx_list,
                              sentence_heads_list))}


"""Basic Model"""


class KiperwasserDependencyParser(nn.Module):
    def __init__(self, dataset: DependencyDataset, hidden_dim, MLP_inner_dim, dropout=0.0, use_pre_trained=True):
        """
        :param dataset: dataset for training
        :param hidden_dim: size of hidden dim (output of LSTM, aka v_i)
        :param MLP_inner_dim: controls the matrix size W1 (MLP_inner_dim x 500) and so that the length of W2 vector
        :param dropout:
        :param use_pre_trained: bool.
        """
        super(KiperwasserDependencyParser, self).__init__()
        self.dataset = dataset
        self.dropout_p = dropout
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Implement embedding layer for words (can be new or pertained - word2vec/glove)
        if use_pre_trained:  # use pre trained vectors
            # this is not matrix of embeddings. its function that gets indexes and return embeddings
            self.word_embedding = nn.Embedding.from_pretrained(dataset.pre_trained_word_vectors, freeze=False)
        else:
            self.word_embedding = dataset.word_vectors

        # Implement embedding layer for POS tags
        self.pos_embedding = dataset.pos_vectors

        self.input_dim = self.word_embedding.embedding_dim + self.pos_embedding.embedding_dim  # input for LSTM

        # Dropout layer
        if dropout:
            self.dropout = nn.Dropout(p=dropout)

        # Implement BiLSTM module which is fed with word+pos embeddings and outputs hidden representations
        self.encoder = nn.LSTM(input_size=self.input_dim, hidden_size=hidden_dim,
                               num_layers=1, bidirectional=True, batch_first=True)

        # Implement a sub-module to calculate the scores for all possible edges in sentence dependency graph
        # MLP(x) = W2 * tanh(W1 * x + b1) + b2
        # W1 - Matrix (MLP_inner_dim x 500) || W2, b1 - Vectors (MLP_inner_dim) || b2 - Scalar
        # TODO possible change to (500, dim) , (dim, 1) - we can control the dimension
        self.edge_scorer = nn.Sequential(
            # W1 * x + b1
            nn.Linear(500, MLP_inner_dim),
            # tanh(W1 * x + b1)
            nn.Tanh(),
            # W2 * tanh(W1 * x + b1) + b2
            nn.Linear(MLP_inner_dim, 1)
        )
        # TODO DEL
        # how to access weights:
        # self.edge_scorer[i].weight || i in [1, 2, 3]

    def forward(self, sample, dropout):  # this is required function. can't change its name
        # TODO added 'dropout' parameter so we will only dropout for train and not for test
        word_idx_tensor, pos_idx_tensor, true_tree_heads = sample
        original_sentence_in_words = [self.dataset.idx_word_mappings[w_idx] for w_idx in word_idx_tensor[0]]
        # TODO GAL what is index 0? w is confusing because its index and not word

        # Pass word_idx and pos_idx through their embedding layers
        # size = [batch_size, seq_length, word_dim]
        word_embeddings = self.word_embedding(word_idx_tensor.to(self.device))
        # size = [batch_size, seq_length, pos_dim]
        pos_embeddings = self.pos_embedding(pos_idx_tensor.to(self.device))

        # Concat both embedding outputs: combine both word_embeddings + pos_embeddings
        # size = [batch_size, seq_length, word_dim + pos_dim]
        word_pos_embeddings = torch.cat((word_embeddings, pos_embeddings), dim=2)

        # Dropout
        # 1. first condition - if added dropout to init
        # 2. second condition - specify if we wish to make dropout in this current forward pass (depends if train/test)
        if self.dropout_p and dropout:
            word_pos_embeddings = self.dropout(word_pos_embeddings)

        # Get Bi-LSTM hidden representation for each word+pos in sentence
        # size  = [batch_size, seq_length, 2*hidden_dim]
        # encoder wants to get tensor. it is not defined in our code but that's how NN works
        lstm_out, _ = self.encoder(word_pos_embeddings)

        # Get score for each possible edge in the parsing graph, construct score matrix
        # n = original sentence length + 1
        n = lstm_out.shape[1]

        MLP_scores_mat = torch.zeros((n, n))  # modifiers are rows, heads are cols
        for m in range(n):
            modifier_vector = lstm_out.data[0][m]
            for h in range(n):
                head_vector = lstm_out.data[0][h]  # 0 because the batch size = 1 always in our case
                h_m_concat = torch.cat((head_vector, modifier_vector))
                MLP_score = self.edge_scorer(h_m_concat)
                MLP_scores_mat[m][h] = MLP_score

        return MLP_scores_mat


def train_kiperwasser_parser(model, train_dataloader, test_dataloader, epochs, learning_rate, weight_decay):
    start = time.time()
    total_test_time = 0

    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda:0" if use_cuda else "cpu")

    if use_cuda:
        model.cuda()

    # Define the loss function as the Negative Log Likelihood loss (NLLLoss)
    loss_function = nn.NLLLoss(ignore_index=-1, reduction='mean')  # TODO check ignore index

    # We will be using a simple SGD optimizer to minimize the loss function
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    # TODO optimize learning rate 'lr'
    acumulate_grad_steps = 50  # This is the actual batch_size, while we officially use batch_size=1

    # Training start
    print("Training Started")
    train_accuracy_list, train_loss_list = [], []
    test_accuracy_list, test_loss_list = [], []

    model.zero_grad()
    for epoch in range(epochs):
        train_acc = 0  # to keep track of accuracy
        train_loss = 0  # to keep track of the loss value
        mst_trees_calculated = 0  # keep track of amount of trees calculated to plot the accuracy graph
        i = 0  # keep track of samples processed
        for batch_idx, input_data in enumerate(train_dataloader):
            i += 1
            # size = [sentence_length + 1, sentence_length + 1]
            MLP_scores_mat = model(input_data, dropout=True)  # forward activated inside

            gold_heads = input_data[2]

            # concat -1 to true heads, we ignore this target value of -1
            target = torch.cat((torch.tensor([-1]), gold_heads[0]))

            # calculate negative log likelihood loss
            # log softmax over the rows (modifiers in rows)
            loss = loss_function(F.log_softmax(MLP_scores_mat, dim=1), target)
            loss = loss / acumulate_grad_steps
            loss.backward()
            train_loss += loss.item()

            # calculated sampled tress - only for accuracy calculations during train
            if i % (acumulate_grad_steps / 2) == 0:
                # res=[-1, 5, 0, , 4] - always -1 at the beginning because it's '<root>' token in every sentence's start
                predicted_tree = decode_mst(MLP_scores_mat.detach().numpy().T, length=MLP_scores_mat.shape[0],
                                            has_labels=False)[0]

                train_acc += sum(gold_heads[0].numpy() == predicted_tree[1:]) / len(gold_heads[0])
                mst_trees_calculated += 1

            # perform optimization step
            if i % acumulate_grad_steps == 0 or i == len(train_dataloader):
                optimizer.step()
                model.zero_grad()

        train_loss = acumulate_grad_steps * train_loss / len(train_dataloader)
        train_acc = train_acc / mst_trees_calculated
        train_loss_list.append(train_loss)
        train_accuracy_list.append(train_acc)

        start_test_time = time.time()
        # calculate test accuracy - TODO skip the next 3 lines if no need to know the test accuracy during training
        test_acc, test_loss = evaluate(model, test_dataloader)
        test_accuracy_list.append(test_acc)
        test_loss_list.append(test_loss)
        stop_test_time = time.time()
        total_test_time += stop_test_time - start_test_time

        print(f"Epoch {epoch + 1} Completed,\tTrain Loss {train_loss}\t Train Accuracy: {train_acc}\t "
              f"Test Loss {test_loss}\t Test Accuracy: {test_acc}")

        # print time for the end of epoch
        print(f"Epoch {epoch + 1} Time "
              f"{time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(int(time.time())))}")

    stop = time.time()

    total_train_time = stop - start - total_test_time

    print(f'\n\n\ntotal_train_time = {int(total_train_time)} SECS \t total_test_time = {int(total_test_time)} SECS')
    return train_accuracy_list, train_loss_list, test_accuracy_list, test_loss_list


"""Advanced Model - GoldMart = Goldstein-Martin"""


class GoldMartDependencyParser(nn.Module):
    pass


def train_goldmart_parser():
    pass


def evaluate(model, dataloader):
    acc = 0
    loss_value = 0

    # tell the model not to learn
    with torch.no_grad():
        loss_function = nn.NLLLoss(ignore_index=-1, reduction='mean')  # TODO check ignore index
        for batch_idx, input_data in enumerate(dataloader):
            MLP_scores_mat = model(input_data, dropout=False)
            gold_heads = input_data[2]

            # concat -1 to true heads, we ignore this target value of -1
            target = torch.cat((torch.tensor([-1]), gold_heads[0]))

            # calculate negative log likelihood loss
            # log softmax over the rows (modifiers in rows)
            loss = loss_function(F.log_softmax(MLP_scores_mat, dim=1), target)
            loss_value += loss.item()  # TODO change this to take only the value itself and not reference !!!!

            # Use Chu-Liu-Edmonds to get the predicted parse tree T' given the calculated score matrix
            # res=[-1, 5, 0, , 4] - always -1 at the beginning because it's '<root>' token in every sentence's start
            predicted_tree = decode_mst(MLP_scores_mat.detach().numpy().T, length=MLP_scores_mat.shape[0],
                                        has_labels=False)[0]

            acc += sum(gold_heads[0].numpy() == predicted_tree[1:]) / len(gold_heads[0])
        acc = acc / len(dataloader)
        loss_value = loss_value / len(dataloader)

    return acc, loss_value


def plot_graphs(train_accuracy_list, train_loss_list, test_accuracy_list, test_loss_list):
    pass


def main():
    word_embd_dim = 100  # if using pre-trained choose word_embd_dim from [50, 100, 200, 300]
    pos_embd_dim = 25
    hidden_dim = 125
    MLP_inner_dim = 500
    epochs = 15
    learning_rate = 0.01
    dropout = 0.0
    weight_decay = 0.1
    use_pre_trained = False
    vectors = 'glove.6B.300d' if use_pre_trained else ''
    path_train = "train.labeled"
    path_test = "test.labeled"

    run_description = f"first run for the KiperwasserDependencyParser + Weight Decay\n" \
                      f"-------------------------------------------------------------------------------------------\n" \
                      f"word_embd_dim = {word_embd_dim}\n" \
                      f"pos_embd_dim = {pos_embd_dim}\n" \
                      f"hidden_dim = {hidden_dim}\n" \
                      f"MLP_inner_dim = {MLP_inner_dim}\n" \
                      f"epochs = {epochs}\n" \
                      f"learning_rate = {learning_rate}\n" \
                      f"dropout = {dropout}\n" \
                      f"weight_decay = {weight_decay}\n" \
                      f"use_pre_trained = {use_pre_trained}\n" \
                      f"vectors = {vectors}\n" \
                      f"path_train = {path_train}\n" \
                      f"path_test = {path_test}\n"

    current_machine_date_time = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(int(time.time())))
    print(f"{current_machine_date_time}\n"
          f"{run_description}")

    path_to_save_model = os.path.join('saved_models', f'model {current_machine_date_time}.pt')

    """TRAIN DATA"""
    train_word_dict, train_pos_dict = get_vocabs_counts([path_train])
    train = DependencyDataset(train_word_dict, train_pos_dict, path_train, word_embd_dim, pos_embd_dim,
                              padding=False, use_pre_trained=use_pre_trained, pre_trained_vectors_name=vectors)
    train_dataloader = DataLoader(train, shuffle=True)
    model = KiperwasserDependencyParser(train, hidden_dim, MLP_inner_dim, dropout, use_pre_trained=use_pre_trained)

    """TEST DATA"""
    test_word_dict, test_pos_dict = get_vocabs_counts([path_test])

    test = DependencyDataset(test_word_dict, test_pos_dict, path_test, word_embd_dim, pos_embd_dim,
                             padding=False, use_pre_trained=use_pre_trained, pre_trained_vectors_name=vectors)
    test_dataloader = DataLoader(test, shuffle=False)

    """TRAIN THE PARSER ON TRAIN DATA"""
    train_accuracy_list, train_loss_list, test_accuracy_list, test_loss_list = \
        train_kiperwasser_parser(model, train_dataloader, test_dataloader, epochs, learning_rate, weight_decay)

    print(f'\ntrain_accuracy_list = {train_accuracy_list}'
          f'\ntrain_loss_list = {train_loss_list}'
          f'\ntest_accuracy_list = {test_accuracy_list}'
          f'\ntest_loss_list = {test_loss_list}')

    """SAVE MODEL"""
    torch.save(model.state_dict(), path_to_save_model.replace(':', '-'))

    """PLOT GRAPHS"""
    # plot_graphs(train_accuracy_list, train_loss_list, test_accuracy_list, test_loss_list)

    # """EVALUATE ON TEST DATA"""
    # evaluate(model, test_dataloader)


if __name__ == "__main__":
    main()
