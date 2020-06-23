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
import matplotlib.pyplot as plt
import time
import os
from chu_liu_edmonds import decode_mst

# Constants
ROOT_TOKEN = "<root_token>"
ROOT_POS = "<root_pos>"
UNKNOWN_TOKEN = "<unk_token>"
SPECIAL_TOKENS = [ROOT_TOKEN, UNKNOWN_TOKEN]


# returns {'The': 15374, 'I': 1556, 'Boeing': 85....}, {'DT': 17333, 'NNP': 5371, 'VBG': 5353....}
def get_vocabs_counts(list_of_paths):
    """
        creates dictionary with number of appearances (counts) of each word and each tag
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
            sentence, tags, heads = [ROOT_TOKEN], [ROOT_POS], []
            for line in f:
                if line == "\n":
                    if heads:
                        self.D.append((sentence, tags, heads))
                    sentence, tags, heads = [ROOT_TOKEN], [ROOT_POS], []
                else:
                    splited_values = re.split('\t', line)
                    # m = int(splited_values[0])
                    h = int(splited_values[6]) if splited_values[6] != '_' else '_'
                    word = splited_values[1]
                    pos = splited_values[3]

                    sentence.append(word)
                    tags.append(pos)
                    heads.append(h)

                    # e.g.
                    # ['<root_token>', 'It', 'has', 'no', 'on', 'our', 'work', 'force', 'today', '.'] len = 10
                    # ['<root_pos>', 'PRP', 'VBZ', 'DT', 'NN', 'PRP$', 'NN', 'NN', 'NN', '.']         len = 10
                    # ['2', '0', '4', '2', '4', '8', '8', '5', '2']                                   len = 9

    def get_num_sentences(self):
        """returns num of sentences in data"""
        return len(self.D)


class DependencyDataset(Dataset):
    def __init__(self, path: str, word_dict=None, pos_dict=None, word_embd_dim=None, pos_embd_dim=None,
                 test=None, use_pre_trained=True, pre_trained_vectors_name: str = None, min_freq=1, comp_mode=False):
        """
        :param path: path to train / test file
        :param word_dict: defaultdict(<class 'int'>, {'Pierre': 1, 'Vinken': 2, ',': 6268,...}
        :param pos_dict: defaultdict(<class 'int'>, {'NNP': 11837, ',': 6270, 'CD': 4493,...}
        :param word_embd_dim: dimension of word embedding
        :param pos_embd_dim: dimension of pos embedding
        :param test: if False / None we train vectors (or use-pertained).
                     else should be a list train.word_idx_mappings, train.pos_idx_mappings
        :param use_pre_trained: True / False
        :param pre_trained_vectors_name: What pre-trained vectors to use
        """
        super().__init__()
        self.file = path
        self.datareader = DataReader(self.file, word_dict, pos_dict)
        self.vocab_size = len(self.datareader.word_dict)
        self.comp_mode = comp_mode
        if test:
            # no need to train vectors or create them, and also not vocabulary
            # that's because we use the vectors and vocabulary from train
            self.word_idx_mappings = test[0]
            self.pos_idx_mappings = test[1]
            self.sentences_dataset = self.convert_sentences_to_dataset()

        else:  # training
            if use_pre_trained:  # pre-trained word embeddings
                self.word_idx_mappings, self.idx_word_mappings, self.word_vectors = \
                    self.init_word_embeddings(self.datareader.word_dict, pre_trained_vectors_name)
            else:
                # create Vocab variable just for the ease of using the special tokens and the other nice features
                # like it will create the word_idx_mapping by itself
                vocab = Vocab(Counter(word_dict), vectors=None, specials=SPECIAL_TOKENS, min_freq=min_freq)

                # set rand vectors and get the weights (the vector embeddings themselves)
                words_embeddings_tensor = nn.Embedding(len(vocab.stoi), word_embd_dim).weight.data
                vocab.set_vectors(stoi=vocab.stoi, vectors=words_embeddings_tensor, dim=word_embd_dim)
                # take all 3 attributes like in the pre-trained part
                self.word_idx_mappings, self.idx_word_mappings, self.word_vectors = \
                    vocab.stoi, vocab.itos, vocab.vectors

            # pos embeddings
            self.pos_idx_mappings, self.idx_pos_mappings = self.init_pos_vocab()
            self.pos_vectors = nn.Embedding(len(self.pos_idx_mappings), pos_embd_dim)

            self.word_vector_dim = self.word_vectors.size(-1)
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
        glove = Vocab(Counter(word_dict), vectors=vectors, specials=SPECIAL_TOKENS, min_freq=1)
        return glove.stoi, glove.itos, glove.vectors

    # return idx mapping for POS tags
    # pos_idx_mappings - {'<root_pos>': 0, '#': 1, '$': 2, "''": 3, ...}
    # idx_pos_mappings - ['<root_pos>', '#', '$', "''", ... ]
    def init_pos_vocab(self):
        """
        :return: index mapping for POS tags
        """
        idx_pos_mappings = [ROOT_POS]
        pos_idx_mappings = {pos: idx for idx, pos in enumerate(idx_pos_mappings)}

        for i, pos in enumerate(sorted(self.datareader.pos_dict.keys())):
            pos_idx_mappings[str(pos)] = int(i + 1)  # +1 for <root_pos>
            idx_pos_mappings.append(str(pos))
        return pos_idx_mappings, idx_pos_mappings

    def convert_sentences_to_dataset(self):
        """
        returns a dictionary that contains all the input train sample
        :return: dictionary as described
        """
        sentence_word_idx_list = list()
        sentence_pos_idx_list = list()
        sentence_heads_list = list()

        for sample_idx, sample in enumerate(self.datareader.D):
            words, tags, heads = sample
            words_idx_list = [self.word_idx_mappings[word] if word in self.word_idx_mappings
                              else self.word_idx_mappings[UNKNOWN_TOKEN] for word in words]

            # TODO try to see if this helps
            # for word in words:
            #     if word not in self.word_idx_mappings and \
            #         (word[0].lower() + word[1:] in self.word_idx_mappings or
            #          word[0].upper() + word[1:] in self.word_idx_mappings):
            #         print(self.file, word)

            pos_idx_list = [self.pos_idx_mappings[tag] for tag in tags]

            # we don't want to activate grads for the indexes because these are not parameters
            sentence_word_idx_list.append(torch.tensor(words_idx_list, dtype=torch.long, requires_grad=False))
            sentence_pos_idx_list.append(torch.tensor(pos_idx_list, dtype=torch.long, requires_grad=False))
            if self.comp_mode:  # no heads
                sentence_heads_list.append([])
            else:
                sentence_heads_list.append(torch.tensor(heads, dtype=torch.long, requires_grad=False))

        return {i: sample_tuple for i, sample_tuple in
                enumerate(zip(sentence_word_idx_list,  # its just indexes. next phase will be convert it to embeddings
                              sentence_pos_idx_list,
                              sentence_heads_list))}


"""Basic Model"""


class KiperwasserDependencyParser(nn.Module):
    def __init__(self, dataset: DependencyDataset, hidden_dim, MLP_inner_dim, BiLSTM_layers, dropout_layers):
        """
        :param dataset: dataset for training
        :param hidden_dim: size of hidden dim (output of LSTM, aka v_i)
        :param MLP_inner_dim: controls the matrix size W1 (MLP_inner_dim x 500) and so that the length of W2 vector
        :param dropout_layers: in between layers (doc: https://pytorch.org/docs/master/generated/torch.nn.LSTM.html)
        """
        super(KiperwasserDependencyParser, self).__init__()
        self.dataset = dataset
        self.dropout_layers_p = dropout_layers
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Implement embedding layer for words (can be new or pertained - word2vec/glove)
        # this is not matrix of embeddings. its function that gets indexes and return embeddings
        self.word_embedding = nn.Embedding.from_pretrained(dataset.word_vectors, freeze=False)

        # Implement embedding layer for POS tags
        self.pos_embedding = dataset.pos_vectors

        self.input_dim = self.word_embedding.embedding_dim + self.pos_embedding.embedding_dim  # input for LSTM

        # Implement BiLSTM module which is fed with word+pos embeddings and outputs hidden representations
        self.encoder = nn.LSTM(input_size=self.input_dim, hidden_size=hidden_dim,
                               num_layers=BiLSTM_layers, dropout=dropout_layers, bidirectional=True, batch_first=True)

        # Implement a sub-module to calculate the scores for all possible edges in sentence dependency graph
        # MLP(x) = W2 * tanh(W1 * x + b1) + b2
        # W1 - Matrix (MLP_inner_dim x 4 * hidden_dim) || W2, b1 - Vectors (MLP_inner_dim) || b2 - Scalar
        self.edge_scorer = nn.Sequential(
            # W1 * x + b1
            nn.Linear(4 * hidden_dim, MLP_inner_dim),
            # tanh(W1 * x + b1)
            nn.Tanh(),
            # W2 * tanh(W1 * x + b1) + b2
            nn.Linear(MLP_inner_dim, 1)
        )

    def forward(self, sample):  # this is required function. can't change its name
        word_idx_tensor, pos_idx_tensor = sample[0], sample[1]
        original_sentence_in_words = [self.dataset.idx_word_mappings[w_idx] for w_idx in word_idx_tensor[0]]

        # Pass word_idx and pos_idx through their embedding layers
        # size = [batch_size, seq_length, word_dim]
        word_embeddings = self.word_embedding(word_idx_tensor.to(self.device))
        # size = [batch_size, seq_length, pos_dim]
        pos_embeddings = self.pos_embedding(pos_idx_tensor.to(self.device))

        # Concat both embedding outputs: combine both word_embeddings + pos_embeddings
        # size = [batch_size, seq_length, word_dim + pos_dim]
        word_pos_embeddings = torch.cat((word_embeddings, pos_embeddings), dim=2)

        # Get Bi-LSTM hidden representation for each word+pos in sentence
        # size  = [batch_size, seq_length, 2*hidden_dim]
        # encoder wants to get tensor. it is not defined in our code but that's how NN works
        lstm_out, _ = self.encoder(word_pos_embeddings)
        n = lstm_out.shape[1]

        # source: https://discuss.pytorch.org/t/how-to-create-a-combination-of-concatenations-with-two-tensors/28709
        heads = lstm_out[0].unsqueeze(0)
        modifiers = lstm_out[0].unsqueeze(1)
        heads_tmp = heads.repeat(lstm_out[0].shape[0], 1, 1)
        modifiers_tmp = modifiers.repeat(1, lstm_out[0].shape[0], 1)
        heads_modifier_cat = torch.cat([heads_tmp, modifiers_tmp], -1)
        heads_modifier_cat = heads_modifier_cat.view(-1, heads_modifier_cat.shape[-1])

        MLP_scores_mat_new = self.edge_scorer(heads_modifier_cat).view(n, n)
        return MLP_scores_mat_new


def train_kiperwasser_parser(model, train_dataloader, test_dataloader, epochs, learning_rate, weight_decay, alpha,
                             path_to_save_model):
    start = time.time()
    total_evaluate_time = 0
    max_test_accuracy = 0

    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda:0" if use_cuda else "cpu")

    if use_cuda:
        model.cuda()

    # Define the loss function as the Negative Log Likelihood loss (NLLLoss)
    loss_function = nn.NLLLoss(ignore_index=-1, reduction='mean')

    # We will be using a simple SGD optimizer to minimize the loss function
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    acumulate_grad_steps = 50  # This is the actual batch_size, while we officially use batch_size=1

    # Training start
    print("Training Started")
    train_accuracy_list, train_loss_list = [], []
    test_accuracy_list, test_loss_list = [], []

    model.zero_grad()
    for epoch in range(epochs):
        i = 0  # keep track of samples processed
        data = list(enumerate(train_dataloader))  # save the data so we can modify it to introduce word-dropout
        word_dropout(model, data, alpha=alpha)  # changes data in-place

        for batch_idx, input_data in data:
            i += 1
            # size = [sentence_length + 1, sentence_length + 1]
            MLP_scores_mat = model(input_data)  # forward activated inside

            gold_heads = input_data[2]

            # concat -1 to true heads, we ignore this target value of -1
            target = torch.cat((torch.tensor([-1]), gold_heads[0])).to(device)

            # calculate negative log likelihood loss
            # log softmax over the rows (modifiers in rows)
            loss = loss_function(F.log_softmax(MLP_scores_mat, dim=1), target)
            loss = loss / acumulate_grad_steps
            loss.backward()

            # perform optimization step
            if i % acumulate_grad_steps == 0 or i == len(train_dataloader):
                optimizer.step()
                model.zero_grad()

        # evaluate at the end of the epoch
        start_evaluate_time = time.time()
        # train
        train_acc, train_loss = evaluate(model, train_dataloader)
        train_loss_list.append(train_loss)
        train_accuracy_list.append(train_acc)

        # test
        test_acc, test_loss = evaluate(model, test_dataloader)
        test_accuracy_list.append(test_acc)
        test_loss_list.append(test_loss)

        stop_evaluate_time = time.time()
        total_evaluate_time += stop_evaluate_time - start_evaluate_time

        # Save model if test accuracy is better than before
        if test_acc > max_test_accuracy:
            max_test_accuracy = test_acc
            torch.save(model.state_dict(),
                       path_to_save_model.replace(':', '-') + f'acc = {test_acc} epoch = {epoch + 1}.pt')

        print(f"Epoch {epoch + 1} Completed,\tTrain Loss {train_loss}\t Train Accuracy: {train_acc}\t "
              f"Test Loss {test_loss}\t Test Accuracy: {test_acc}")

        # print time for the end of epoch
        print(f"Epoch {epoch + 1} Time "
              f"{time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(int(time.time())))}")

    stop = time.time()

    total_train_time = stop - start - total_evaluate_time

    print(f'\n\n\ntotal_train_time = {int(total_train_time)} SECS \t total_evaluate_time (train and test) = '
          f'{int(total_evaluate_time)} SECS')
    return train_accuracy_list, train_loss_list, test_accuracy_list, test_loss_list


def word_dropout(model, data, alpha=0.25):
    """
    During training, we employ a variant of word dropout (Iyyer et al., 2015), and replace a word with
    the unknown-word symbol with probability that is inversely proportional to the frequency of the word.
    A word w appearing #(w) times in the training corpus is replaced with the unknown symbol with probability
    p(w) = alpha / (tf(w) + alpha). where tf(w) is the number of appearances of term w in the train corpus
    :param model: nn.Module
    :param data: the train data for the current epoch
    :param alpha: hyper parameter
    :return: None. changes  'data' (only for the current epoch).
    """
    word_counter_dict = Counter(model.dataset.datareader.word_dict)
    idx_word_dict = {v: k for k, v in model.dataset.word_idx_mappings.items()}
    idx_dropout_prob_dict = dict()
    for idx, word in idx_word_dict.items():
        # we will assign the probability of each word (it's index) to be dropped
        if word not in word_counter_dict:  # e.g. any special token (in practice - only for <root_token>)
            idx_dropout_prob_dict[idx] = 0
        else:  # word is a word in our dictionary of train words
            idx_dropout_prob_dict[idx] = alpha / (word_counter_dict[word] + alpha)

    for _, sample_tensor in data:
        sentence_tensor = sample_tensor[0][0]
        sentence_dropout_probabilities = torch.tensor([idx_dropout_prob_dict[int(idx)]
                                                       for idx in sentence_tensor], dtype=torch.float32)
        n = len(sentence_tensor)
        # based on every word-drop out probabilities create Bernoulli vector
        bernoulli_toss = torch.bernoulli(sentence_dropout_probabilities)  # 1 = dropout, 0 = no dropout
        # tensor with [1, 1, 1, 1, ...] (which is the index of '<unk_token>')
        unk_token_tensor = torch.empty(n, dtype=torch.int64). \
            fill_(model.dataset.word_idx_mappings[UNKNOWN_TOKEN])

        sample_tensor[0][0] = torch.where(bernoulli_toss == 0,  # condition
                                          sentence_tensor,  # if condition true take element from this tensor
                                          unk_token_tensor)  # else take from this tensor


def evaluate(model, dataloader):
    acc = 0
    loss_value = 0

    # tell the model not to learn
    with torch.no_grad():
        loss_function = nn.NLLLoss(ignore_index=-1, reduction='mean')
        for batch_idx, input_data in enumerate(dataloader):
            MLP_scores_mat = model(input_data)
            gold_heads = input_data[2]

            # concat -1 to true heads, we ignore this target value of -1
            target = torch.cat((torch.tensor([-1]), gold_heads[0])).to(model.device)

            # calculate negative log likelihood loss
            # log softmax over the rows (modifiers in rows)
            loss = loss_function(F.log_softmax(MLP_scores_mat, dim=1), target)
            loss_value += loss.item()

            # Use Chu-Liu-Edmonds to get the predicted parse tree T' given the calculated score matrix
            # res=[-1, 5, 0, , 4] - always -1 at the beginning because it's '<root>' token in every sentence's start
            predicted_tree = decode_mst(MLP_scores_mat.data.cpu().numpy().T, length=MLP_scores_mat.shape[0],
                                        has_labels=False)[0]

            acc += sum(gold_heads[0].numpy() == predicted_tree[1:]) / len(gold_heads[0])
        acc = acc / len(dataloader)
        loss_value = loss_value / len(dataloader)

    return acc, loss_value


def tag_file_save_output(model, dataloader, original_unlabeled_file, result_path):
    # read the whole file we wish to tag to list of lines
    with open(original_unlabeled_file) as file_to_tag:
        lines = file_to_tag.readlines()

    # inference and write output to file in the wanted format
    with open(result_path, 'w') as result:
        with torch.no_grad():
            for batch_idx, input_data in enumerate(dataloader):
                MLP_scores_mat = model(input_data)

                # res=[-1, 5, 0, , 4] - always -1 at the beginning because it's '<root>' token in every sentence's start
                predicted_tree = decode_mst(MLP_scores_mat.data.cpu().numpy().T, length=MLP_scores_mat.shape[0],
                                            has_labels=False)[0]

                for head in predicted_tree[1:]:
                    original_line = lines[0]
                    tabs_locs = [idx for idx, char in enumerate(original_line) if char == "\t"]
                    # search for the 6th '\t'
                    line_to_save = original_line[:tabs_locs[5] + 1] + str(head) + original_line[tabs_locs[6]:]
                    result.write(line_to_save)
                    del lines[0]

                result.write(lines[0])
                del lines[0]  # the separating \n
    # lines should be empty list by now


def plot_graphs(train_accuracy_list, train_loss_list, test_accuracy_list, test_loss_list):
    indices_list = [(1 + i) for i in range(len(train_accuracy_list))]
    min_acc = min(min(train_accuracy_list), min(test_accuracy_list))
    plt.plot(indices_list, train_accuracy_list, '-', c="tab:blue", label="Train accuracy")
    # plt.plot(test_accuracy_list, c="orange", label="Test accuracy")
    plt.plot(indices_list, test_accuracy_list, '-', c="tab:orange", label="Test accuracy")

    plt.plot(indices_list, train_accuracy_list, 'o', color='tab:blue', markersize=4)
    plt.plot(indices_list, test_accuracy_list, 'o', color='tab:orange', markersize=4)
    plt.xlim(left=1)
    plt.xticks(np.arange(1, len(indices_list) + 1, step=1))
    plt.ylim((min_acc * 0.99, 1))
    plt.grid(linewidth=1)
    plt.title("Train and test accuracies along epochs")
    plt.xlabel("Epochs")
    plt.ylabel("Accuracy")
    plt.legend(loc='lower right')
    plt.show()

    max_loss = max(max(train_loss_list), max(test_loss_list))
    plt.plot(indices_list, train_loss_list, '-', c="tab:blue", label="Train loss")
    plt.plot(indices_list, test_loss_list, '-', c="tab:orange", label="Test loss")
    plt.plot(indices_list, train_loss_list, 'o', color='tab:blue', markersize=4)
    plt.plot(indices_list, test_loss_list, 'o', color='tab:orange', markersize=4)
    plt.xlim(left=1)
    plt.xticks(np.arange(1, len(indices_list) + 1, step=1))
    plt.ylim((0, max_loss * 1.01))
    plt.grid(linewidth=1)
    plt.title("Train and test losses along epochs")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.legend()
    plt.show()


def main():
    word_embd_dim = 100  # article's default
    pos_embd_dim = 25  # article's default
    hidden_dim = 125  # article's default
    MLP_inner_dim = 100  # article's default
    epochs = 30
    learning_rate = 0.01  # Adam's default
    dropout_layers_probability = 0.0  # nn.LSTM default
    weight_decay = 0.0  # Adam's default
    alpha = 0.25  # 0.0 means no word dropout | 0.25 article's default
    min_freq = 1  # minimum term-frequency to include in vocabulary, use 1 if you wish to use all words
    BiLSTM_layers = 2  # article's default
    use_pre_trained = False
    vectors = f'glove.6B.{word_embd_dim}d' if use_pre_trained else ''
    path_train = "train.labeled"
    path_test = "test.labeled"

    run_description = f"KiperwasserDependencyParser\n" \
                      f"-------------------------------------------------------------------------------------------\n" \
                      f"word_embd_dim = {word_embd_dim}\n" \
                      f"pos_embd_dim = {pos_embd_dim}\n" \
                      f"hidden_dim = {hidden_dim}\n" \
                      f"MLP_inner_dim = {MLP_inner_dim}\n" \
                      f"epochs = {epochs}\n" \
                      f"learning_rate = {learning_rate}\n" \
                      f"dropout_layers_probability = {dropout_layers_probability}\n" \
                      f"weight_decay = {weight_decay}\n" \
                      f"alpha = {alpha}\n" \
                      f"min_freq = {min_freq}\n" \
                      f"BiLSTM_layers = {BiLSTM_layers}\n" \
                      f"use_pre_trained = {use_pre_trained}\n" \
                      f"vectors = {vectors}\n" \
                      f"path_train = {path_train}\n" \
                      f"path_test = {path_test}\n"

    current_machine_date_time = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(int(time.time())))
    print(f"{current_machine_date_time}\n"
          f"{run_description}")

    path_to_save_model = os.path.join('saved_models', f'model {current_machine_date_time} ')

    """TRAIN DATA"""
    train_word_dict, train_pos_dict = get_vocabs_counts([path_train])
    train = DependencyDataset(path=path_train, word_dict=train_word_dict, pos_dict=train_pos_dict,
                              word_embd_dim=word_embd_dim, pos_embd_dim=pos_embd_dim, test=False,
                              use_pre_trained=use_pre_trained, pre_trained_vectors_name=vectors, min_freq=min_freq,
                              comp_mode=False)
    train_dataloader = DataLoader(train, shuffle=True)
    model = KiperwasserDependencyParser(train, hidden_dim, MLP_inner_dim, BiLSTM_layers, dropout_layers_probability)

    """TEST DATA"""

    test = DependencyDataset(path=path_test, word_dict=train_word_dict, pos_dict=train_pos_dict,
                             test=[train.word_idx_mappings, train.pos_idx_mappings], comp_mode=False)
    test_dataloader = DataLoader(test, shuffle=False)

    """TRAIN THE PARSER ON TRAIN DATA"""
    train_accuracy_list, train_loss_list, test_accuracy_list, test_loss_list = \
        train_kiperwasser_parser(model, train_dataloader, test_dataloader, epochs, learning_rate, weight_decay, alpha,
                                 path_to_save_model)

    print(f'\ntrain_accuracy_list = {train_accuracy_list}'
          f'\ntrain_loss_list = {train_loss_list}'
          f'\ntest_accuracy_list = {test_accuracy_list}'
          f'\ntest_loss_list = {test_loss_list}')

    """PLOT GRAPHS"""
    # plot_graphs(train_accuracy_list, train_loss_list, test_accuracy_list, test_loss_list) # TODO

    """TAG THE TEST DATA"""
    # tag_file_save_output(model, test_dataloader, path_test, 'yotam_test_tagged.labeled')  # TODO


def plot_graphs_test_accuracy_analyze(test0_accuracy_list, test01_accuracy_list, test1_accuracy_list,
                                      test2_accuracy_list, test3_accuracy_list):
    indices_list = [(1 + i) for i in range(len(test1_accuracy_list))]
    min_acc = min(min(test1_accuracy_list), min(test2_accuracy_list))
    max_acc = max(max(test1_accuracy_list), max(test2_accuracy_list))
    plt.plot(indices_list, test0_accuracy_list, '-', c="yellow", label="Test100 accuracy")
    plt.plot(indices_list, test01_accuracy_list, '-', c="green", label="Test75 accuracy")
    plt.plot(indices_list, test1_accuracy_list, '-', c="blue", label="Test50 accuracy")
    plt.plot(indices_list, test2_accuracy_list, '-', c="orange", label="Test25 accuracy")
    plt.plot(indices_list, test3_accuracy_list, '-', c="red", label="Test15 accuracy")

    plt.plot(indices_list, test0_accuracy_list, 'o', color='yellow', markersize=4)
    plt.plot(indices_list, test01_accuracy_list, 'o', color='green', markersize=4)
    plt.plot(indices_list, test1_accuracy_list, 'bo', markersize=4)
    plt.plot(indices_list, test2_accuracy_list, 'o', color='orange', markersize=4)
    plt.plot(indices_list, test3_accuracy_list, 'ro', markersize=4)

    plt.xlim(left=1)
    plt.ylim((0.85, 0.93))
    plt.yticks(np.arange(0.85, 0.93, step=0.01))  # Set label locations.
    plt.grid(linewidth=1)
    plt.xlabel("Epochs")
    plt.ylabel("Accuracy")
    plt.legend(loc='lower right')
    plt.show()


if __name__ == "__main__":
    main()
    # TODO gal do not touch
    # test0_accuracy_list = [0.8727104966642354, 0.8893802954456296, 0.897930519999827, 0.8985994775363773, 0.9024147626896759, 0.9032315328233659, 0.9057396530796497, 0.9041129008595366, 0.9074323189898621, 0.9085594565181768, 0.9068517342149139, 0.9087980613642939, 0.9079916950507139, 0.9077482190613283, 0.9115374263674864]
    # test01_accuracy_list = [0.8745952303818547, 0.891846345128242, 0.9000240995082567, 0.8975676189017281, 0.9023449139842078, 0.8996635935558741, 0.9031568997647944, 0.903084222231001, 0.904563341273347, 0.9040021500833616, 0.9057205887688459, 0.9087909525565078, 0.9036278991177055, 0.9055740197581077, 0.9080697350369914]
    # test1_accuracy_list = [0.8724469579477907, 0.8931250160250392, 0.9012826544100603, 0.8971577191400117, 0.9025055033866338, 0.9039649728566629, 0.9068932193480703, 0.9030561335556132, 0.9077176005582834, 0.908902367632979, 0.9066836905249072, 0.9106718023398801, 0.9085296473936154, 0.9062363725260917, 0.9072766937607392]
    # test2_accuracy_list = [0.8693521109877581, 0.8865615967570638, 0.8962231260051488, 0.9005227467073518, 0.9043345219473015, 0.9039425811643059, 0.9073953874092233, 0.9052128700217318, 0.9037999714323558, 0.9047589755353243, 0.9041192373737272, 0.9022990229207869, 0.9053757493664162, 0.9056269514471683, 0.9064894272351405]
    # test3_accuracy_list = [0.847536410303437, 0.8883146342123204, 0.8887979055703453, 0.8952708644741162,
    # plot_graphs_test_accuracy_analyze(test0_accuracy_list, test01_accuracy_list, test1_accuracy_list, test2_accuracy_list, test3_accuracy_list)
