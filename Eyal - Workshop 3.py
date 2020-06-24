import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import random
from torch.utils.data.dataloader import DataLoader

data_dir = 'HW1_files/'

"""Create Vocabulary"""

def split(string, delimiters):
    """
        Split strings according to delimiters
        :param string: full sentence
        :param delimiters string: characters for spliting
            function splits sentence to words
    """
    delimiters = tuple(delimiters)
    stack = [string, ]

    for delimiter in delimiters:
        for i, substring in enumerate(stack):
            substack = substring.split(delimiter)
            stack.pop(i)
            for j, _substring in enumerate(substack):
                stack.insert(i + j, _substring)

    return stack


def get_vocabs(list_of_paths):
    """
        Extract vocabs from given datasets. Return a word2ids and tag2idx.
        :param file_paths: a list with a full path for all corpuses
            Return:
              - word2idx
              - tag2idx
    """
    word_dict = defaultdict(int)
    pos_dict = defaultdict(int)
    for file_path in list_of_paths:
        with open(file_path) as f:
            for line in f:
                splited_words = split(line, (' ', '\n'))
                del splited_words[-1]
                for word_and_tag in splited_words:
                    word, pos_tag = split(word_and_tag, '_')
                    word_dict[word] += 1
                    pos_dict[pos_tag] += 1

    return word_dict, pos_dict


# ******************* USAGE EXAMPLE (this is good practice) *******************
# path_train = "data/train.wtag"
# path_test = "data/test.wtag"
# paths_list = [path_train, path_test]
# word_dict, pos_dict = get_vocabs(paths_list)
# *****************************************************************************

"""Data Reader"""

from collections import defaultdict


class PosDataReader:
    def __init__(self, file, word_dict, pos_dict):
        self.file = file
        self.word_dict = word_dict
        self.pos_dict = pos_dict
        self.sentences = []
        self.__readData__()

    def __readData__(self):
        """main reader function which also populates the class data structures"""
        with open(self.file, 'r') as f:
            for line in f:
                cur_sentence = []
                splited_words = split(line, (' ', '\n'))
                del splited_words[-1]
                for word_and_tag in splited_words:
                    cur_word, cur_tag = split(word_and_tag, '_')
                    cur_sentence.append((cur_word, cur_tag))
                self.sentences.append(cur_sentence)

    def get_num_sentences(self):
        """returns num of sentences in data"""
        return len(self.sentences)

"""Dataset"""

from torchtext.vocab import Vocab
from torch.utils.data.dataset import Dataset, TensorDataset
from pathlib import Path
from collections import Counter

# These are not relevant for our POS tagger but might be usefull for HW2
UNKNOWN_TOKEN = "<unk>"
PAD_TOKEN = "<pad>"  # Optional: this is used to pad a batch of sentences in different lengths.
# ROOT_TOKEN = PAD_TOKEN # this can be used if you are not padding your batches
# ROOT_TOKEN = "<root>" # use this if you are padding your batches and want a special token for ROOT
SPECIAL_TOKENS = [PAD_TOKEN, UNKNOWN_TOKEN]


class PosDataset(Dataset):
    def __init__(self, word_dict, pos_dict, dir_path: str, subset: str,
                 padding=False, word_embeddings=None):
        super().__init__()
        self.subset = subset  # One of the following: [train, test]
        self.file = dir_path + subset + ".wtag"
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
        self.pos_idx_mappings, self.idx_pos_mappings = self.init_pos_vocab(self.datareader.pos_dict)

        # במודל הזה אנחנו לא נעשה באטצ'ינג ואנחנו לא נעשה את הריפוד
        # נעשה משהו שקול לעבודה עם באטצ'ים עלידי איזשהו טריק
        # אבל בלי שימוש בבאטצ'ים באמת
        self.pad_idx = self.word_idx_mappings.get(PAD_TOKEN)
        # עבור מילים שלא ראיתי - המודל שלי ידע להתייחס אליהן אבל אני לא מכיר אותן
        self.unknown_idx = self.word_idx_mappings.get(UNKNOWN_TOKEN)
        self.word_vector_dim = self.word_vectors.size(-1)
        self.sentence_lens = [len(sentence) for sentence in self.datareader.sentences]
        # משפטים שארוכים מהאורך הזה ייחתכו
        self.max_seq_len = max(self.sentence_lens)
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

    def init_pos_vocab(self, pos_dict):
        idx_pos_mappings = sorted([self.word_idx_mappings.get(token) for token in SPECIAL_TOKENS])
        pos_idx_mappings = {self.idx_word_mappings[idx]: idx for idx in idx_pos_mappings}

        for i, pos in enumerate(sorted(pos_dict.keys())):
            # pos_idx_mappings[str(pos)] = int(i)
            pos_idx_mappings[str(pos)] = int(i + len(SPECIAL_TOKENS))
            idx_pos_mappings.append(str(pos))
        print("idx_pos_mappings -", idx_pos_mappings)
        print("pos_idx_mappings -", pos_idx_mappings)
        return pos_idx_mappings, idx_pos_mappings

    def get_pos_vocab(self):
        return self.pos_idx_mappings, self.idx_pos_mappings

    def convert_sentences_to_dataset(self, padding):
        # מחזירה מילון שמכיל דוגמאות של אינפוט ,לייבל וגודל הבאטצ' -
        # מי שלא יעבוד עם באטצ' לא צריך את האחרון מביניהם
        sentence_word_idx_list = list()
        sentence_pos_idx_list = list()
        sentence_len_list = list()
        for sentence_idx, sentence in enumerate(self.datareader.sentences):
            words_idx_list = []
            pos_idx_list = []
            for word, pos in sentence:
                words_idx_list.append(self.word_idx_mappings.get(word))
                pos_idx_list.append(self.pos_idx_mappings.get(pos))
            sentence_len = len(words_idx_list)
            # if padding:
            #     while len(words_idx_list) < self.max_seq_len:
            #         words_idx_list.append(self.word_idx_mappings.get(PAD_TOKEN))
            #         pos_idx_list.append(self.pos_idx_mappings.get(PAD_TOKEN))
            sentence_word_idx_list.append(torch.tensor(words_idx_list, dtype=torch.long, requires_grad=False))
            sentence_pos_idx_list.append(torch.tensor(pos_idx_list, dtype=torch.long, requires_grad=False))
            sentence_len_list.append(sentence_len)

        # if padding:
        #     all_sentence_word_idx = torch.tensor(sentence_word_idx_list, dtype=torch.long)
        #     all_sentence_pos_idx = torch.tensor(sentence_pos_idx_list, dtype=torch.long)
        #     all_sentence_len = torch.tensor(sentence_len_list, dtype=torch.long, requires_grad=False)
        #     return TensorDataset(all_sentence_word_idx, all_sentence_pos_idx, all_sentence_len)

        return {i: sample_tuple for i, sample_tuple in enumerate(zip(sentence_word_idx_list,
                                                                     sentence_pos_idx_list,
                                                                     sentence_len_list))}

path_train = data_dir + "train1.wtag"
print("path_train -", path_train)
path_test = data_dir + "test1.wtag"
print("path_test -", path_test)

paths_list = [path_train, path_test]
word_dict, pos_dict = get_vocabs(paths_list)
train = PosDataset(word_dict, pos_dict, data_dir, 'train1', padding=False)
# ידגום לנו כל פעם משפטים מתוך הדאטא סט כי נרצה רנדומליות
train_dataloader = DataLoader(train, shuffle=True)
test = PosDataset(word_dict, pos_dict, data_dir, 'test1', padding=False)
# אין צורך ברנדומליות כי לא לומדים בשלב הזה
test_dataloader = DataLoader(test, shuffle=False)

print("Number of Train Tagged Sentences ", len(train))
print("Number of Test Tagged Sentences ",len(test))


"""Create a model"""

class DnnPosTagger(nn.Module):
    def __init__(self, word_embeddings, hidden_dim, word_vocab_size, tag_vocab_size):
        super(DnnPosTagger, self).__init__()
        emb_dim = word_embeddings.shape[1]
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        #self.word_embedding = nn.Embedding(word_vocab_size, word_embedding_dim)
        # משתמשים בפרי-טריינד, פרייז=פאלס אומר שאנחנו נאמן את המשקולות בעצמנו גם
        self.word_embedding = nn.Embedding.from_pretrained(word_embeddings, freeze=False)
        # emb_dim גודל האינפוט
        # hidden_dim גודל הפלט
        self.lstm = nn.LSTM(input_size=emb_dim, hidden_size=hidden_dim, num_layers=2, bidirectional=True, batch_first=False)
        # אם נקח את הערך המקסימלי ממה שיוצא בוקטור פה אז זה החיזוי של המודל שלנו
        # כפול 2 כי אנחנו עושים בי-דירקושנל
        self.hidden2tag = nn.Linear(hidden_dim*2, tag_vocab_size)

        # מה שמצפה כל מודל של למידה עמוקה - איך להתקדם קדימה ברשת
        # word_idx_tenseor, pos_idx_tensor אנחנו נדרש לקלט לפונקציה שהוא
    def forward(self, word_idx_tensor):
        embeds = self.word_embedding(word_idx_tensor.to(self.device))   # [batch_size, seq_length, emb_dim]
        lstm_out, _ = self.lstm(embeds.view(embeds.shape[1], 1, -1))    # [seq_length, batch_size, 2*hidden_dim]
        tag_space = self.hidden2tag(lstm_out.view(embeds.shape[1], -1)) # [seq_length, tag_dim]
        tag_scores = F.log_softmax(tag_space, dim=1)                    # [seq_length, tag_dim]
        return tag_scores


"""Evaluation Method"""


def evaluate():
    acc = 0
    # להגיד למודל לא ללמוד כרגע
    with torch.no_grad():
        # דוגמים מהדאטא לאודר
        for batch_idx, input_data in enumerate(test_dataloader):
            words_idx_tensor, pos_idx_tensor, sentence_length = input_data
            tag_scores = model(words_idx_tensor)
            tag_scores = tag_scores.unsqueeze(0).permute(0, 2, 1)

            _, indices = torch.max(tag_scores, 1)
            # עושים מעבר למעבד הרגיל לפני
            acc += torch.mean(torch.tensor(pos_idx_tensor.to("cpu") == indices.to("cpu"), dtype=torch.float))
        acc = acc / len(test)
    return acc

"""Training The LSTM Model"""
# איך בפועל מעבדים לפי באטצ'ים
# CUDA_LAUNCH_BLOCKING=1

# פרמטרים למודל
EPOCHS = 15
WORD_EMBEDDING_DIM = 100
HIDDEN_DIM = 1000
word_vocab_size = len(train.word_idx_mappings)
tag_vocab_size = len(train.pos_idx_mappings)

model = DnnPosTagger(train_dataloader.dataset.word_vectors, HIDDEN_DIM, word_vocab_size, tag_vocab_size)

use_cuda = torch.cuda.is_available()
device = torch.device("cuda:0" if use_cuda else "cpu")

if use_cuda:
    model.cuda()

# Define the loss function as the Negative Log Likelihood loss (NLLLoss)
loss_function = nn.NLLLoss()

# We will be using a simple SGD optimizer to minimize the loss function
optimizer = optim.Adam(model.parameters(), lr=0.01)
acumulate_grad_steps = 50  # This is the actual batch_size, while we officially use batch_size=1

# Training start
print("Training Started")
accuracy_list = []
loss_list = []
epochs = EPOCHS
for epoch in range(epochs):
    acc = 0  # to keep track of accuracy
    printable_loss = 0  # To keep track of the loss value
    i = 0
    for batch_idx, input_data in enumerate(train_dataloader):
        i += 1
        words_idx_tensor, pos_idx_tensor, sentence_length = input_data

        tag_scores = model(words_idx_tensor)
        tag_scores = tag_scores.unsqueeze(0).permute(0, 2, 1)
        # print("tag_scores shape -", tag_scores.shape)
        # print("pos_idx_tensor shape -", pos_idx_tensor.shape)
        loss = loss_function(tag_scores, pos_idx_tensor.to(device))
        # כאילו צוברים את הדרגיאנטים
        loss = loss / acumulate_grad_steps
        loss.backward()

        # במשך 50 צעדים צברנו גרדיאנטים מנורמלים ואז אנחנו רק עושים את הצעד
        if i % acumulate_grad_steps == 0:
            optimizer.step()
            # כדי שפעם הבאה שנעשה בקוורד זה יתווסף ל0 ולא למה שהיה לנו פה
            model.zero_grad()
        printable_loss += loss.item()
        #
        _, indices = torch.max(tag_scores, 1)
        # print("tag_scores shape-", tag_scores.shape)
        # print("indices shape-", indices.shape)
        # acc += indices.eq(pos_idx_tensor.view_as(indices)).mean().item()
        # הממוצע בפקודה הבאה חסר משמעות כי זה רק דגימה 1
        acc += torch.mean(torch.tensor(pos_idx_tensor.to("cpu") == indices.to("cpu"), dtype=torch.float))
    printable_loss = printable_loss / len(train)
    # צריך להיות על כל האפוק של הטריין
    acc = acc / len(train)
    loss_list.append(float(printable_loss))
    accuracy_list.append(float(acc))
    # מחשב את ההצלחה על המבחן
    test_acc = evaluate()
    e_interval = i
    print("Epoch {} Completed,\tLoss {}\tAccuracy: {}\t Test Accuracy: {}".format(epoch + 1,
                                                                                  np.mean(loss_list[-e_interval:]),
                                                                                  np.mean(accuracy_list[-e_interval:]),
                                                                                  test_acc))

    # עם עוד עבודה על פרמטרים וגם עבודה אלגוריתמית אפשר לשפר מאוד את המודל