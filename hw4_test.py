# ML_HW4_RNN
# ----- strong baseline ----- 0.82171   
# ----- simple baseline ----- 0.76978

# **可以改的地方**
# 1. def train_word2vec(x):  訓練 word to vector 的 word embedding   
#     model = word2vec.Word2Vec(x, size=250, window=5, min_count=5, workers=12, iter=10, sg=1) #原本的

# 2. class LSTM_Net(nn.Module):  
#     def __init__(self, embedding, embedding_dim, hidden_dim, num_layers,  dropout=0.5, fix_embedding=True)

# 3. training
#     def training(batch_size, n_epoch, lr, model_dir, train, valid, model, device)
# 4. Ensemble Learning   
#     Reference: http://violin-tao.blogspot.com/2018/01/ml-ensemble.html

########################################################################

path_prefix = './'
# 讀進去 data_set
import sys
testing_data = sys.argv[1]
prediction_file = sys.argv[2]

# 測試本機端資料
# testing_data = path_prefix + 'testing_data.txt'
# prediction_file = path_prefix + 'prediction.csv'

########################################################################

# this is for filtering the warnings
# 警告过滤器可以用来控制是否发出警告消息，警告过滤器是一些匹配规则和动作的序列。
# 可以通过调用 filterwarnings() 将规则添加到过滤器，并通过调用 resetwarnings() 将其重置为默认状态。
import warnings
warnings.filterwarnings('ignore')

########################################################################

"""### Utils"""

# utils.py
# 這個 block 用來先定義一些等等常用到的函式
import torch
# torch.manual_seed(0) #reproducible
import numpy as np
np.random.seed(0)
import pandas as pd
import torch.optim as optim
import torch.nn.functional as F
import time

################################################################################

def load_testing_data(path='testing_data'):
    # 把 testing 時需要的 data 讀進來
    with open(path, 'r') as f:
        lines = f.readlines()
        X = ["".join(line.strip('\n').split(",")[1:]).strip() for line in lines[1:]]
        X = [sen.split(' ') for sen in X]
    return X

def evaluation(outputs, labels):
    # outputs => probability (float)
    # labels => labels
    outputs[outputs>=0.5] = 1 # 大於等於 0.5 為有惡意 #原本的
    outputs[outputs<0.5] = 0 # 小於 0.5 為無惡意 #原本的
    correct = torch.sum(torch.eq(outputs, labels)).item()
    return correct

########################################################################

"""### Train Word to Vector

CBOW：From that context, predict the target word (Continuous Bag of Words or CBOW approach)

Skip-Gram：From the target word, predict the context it came from (Skip-gram approach)
"""

# w2v.py
# 這個 block 是用來訓練 word to vector 的 word embedding
# 注意！這個 block 在訓練 word to vector 時是用 cpu，可能要花到 10 分鐘以上
import os
import numpy as np
import pandas as pd
import argparse
from gensim.models import word2vec

# word2vec使用
# size：特徵向量的維度，預設值為100
# window：代表input word 與預測word的最大距離，若設為1，代表向左向右各看一個詞
# min_count：該詞最少出現幾次，才可以被當作是訓練資料，例如min_count設為5，則出現5次以下的字詞都不會被放進來訓練
# negative：大於0，代表採用negative sampling，設置多少個noise words
#         例如「喜歡 麥當當 蘋果派 又 甜 又 香」，以「蘋果派」向左向右各看一個詞，就是「麥當當 蘋果派 又」
# workers：多執行緒的數量
# iter：迭代數量，預設為5
# sg：演算法，預設為0，代表是CBOW，若設為1則是使用Skip-Gram

def train_word2vec(x):
    # 訓練 word to vector 的 word embedding
    # model = word2vec.Word2Vec(x, size=250, window=5, min_count=5, workers=12, iter=10, sg=1) #原本的
    model = word2vec.Word2Vec(x, size=250, window=5, min_count=6, workers=12, iter=10, sg=1)
    return model

########################################################################

if __name__ == "__main__":

    print("loading testing data ...")
    test_x = load_testing_data(testing_data)

    model = train_word2vec(test_x) #semi-supervised

    print("saving model ...")
    # model.save(os.path.join(path_prefix, 'model/w2v_all.model'))
    model.save(os.path.join(path_prefix, 'w2v_all.model'))

########################################################################

"""### Data Preprocess"""

# preprocess.py
# 這個 block 用來做 data 的預處理
from torch import nn
from gensim.models import Word2Vec

class Preprocess():
    def __init__(self, sentences, sen_len, w2v_path="./w2v_all.model"):
        self.w2v_path = w2v_path
        self.sentences = sentences
        self.sen_len = sen_len
        self.idx2word = []
        self.word2idx = {}
        self.embedding_matrix = []
    def get_w2v_model(self):
        # 把之前訓練好的 word to vec 模型讀進來
        self.embedding = Word2Vec.load(self.w2v_path)
        self.embedding_dim = self.embedding.vector_size
    def add_embedding(self, word):
        # 把 word 加進 embedding，並賦予他一個隨機生成的 representation vector
        # word 只會是 "<PAD>" 或 "<UNK>"
          # PAD : 因為每個 batch 的單字長度要一致，所以我們要用 PAD 來填充過短的單字，主要用来进行字符补全。
          # UNK : 如果輸入字元沒在字典裡出現過，就用 UNK 的索引替代它，用来替代一些未出现过的词或者低频词。
        vector = torch.empty(1, self.embedding_dim)
        torch.nn.init.uniform_(vector)
        self.word2idx[word] = len(self.word2idx)
        self.idx2word.append(word)
        self.embedding_matrix = torch.cat([self.embedding_matrix, vector], 0)
    def make_embedding(self, load=True):
        print("Get embedding ...")
        # 取得訓練好的 Word2vec word embedding
        if load:
            print("loading word to vec model ...")
            self.get_w2v_model()
        else:
            raise NotImplementedError
        # 製作一個 word2idx 的 dictionary
        # 製作一個 idx2word 的 list
        # 製作一個 word2vector 的 list
        for i, word in enumerate(self.embedding.wv.vocab):
            print('get words #{}'.format(i+1), end='\r')
            #e.g. self.word2index['he'] = 1 
            #e.g. self.index2word[1] = 'he'
            #e.g. self.vectors[1] = 'he' vector
            self.word2idx[word] = len(self.word2idx)
            self.idx2word.append(word)
            self.embedding_matrix.append(self.embedding[word])
        print('')
        self.embedding_matrix = torch.tensor(self.embedding_matrix)
        # 將 "<PAD>" 跟 "<UNK>" 加進 embedding 裡面
        self.add_embedding("<PAD>")
        self.add_embedding("<UNK>")
        print("total words: {}".format(len(self.embedding_matrix)))
        return self.embedding_matrix
    def pad_sequence(self, sentence):
        # 將每個句子變成一樣的長度
        if len(sentence) > self.sen_len:
            sentence = sentence[:self.sen_len]
        else:
            pad_len = self.sen_len - len(sentence)
            for _ in range(pad_len):
                sentence.append(self.word2idx["<PAD>"])
        assert len(sentence) == self.sen_len
        return sentence
    def sentence_word2idx(self):
        # 把句子裡面的字轉成相對應的 index
        sentence_list = []
        for i, sen in enumerate(self.sentences):
            print('sentence count #{}'.format(i+1), end='\r')
            sentence_idx = []
            for word in sen:
                if (word in self.word2idx.keys()):
                    sentence_idx.append(self.word2idx[word])
                else:
                    sentence_idx.append(self.word2idx["<UNK>"])
            # 將每個句子變成一樣的長度
            sentence_idx = self.pad_sequence(sentence_idx)
            sentence_list.append(sentence_idx)
        return torch.LongTensor(sentence_list)
    def labels_to_tensor(self, y):
        # 把 labels 轉成 tensor
        y = [int(label) for label in y]
        return torch.LongTensor(y)

########################################################################

"""### Dataset"""

# data.py
# 實作了 dataset 所需要的 '__init__', '__getitem__', '__len__'
# 好讓 dataloader 能使用
import torch
from torch.utils import data

class TwitterDataset(data.Dataset):
    """
    Expected data shape like:(data_num, data_len)
    Data can be a list of numpy array or a list of lists
    input data shape : (data_num, seq_len, feature_dim)
    
    __len__ will return the number of data
    """
    def __init__(self, X, y):
        self.data = X
        self.label = y
    def __getitem__(self, idx):
        if self.label is None: return self.data[idx]
        return self.data[idx], self.label[idx]
    def __len__(self):
        return len(self.data)

########################################################################

# model.py
# 這個 block 是要拿來訓練的模型
import torch
from torch import nn
class LSTM_Net(nn.Module):
    def __init__(self, embedding, embedding_dim, hidden_dim, num_layers, dropout=0.5, fix_embedding=True):
        super(LSTM_Net, self).__init__()
        # 製作 embedding layer
        self.embedding = torch.nn.Embedding(embedding.size(0),embedding.size(1))
        self.embedding.weight = torch.nn.Parameter(embedding)
        # 是否將 embedding fix 住，如果 fix_embedding 為 False，在訓練過程中，embedding 也會跟著被訓練
        self.embedding.weight.requires_grad = False if fix_embedding else True
        self.embedding_dim = embedding.size(1)
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.dropout = dropout
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, num_layers=num_layers, batch_first=True)
        self.classifier = nn.Sequential( nn.Dropout(dropout),
                                         nn.Linear(hidden_dim, 1),
                                         nn.Sigmoid() )
    def forward(self, inputs):
        inputs = self.embedding(inputs)
        x, _ = self.lstm(inputs, None)
        # x 的 dimension (batch, seq_len, hidden_size)
        # 取用 LSTM 最後一層的 hidden state
        x = x[:, -1, :] 
        x = self.classifier(x)
        return x

########################################################################

"""### Test"""

# test.py
# 這個 block 用來對 testing_data.txt 做預測
import torch
from torch import nn
import torch.optim as optim
import torch.nn.functional as F

def testing(batch_size, test_loader, model, device):
    model.eval()
    ret_output = []
    with torch.no_grad():
        for i, inputs in enumerate(test_loader):
            inputs = inputs.to(device, dtype=torch.long)
            outputs = model(inputs)
            outputs = outputs.squeeze()
            outputs[outputs>=0.5] = 1 # 大於等於 0.5 為負面
            outputs[outputs<0.5] = 0 # 小於 0.5 為正面
            ret_output += outputs.int().tolist()
               
    return ret_output

########################################################################

"""### Main"""

# main.py
if __name__ == "__main__":
    
    import os
    import torch
    import argparse
    import numpy as np
    from torch import nn
    from gensim.models import word2vec
    from sklearn.model_selection import train_test_split

    # 通過 torch.cuda.is_available() 的回傳值進行判斷是否有使用 GPU 的環境，如果有的話 device 就設為 "cuda"，沒有的話就設為 "cpu"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 處理好各個 data 的路徑
    testing_data = os.path.join(path_prefix, testing_data)


    w2v_path = os.path.join(path_prefix, 'w2v_all_drive.model') # 處理 word to vec model 的路徑

    # 定義句子長度、要不要固定 embedding、batch 大小、要訓練幾個 epoch、learning rate 的值、model 的資料夾路徑
    # sen_len = 20 #原本的
    sen_len = 40
    fix_embedding = True # fix embedding during training
    batch_size = 128
    epoch = 15
    lr = 0.001
    # model_dir = os.path.join(path_prefix, 'model/') # model directory for checkpoint model
    model_dir = path_prefix # model directory for checkpoint model

    ################################################################################


    """### Predict test_data and Write to csv file"""

    # 開始測試模型並做預測
    print("loading testing data ...")
    test_x = load_testing_data(testing_data)
    preprocess = Preprocess(test_x, sen_len, w2v_path=w2v_path)
    embedding = preprocess.make_embedding(load=True)
    test_x = preprocess.sentence_word2idx()
    test_dataset = TwitterDataset(X=test_x, y=None)
    test_loader = torch.utils.data.DataLoader(dataset = test_dataset,
                                                batch_size = batch_size,
                                                shuffle = False,
                                                num_workers = 8)
    print('\nload model ...')
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = torch.load(os.path.join(model_dir, 'drive_ckpt.model'))
    outputs_test = testing(batch_size, test_loader, model, device)
    # 寫到 csv 檔案供上傳 Kaggle
    tmp = pd.DataFrame({"id":[str(i) for i in range(len(test_x))],"label":outputs_test})
    print("save csv ...")
    tmp.to_csv(os.path.join(path_prefix, prediction_file), index=False)
    print("Finish Predicting")

    ########################################################################

    # #將結果寫入 csv 檔
    # with open("prediction file.csv", 'w') as f:
    #     f.write('Id,Category\n')
    #     for i, y in  enumerate(prediction):
    #         f.write('{},{}\n'.format(i, y))




