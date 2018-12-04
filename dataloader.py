import sys
from config import config
sys.path.append('..')
import pickle
import json
from absl import app
import numpy as np
import os
import nltk
import h5py
from gensim.models import KeyedVectors
from torch.utils.data import Dataset, DataLoader

def load_json(filename):
    with open(filename) as fr:
        return json.load(fr)

def save_json(obj, filename):
    with open(filename) as fr:
        json.dump(obj,fr)

class Loader(Dataset):
    def __init__(self, params, key_file):
        #general
        self.params = params
        self.feature_path = params.feature_path
        self.max_batch_size = params.batch_size

        # dataset
        self.key_file = load_json(key_file)[1000:1000+config.batch_size * 20]
        self.dataset_size = len(self.key_file)

        # frame / question
        self.max_frames = params.max_frames
        self.input_video_dim = params.input_video_dim
        self.max_words = params.max_words
        self.input_ques_dim = params.input_ques_dim


    def __getitem__(self, index):

        frame_vecs = np.zeros((self.max_frames, self.input_video_dim), dtype=np.float32)
        ques_vecs = np.zeros((self.max_words, self.input_ques_dim), dtype=np.float32)

        keys = self.key_file[index]
        vid, duration, timestamps, ques = keys[0], keys[1], keys[2], keys[3]

        # video
        if not os.path.exists(self.feature_path + '/%s.h5' % vid):
            print('the video is not exist:', vid)
        with h5py.File(self.feature_path + '%s.h5' % vid, 'r') as fr:
            feats = np.asarray(fr['feature'])

        inds = np.floor(np.arange(0, len(feats)-0.000001, len(feats) / self.max_frames)).astype(int)
        frames = feats[inds, :]
        frames = np.vstack(frames)
        frame_vecs[:self.max_frames, :] = frames[:self.max_frames, :]
        frame_n = np.array(len(frame_vecs),dtype=np.int32)
        frame_per_sec = self.max_frames/duration
        start_frame = round(frame_per_sec * timestamps[0])
        end_frame = round(frame_per_sec * timestamps[1]) - 1
        start_frame = np.array(start_frame, dtype=np.long)
        end_frame = np.array(end_frame, dtype=np.long)
        try:
            ques_n = ques.index(0)
        except:
            ques_n = config.max_words

        return frame_vecs, frame_n, np.array(ques), np.array(ques_n), start_frame, end_frame

    def __len__(self):
        return self.dataset_size

def preprocess(_):
    word2vec = KeyedVectors.load_word2vec_format(config.word2vec, binary=True)
    qs = []

    for keys in [load_json(config.train_data),load_json(config.val_data),load_json(config.test_data)]:
        for key in keys:
            sent = key[3]
            stopwords = ['.', '?', ',', '']
            sent = nltk.word_tokenize(sent)
            ques = [word.lower() for word in sent if word not in stopwords]
            qs += ques
    print(len(qs))
    qs = list(set(qs))
    print(len(qs))
    embeddings = [[0] * config.input_ques_dim,[0] * config.input_ques_dim,[0] * config.input_ques_dim]
    qs = [i for i in qs if i in word2vec]

    for i in qs:
        embeddings.append(word2vec[i])
    print('length of word dict : ', len(qs))
    word2index = dict(zip(qs,range(3,len(qs)+3)))
    word2index['PAD'] = 0
    word2index['UNK'] = 1
    word2index['EOS'] = 2
    index2word = dict([(v, k) for (k, v) in word2index.items()])
    print(len(embeddings))
    print(len(index2word))
    train_data = []
    test_data = []
    val_data= []
    for data, keys in zip([train_data,val_data,test_data],[load_json(config.train_data),load_json(config.val_data),load_json(config.test_data)]):
        for key in keys:
            tmp = [0] * config.max_words
            count = 0
            for i in range(config.max_words):
                try:
                    tmp[i-count] = word2index[key[3][i]]
                except:
                    count += 1
            key[3] = tmp
            data.append(key)
    print(train_data[0])
    print(test_data[0])
    print(val_data[0])
    json.dump(train_data,open(config.p_train_data,'w'))
    json.dump(test_data, open(config.p_test_data,'w'))
    json.dump(val_data, open(config.p_val_data,'w'))
    pickle.dump([word2index,index2word,embeddings],open(config.data_pickle,'wb'))

def main(_):
    train_dataset = Loader(config, config.p_train_data)

    # Data loader (this provides queues and threads in a very simple way).
    train_loader = DataLoader(dataset=train_dataset, batch_size=64, shuffle=True)

    # When iteration starts, queue and thread start to load data from files.
    data_iter = iter(train_loader)

    frame_vecs, frame_n, ques, ques_n, gt_windows = data_iter.next()
    print(frame_vecs.shape)
    print(frame_n.shape)
    print(ques.shape)
    print(ques_n.shape)
    print(gt_windows)

    print(frame_vecs)
    print(frame_n)
    print(ques)
    print(ques_n)



if __name__ == '__main__':
    app.run(main)
