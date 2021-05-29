# -*- coding: utf-8 -*-

import os
import json
import nltk
import random
import pickle

import numpy as np

from transformers import AutoTokenizer, AutoModel, AutoConfig

def build_embedding(data_dir, token2idx, embed_size):
    if os.path.exists(os.path.join(data_dir, 'embedding.pt')):
        print('>> loading embedding: {}'.format(
            os.path.join(data_dir, 'embedding.pt')))
        embedding = pickle.load(
            open(os.path.join(data_dir, 'embedding.pt'), 'rb'))
    else:
        # words not found in embedding index will be randomly initialized.
        embedding = np.random.uniform(-1 / np.sqrt(embed_size),
                                      1 / np.sqrt(embed_size),
                                      (len(token2idx), embed_size))
        embedding[0, :] = np.zeros((1, embed_size))
        fn = 'glove.840B.300d.txt'
        print('>> loading word vectors')
        word2vec = {}
        with open(fn, 'r', encoding='utf-8', errors='ignore') as f:
            for line in f:
                numbers = line.rstrip().split()
                token, vec = ' '.join(numbers[:-embed_size]), numbers[-embed_size:]
                if token in token2idx.keys():
                    word2vec[token] = np.asarray(vec, dtype=np.float32)
        print('>> building embedding: {}'.format(
            os.path.join(data_dir, 'embedding.pt')))
        for token, i in token2idx.items():
            vec = word2vec.get(token)
            if vec is not None:
                embedding[i] = vec
        pickle.dump(embedding,
                    open(os.path.join(data_dir, 'embedding.pt'), 'wb'))

    return embedding

def build_embedding_for_bert(data_dir, cache_dir='cahces'):
    config = AutoConfig.from_pretrained(data_dir, cache_dir=cache_dir)                           
    embedding = AutoModel.from_pretrained(data_dir, config=config, cache_dir=cache_dir)

    return embedding

class Tokenizer(object):
    def __init__(self, token2idx=None):
        if token2idx is None:
            self.token2idx = {}
            self.idx2token = {}
            self.idx = 0
            self.token2idx['<pad>'] = self.idx
            self.idx2token[self.idx] = '<pad>'
            self.idx += 1
            self.token2idx['<unk>'] = self.idx
            self.idx2token[self.idx] = '<unk>'
            self.idx += 1
        else:
            self.token2idx = token2idx
            self.idx2token = {v: k for k, v in token2idx.items()}

    def fit_on_text(self, text):
        tokens = text.split()
        for token in tokens:
            if token not in self.token2idx:
                self.token2idx[token] = self.idx
                self.idx2token[self.idx] = token
                self.idx += 1

    @staticmethod
    def tokenize(text):
        return nltk.word_tokenize(text.lower(), preserve_line=True)

    def convert_tokens_to_ids(self, tokens):
        return [self.token2idx[t] if t in self.token2idx else 1 for t in tokens]

    def __call__(self, text):
        return self.convert_tokens_to_ids(self.tokenize(text))


def build_tokenizer(data_dir):
    if os.path.exists(os.path.join(data_dir, 'token2idx.pt')):
        print('>> loading {} tokenizer'.format(data_dir))
        with open(os.path.join(data_dir, 'token2idx.pt'), 'rb') as f:
            token2idx = pickle.load(f)
            tokenizer = Tokenizer(token2idx=token2idx)
    else:
        all_text = ''
        set_types = ['train', 'dev', 'test']
        for set_type in set_types:
            with open(os.path.join(data_dir, '{}.json'.format(set_type)),
                      'r',
                      encoding='utf-8') as f:
                set_dict = json.load(f)
                for k in set_dict:
                    text = ' '.join(Tokenizer.tokenize(set_dict[k]['sentence']))
                    all_text += (text + ' ')
        tokenizer = Tokenizer()
        tokenizer.fit_on_text(all_text)
        print('>> building {} tokenizer'.format(data_dir))
        with open(os.path.join(data_dir, 'token2idx.pt'), 'wb') as f:
            pickle.dump(tokenizer.token2idx, f)

    return tokenizer

def build_tokenizer_for_bert(data_dir, cache_dir='caches', use_fast=True):
    tokenizer = AutoTokenizer.from_pretrained(data_dir, cache_dir=cache_dir, use_fast=True)

    return tokenizer

def truncate_and_pad(indices, max_length=128, pad_idx=0):
    indices = indices[:max_length]
    _len = len(indices)
    indices = indices + [pad_idx] * (max_length - _len)
    mask = [1] * _len + [0] * (max_length - _len)
    
    return indices, mask

def build_data(data_dir, tokenizer, max_length=128):
    data_dict = {'train': [], 'dev': [], 'test': []}
    set_types = ['train', 'dev', 'test']
    polarity_map = {'negative': 0, 'neutral': 1, 'positive': 2}
    for set_type in set_types:
        with open(os.path.join(data_dir, '{}.json'.format(set_type)), 'r', encoding='utf-8') as f:
            set_dict = json.load(f)
            for k in set_dict:
                sentence = set_dict[k]['sentence']

                text_left = sentence[:set_dict[k]['from']]
                text_right = sentence[set_dict[k]['to']:]
                aspect = sentence[set_dict[k]['from']:set_dict[k]['to']]

                left_indices = tokenizer(text_left)
                right_indices = tokenizer(text_right)
                aspect_indices = tokenizer(aspect)

                text_indices, text_mask = truncate_and_pad(left_indices + aspect_indices + right_indices, max_length=max_length)
                #aspect_position_mask, _ = truncate_and_pad([0] * len(left_indices) + [1] * len(aspect_indices) + [0] * len(right_indices), max_length=max_length)
                aspect_boundary_indices = [len(left_indices), len(left_indices) + len(aspect_indices) - 1]
                aspect_indices, aspect_mask = truncate_and_pad(aspect_indices, max_length=max_length)
            
                polarity = polarity_map[set_dict[k]['polarity']]

                data = {
                    'text_indices': text_indices,
                    'text_mask': text_mask,
                    #'aspect_position_mask': aspect_position_mask,
                    'aspect_boundary_indices': aspect_boundary_indices,
                    'aspect_indices': aspect_indices,
                    'aspect_mask': aspect_mask,
                    'polarity': polarity,
                }

                data_dict[set_type].append(data)

    return data_dict

def build_data_for_bert(data_dir, tokenizer, max_length=128):
    data_dict = {'train': [], 'dev': [], 'test': []}
    set_types = ['train', 'dev', 'test']
    polarity_map = {'negative': 0, 'neutral': 1, 'positive': 2}

    for set_type in set_types:
        with open(os.path.join(data_dir, '{}.json'.format(set_type)), 'r', encoding='utf-8') as f:
            set_dict = json.load(f)
            for k in set_dict:
                sentence = set_dict[k]['sentence']

                text_left = sentence[:set_dict[k]['from']]
                text_right = sentence[set_dict[k]['to']:]
                aspect = sentence[set_dict[k]['from']:set_dict[k]['to']]

                left_indices = tokenizer(text_left, add_special_tokens=False)['input_ids']
                right_indices = tokenizer(text_right, add_special_tokens=False)['input_ids']
                aspect_indices = tokenizer(aspect, add_special_tokens=False)['input_ids']

                text_indices, text_mask = truncate_and_pad([tokenizer.cls_token_id] + left_indices + aspect_indices + right_indices + [tokenizer.sep_token_id], max_length=max_length, pad_idx=tokenizer.pad_token_id)
                #aspect_position_mask, _ = truncate_and_pad([0] + [0] * len(left_indices) + [1] * len(aspect_indices) + [0] * len(right_indices) + [0], max_length=max_length, pad_idx=0)
                aspect_boundary_indices = [len(left_indices) + 1, len(left_indices) + len(aspect_indices)]
                aspect_indices, aspect_mask = truncate_and_pad([tokenizer.cls_token_id] + aspect_indices  + [tokenizer.sep_token_id], max_length=max_length, pad_idx=tokenizer.pad_token_id)
                
                polarity = polarity_map[set_dict[k]['polarity']]

                data = {
                    'text_indices': text_indices,
                    'text_mask': text_mask,
                    #'aspect_position_mask': aspect_position_mask,
                    'aspect_boundary_indices': aspect_boundary_indices,
                    'aspect_indices': aspect_indices,
                    'aspect_mask': aspect_mask,
                    'polarity': polarity,
                }

                data_dict[set_type].append(data)

    return data_dict