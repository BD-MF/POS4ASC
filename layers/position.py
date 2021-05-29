# -*- coding: utf-8 -*-

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

class PositionDropout(nn.Module):
    def __init__(self):
        super(PositionDropout, self).__init__()

    def forward(self, x, aspect_boundary_indices, text_len, aspect_len):
        batch_size, seq_len = x.shape[0], x.shape[1]
        prox = self.prox_mat(aspect_boundary_indices, text_len, aspect_len, batch_size, seq_len).to(x.device)
        x = prox.unsqueeze(2) * x

        return x

    @staticmethod
    def prox_mat(aspect_boundary_indices, text_len, aspect_len, batch_size, seq_len):
        aspect_boundary_indices = aspect_boundary_indices.cpu().numpy()
        text_len = text_len.cpu().numpy()
        aspect_len = aspect_len.cpu().numpy()
        prox = np.zeros((batch_size, seq_len), dtype=np.float32)
        for i in range(batch_size):
            context_len = text_len[i] - aspect_len[i]
            for j in range(aspect_boundary_indices[i, 0]):
                prox[i, j] = 1 - (aspect_boundary_indices[i, 0] - j) / context_len
            for j in range(aspect_boundary_indices[i, 0], aspect_boundary_indices[i, 1] + 1):
                prox[i, j] = 1 / context_len # or 0
            for j in range(aspect_boundary_indices[i, 1] + 1, text_len[i]):
                prox[i, j] = 1 - (j - aspect_boundary_indices[i, 1]) / context_len
            for j in range(text_len[i], seq_len):
                prox[i, j] = 0
        mask = np.random.binomial(n=1, p=prox, size=prox.shape)#np.random.random(prox.shape) < prox
        prox = mask / (prox + 1e-5)

        return torch.tensor(prox, dtype=torch.float)

class PositionWeight(nn.Module):
    def __init__(self):
        super(PositionWeight, self).__init__()

    def forward(self, x, aspect_boundary_indices, text_len, aspect_len):
        batch_size, seq_len = x.shape[0], x.shape[1]
        prox = self.prox_mat(aspect_boundary_indices, text_len, aspect_len, batch_size, seq_len).to(x.device)
        x = prox.unsqueeze(2) * x

        return x

    @staticmethod
    def prox_mat(aspect_boundary_indices, text_len, aspect_len, batch_size, seq_len):
        aspect_boundary_indices = aspect_boundary_indices.cpu().numpy()
        text_len = text_len.cpu().numpy()
        aspect_len = aspect_len.cpu().numpy()
        prox = np.zeros((batch_size, seq_len), dtype=np.float32)
        for i in range(batch_size):
            context_len = text_len[i] - aspect_len[i]
            for j in range(aspect_boundary_indices[i, 0]):
                prox[i, j] = 1 - (aspect_boundary_indices[i,0] - j) / context_len
            for j in range(aspect_boundary_indices[i, 0], aspect_boundary_indices[i,1]+1):
                prox[i, j] = 1 / context_len #prox[i].append(0); prox[i, j] = 1 / context_len
            for j in range(aspect_boundary_indices[i, 1] + 1, text_len[i]): #TODO 这里出了问题
                prox[i, j] = 1 - (j - aspect_boundary_indices[i, 1]) / context_len
            for j in range(text_len[i], seq_len):
                prox[i, j] = 0

        return torch.tensor(prox, dtype=torch.float)
