# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
import torch.nn.functional as F

from layers.dynamic_rnn import DynamicRNN
from layers.position import PositionWeight, PositionDropout


class LSTM(nn.Module):
    def __init__(self, args, embedding):
        super(LSTM, self).__init__()
        self.embed = nn.Embedding.from_pretrained(torch.tensor(embedding, dtype=torch.float))
        self.weight = args.weight
        self.dropout = args.dropout
        if self.weight:
            self.pos_weight = PositionWeight()
        if self.dropout:
            self.pos_dropout = PositionDropout()
        self.lstm = DynamicRNN(args.embed_size,
                               args.hidden_size,
                               num_layers=1,
                               batch_first=True,
                               bidirectional=True)
        self.fc = nn.Linear(2 * args.hidden_size, args.polarity_size)

    def routine(self, text_indices, text_mask, aspect_boundary_indices, text_len, aspect_len):
        txt_embed = self.embed(text_indices)

        if self.weight:
            txt_embed = self.pos_weight(txt_embed, aspect_boundary_indices, text_len, aspect_len)
        if self.dropout and self.training:
            txt_embed = self.pos_dropout(txt_embed, aspect_boundary_indices, text_len, aspect_len)
        
        return txt_embed

    def forward(self, inputs):
        text_indices, text_mask, aspect_boundary_indices, aspect_indices, aspect_mask = inputs

        txt_len = torch.sum(text_mask, dim=1)
        asp_len = torch.sum(aspect_mask, dim=1)

        txt_embed = self.routine(text_indices, text_mask, aspect_boundary_indices, txt_len, asp_len)
        _, (hid, _) = self.lstm(txt_embed, txt_len)
        feat = torch.cat((hid[0], hid[1]), dim=-1)

        out = self.fc(feat)

        return out
