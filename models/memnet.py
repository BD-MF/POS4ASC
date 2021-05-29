# -*- coding: utf-8 -*-

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from layers.dynamic_rnn import DynamicRNN
from layers.position import PositionWeight, PositionDropout


class MemNet(nn.Module):
    def __init__(self, args, embedding):
        super(MemNet, self).__init__()
        self.num_hops = args.num_hops
        self.embed = nn.Embedding.from_pretrained(torch.tensor(embedding, dtype=torch.float))
        self.weight = args.weight
        self.dropout = args.dropout
        if self.weight:
            self.pos_weight = PositionWeight()
        if self.dropout:
            self.pos_dropout = PositionDropout()
        self.txt_lstm = DynamicRNN(args.embed_size,
                                   args.hidden_size,
                                   num_layers=1,
                                   batch_first=True,
                                   bidirectional=True)
        self.asp_lstm = DynamicRNN(args.embed_size,
                                   args.hidden_size,
                                   num_layers=1,
                                   batch_first=True,
                                   bidirectional=True)
        self.hop_fc = nn.Linear(2 * args.hidden_size, 2 * args.hidden_size)
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
        asp_embed = self.embed(aspect_indices)

        txt_out, (_, _) = self.txt_lstm(txt_embed, txt_len)
        txt_mask = text_mask[:, :txt_out.shape[1]]
        asp_out, (_, _) = self.asp_lstm(asp_embed, asp_len)
        asp_mask = aspect_mask[:, :asp_out.shape[1]]
        
        x = asp_out.sum(dim=1).div(asp_mask.sum(dim=1, keepdim=True)).unsqueeze(1)
        for _ in range(self.num_hops):
            x = self.hop_fc(x)
            a = torch.matmul(x, txt_out.transpose(1, 2))
            a = F.softmax(a.masked_fill(~txt_mask.unsqueeze(1).bool(), -float('inf')), dim=2)
            o = torch.matmul(a, txt_out)
            x = o + x

        out = self.fc(x.squeeze(1))

        return out
