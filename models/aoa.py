# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
import torch.nn.functional as F

from layers.dynamic_rnn import DynamicRNN
from layers.position import PositionWeight, PositionDropout

class AOA(nn.Module):
    def __init__(self, args, embedding):
        super(AOA, self).__init__()
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

        attn = torch.matmul(txt_out, asp_out.transpose(1, 2))
        alpha = attn.masked_fill(~txt_mask.unsqueeze(2).bool(), -10000.0)
        alpha = F.softmax(alpha, dim=1)  # col-wise
        beta = attn.masked_fill(~asp_mask.unsqueeze(1).bool(), -10000.0)
        beta = F.softmax(beta, dim=2)  # row-wise
        beta = beta.masked_fill(~txt_mask.unsqueeze(2).bool(), 0.0)
        beta_avg = beta.sum(dim=1).div(txt_mask.sum(dim=1, keepdim=True)).unsqueeze(1)
        gamma = torch.matmul(alpha, beta_avg.transpose(1, 2))
        feat = torch.matmul(txt_out.transpose(1, 2), gamma).squeeze(2)

        out = self.fc(feat)

        return out
