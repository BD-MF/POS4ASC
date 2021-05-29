# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
import torch.nn.functional as F

from layers.position import PositionWeight, PositionDropout


class Roberta(nn.Module):
    def __init__(self, args, embedding):
        super(Roberta, self).__init__()
        self.embed = embedding
        self.weight = args.weight
        self.dropout = args.dropout
        if self.weight:
            self.pos_weight = PositionWeight()
        if self.dropout:
            self.pos_dropout = PositionDropout()
        self.ffn = nn.Sequential(
            nn.Linear(args.bert_size, args.bert_size),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(args.bert_size, args.polarity_size),
        )

    def routine(self, text_indices, text_mask, aspect_boundary_indices, text_len, aspect_len):
        txt_embed = self.embed(text_indices, attention_mask=text_mask)[0]

        if self.weight:
            txt_embed = self.pos_weight(txt_embed, aspect_boundary_indices, text_len, aspect_len)
        if self.dropout and self.training:
            txt_embed = self.pos_dropout(txt_embed, aspect_boundary_indices, text_len, aspect_len)
        
        return txt_embed

    def forward(self, inputs):
        text_indices, text_mask, aspect_boundary_indices, aspect_indices, aspect_mask = inputs

        txt_len = torch.sum(text_mask, dim=1)
        asp_len = torch.sum(aspect_mask, dim=1) - 2

        txt_embed = self.routine(text_indices, text_mask, aspect_boundary_indices, txt_len, asp_len)
        txt_embed = txt_embed.masked_fill(~text_mask.unsqueeze(2).bool(), -10000.0)

        feat, _ = txt_embed.max(dim=1)

        out = self.ffn(feat)

        return out