# -*- coding: utf-8 -*-

from models.lstm import LSTM
from models.lstmattn import LSTMAttn
from models.ian import IAN
from models.memnet import MemNet
from models.aoa import AOA
from models.roberta import Roberta

MODEL_CLASS_MAP = {
    'lstm': LSTM,
    'lstmattn': LSTMAttn,
    'ian': IAN,
    'memnet': MemNet,
    'aoa': AOA,
    'roberta': Roberta,
}

INPUT_FIELDS_MAP = {
    'lstm': ['text_indices', 'text_mask', 'aspect_boundary_indices', 'aspect_indices', 'aspect_mask'],
    'lstmattn': ['text_indices', 'text_mask', 'aspect_boundary_indices', 'aspect_indices', 'aspect_mask'],
    'ian': ['text_indices', 'text_mask', 'aspect_boundary_indices', 'aspect_indices', 'aspect_mask'],
    'memnet': ['text_indices', 'text_mask', 'aspect_boundary_indices', 'aspect_indices', 'aspect_mask'],
    'aoa': ['text_indices', 'text_mask', 'aspect_boundary_indices', 'aspect_indices', 'aspect_mask'],
    'roberta': ['text_indices', 'text_mask', 'aspect_boundary_indices', 'aspect_indices', 'aspect_mask'],
}

def get_model(model_name):
    return MODEL_CLASS_MAP[model_name], INPUT_FIELDS_MAP[model_name]