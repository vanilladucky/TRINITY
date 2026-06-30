import torch
import torch.nn as nn

from networks.sl_module.transformer import Transformer
from networks.sl_module.score_net import ScoreFCN

class SL_module(nn.Module):

    def __init__(self, input_dim, depth, heads, mlp_dim, dropout_ratio, pos_type = 'rope'):
        super(SL_module, self).__init__()
        
        self.transformer = Transformer(dim=input_dim, depth=depth, heads=heads, mlp_dim=mlp_dim, dropout=dropout_ratio, pos_type=pos_type)
        self.score_model_emotion = ScoreFCN(emb_dim=512, dropout_ratio = dropout_ratio)
        self.score_model_event = ScoreFCN(emb_dim=512, dropout_ratio = dropout_ratio)
        self.score_model_nature = ScoreFCN(emb_dim=512, dropout_ratio = dropout_ratio)
        
    def forward(self, x, mask, train_type):

        transformed_emb = self.transformer(x)
        if train_type=="emotion":
            score = self.score_model_emotion(transformed_emb).squeeze(-1)
        elif train_type=='event':
            score = self.score_model_event(transformed_emb).squeeze(-1)
        else:
            score = self.score_model_nature(transformed_emb).squeeze(-1)
        
        score = torch.sigmoid(score)

        return score, -100

    def load_state_dict(self, state_dict, strict=True):
        # Support both the nested custom checkpoint format and a flat state_dict.
        if 'transformer' in state_dict.keys():
            print("[SL_module] Loading from nested dictionary keys...")
            self.transformer.load_state_dict(state_dict['transformer'], strict=strict)
            self.score_model_emotion.load_state_dict(state_dict['score_model_emotion'], strict=strict)
            self.score_model_event.load_state_dict(state_dict['score_model_event'], strict=strict)
        else:
            super(SL_module, self).load_state_dict(state_dict, strict=strict)

