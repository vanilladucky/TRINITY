# networks/sl_module/sl_module_split_heads.py
"""
SL Module with Split-Head Transformer for Multi-Task Learning (3 Tasks)

This module uses a transformer with split attention heads:
- 4 shared heads: Learn common patterns for emotion, event, and nature
- 4 emotion-specific heads
- 4 event-specific heads
- 4 nature-specific heads
"""

import torch
import torch.nn as nn

from networks.sl_module.transformer_split_heads import SplitHeadTransformer, SplitHeadTransformerSeparateFFN
from networks.sl_module.score_net import ScoreFCN


class SL_module_split_heads(nn.Module):
    def __init__(
        self,
        input_dim=512,
        depth=5,
        total_heads=8,
        shared_heads=4,
        mlp_dim=3072,
        dropout_ratio=0.5,
        separate_ffn=False,
        pos_type='rope',
    ):
        super(SL_module_split_heads, self).__init__()
        
        self.input_dim = input_dim
        self.depth = depth
        self.total_heads = total_heads
        self.shared_heads = shared_heads
        
        # Choose transformer variant
        TransformerClass = SplitHeadTransformerSeparateFFN if separate_ffn else SplitHeadTransformer
        
        self.transformer = TransformerClass(
            dim=input_dim,
            depth=depth,
            total_heads=total_heads,
            shared_heads=shared_heads,
            mlp_dim=mlp_dim,
            dropout=dropout_ratio,
            pos_type=pos_type,
            rope_base=1000.0,
            time_scale=0.5,
        )
        
        # Separate score heads for each task
        score_input = input_dim // 2
        self.score_model_emotion = ScoreFCN(emb_dim=input_dim)
        self.score_model_event = ScoreFCN(emb_dim=input_dim)
        self.score_model_nature = ScoreFCN(emb_dim=input_dim) 
        
    def forward(self, x, mask=None, train_type="emotion"):
        """
        Forward pass through the split-head transformer.
        train_type: "emotion", "event", or "nature"
        """
        # Pass through transformer with task-specific heads
        transformed_emb, last_shared, last_task, shared_attn, task_attn = self.transformer(x, mask=mask, task_type=train_type)        

        # element wise multiplication to reduce to 256
        # print(f"last_shared: {last_shared.shape} | last_task: {last_task.shape}")
        # element_wise = torch.mul(last_shared, last_task)
        
        
        # Apply task-specific score head
        if train_type == "emotion":
            score = self.score_model_emotion(transformed_emb).squeeze(-1)
        elif train_type == "event":
            score = self.score_model_event(transformed_emb).squeeze(-1)
        else: # nature
            score = self.score_model_nature(transformed_emb).squeeze(-1)
        
        score = torch.sigmoid(score)
        
        return score, transformed_emb, shared_attn, task_attn
    
    def forward_all(self, x, mask=None):
        """
        Forward pass for ALL tasks simultaneously (useful for inference).
        """
        # Emotion path
        emotion_emb = self.transformer(x, mask=mask, task_type="emotion")[0]
        emotion_score = torch.sigmoid(self.score_model_emotion(emotion_emb).squeeze(-1))
        
        # Event path
        event_emb = self.transformer(x, mask=mask, task_type="event")[0]
        event_score = torch.sigmoid(self.score_model_event(event_emb).squeeze(-1))

        # Nature path
        nature_emb = self.transformer(x, mask=mask, task_type="nature")[0]
        nature_score = torch.sigmoid(self.score_model_nature(nature_emb).squeeze(-1))
        
        return emotion_score, event_score, nature_score
    
    def get_shared_parameters(self):
        return self.transformer.get_shared_parameters()
    
    def get_emotion_parameters(self):
        params = list(self.transformer.get_emotion_specific_parameters())
        params.extend(self.score_model_emotion.parameters())
        return params
    
    def get_event_parameters(self):
        params = list(self.transformer.get_event_specific_parameters())
        params.extend(self.score_model_event.parameters())
        return params

    def get_nature_parameters(self):
        params = list(self.transformer.get_nature_specific_parameters())
        params.extend(self.score_model_nature.parameters())
        return params
    
    def load_state_dict(self, state_dict, strict=True):
        if 'transformer' in state_dict.keys():
            print("[SL_module_split_heads] Loading from nested dictionary keys...")
            self.transformer.load_state_dict(state_dict['transformer'], strict=strict)
            self.score_model_emotion.load_state_dict(state_dict['score_model_emotion'], strict=strict)
            self.score_model_event.load_state_dict(state_dict['score_model_event'], strict=strict)
            
            # Handle backward compatibility for checkpoints that don't have nature yet
            if 'score_model_nature' in state_dict:
                self.score_model_nature.load_state_dict(state_dict['score_model_nature'], strict=strict)
            else:
                print("Warning: 'score_model_nature' not found in checkpoint. Initializing randomly.")
        else:
            super(SL_module_split_heads, self).load_state_dict(state_dict, strict=strict)