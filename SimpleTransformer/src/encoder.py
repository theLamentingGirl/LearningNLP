import torch
import torch.nn as nn
from math import * 
from .transformer_elements import *

class EncoderBlock(nn.Module):
    
    def __init__(self, self_attention_block: MultiHeadAttentionBlock, feed_forward_block: FeedForwardBlock, dropout: float) -> None:
        super().__init__()
        self.self_attention_block = self_attention_block
        self.feed_forward_block = feed_forward_block
        self.residual_connections = nn.ModuleList([ResidualConnection(dropout)] for _ in range(2))

    def forward(self, x,src_mask):
        x = self.residual_connections[0](x, lambda x: self.self_attention_block(x,x,x, src_mask)) #format of lambda function: var = lambda argument(s) = expression; can also be used without assigning 
        x = self.residual_connections[1](x,self.feed_forward_block)

        return x
    
class Encoder(nn.Module): #to implement the encoder block n times 

    def __init__(self,blocks:nn.ModuleList) -> None:
        super().__init__()
        self.blocks = blocks
        self.norm = LayerNormalisation()

    def forward(self,x,mask):
        for block in self.blocks:
            x = block(x,mask)
        return self.norm(x) #why layernorm here?