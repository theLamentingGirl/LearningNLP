import torch
import torch.nn as nn
from math import * 
from .transformer_elements import *

class DecoderBlock(nn.Module):

    def __init__(self, self_attention_block: MultiHeadAttentionBlock, cross_attention_block: MultiHeadAttentionBlock, feed_forward_block: FeedForwardBlock, dropout: float,encoder: Encoder):
        super().__init__()
        self.self_attention_block = self_attention_block
        self.cross_attention_block = cross_attention_block
        self.feed_forward_block = feed_forward_block
        self.residual_connections = nn.ModuleList([ResidualConnection(dropout)] for _ in range(3))#in the decoder we have 3 residual connections
        self.encoder = encoder

    def forward(self,x,encoded_output, src_mask, tgt_mask): #src_mask is for encoder; tgt_mask is for decoder
        x = self.residual_connections[0](x, lambda x: self.self_attention_block(x,x,x, src_mask))
        x = self.residual_connections[1](x, lambda x: self.cross_attention_block(encoded_output,x,encoded_output,src_mask))
        x = self.residual_connections[2](x,self.feed_forward_block)

class Decoder(nn.Module):

    def __init__(self,blocks:nn.ModuleList) -> None:
        super().__init__()
        self.blocks = blocks
        self.norm = LayerNormalisation()

    def forward(self, x, encoded_output,src_mask,tgt_mask):
        for block in self.blocks:
            x = block(x,encoded_output,src_mask,tgt_mask)
        return self.norm(x)