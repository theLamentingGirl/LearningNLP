import torch
import torch.nn as nn
from math import * 
from ..src.decoder import *
from ..src.encoder import *
from ..src.transformer import *
from ..src.transformer_elements import *


def build_transformer(src_vocab_size:int,tgt_vocab_size:int,src_seq_len: int, tgt_seq_len: int,d_model: int = 512, num_blocks:int = 6, heads: int =8, dropout:float = 0.1,d_ff:int = 2048):
    #create embedding layers
    src_embed = InputEmbeddings(d_model,src_vocab_size)
    tgt_embed = InputEmbeddings(d_model,tgt_vocab_size)

    #create postional encoding layers
    src_pos = PosEncodings(d_model,src_seq_len, dropout)
    tgt_pos = PosEncodings(d_model,tgt_seq_len,dropout)

    #create encoder blocks
    encoder_blocks = []

    for _ in range(num_blocks):
        #creating instances of classes needed to build a transformer
        encoder_self_attention_block = MultiHeadAttentionBlock(d_model,heads,dropout)
        feed_forward_block = FeedForwardBlock(d_model,d_ff,dropout)
        encoder_block = EncoderBlock(encoder_self_attention_block,feed_forward_block,dropout)
        encoder_blocks.append(encoder_block)

    decoder_blocks = []
    #create decoder blocks
    for _ in range(num_blocks):
        #creating instances of classes needed to build a transformer
        decoder_self_attention_block = MultiHeadAttentionBlock(d_model,heads,dropout)
        decoder_cross_attention_block = MultiHeadAttentionBlock(d_model,heads,dropout)
        feed_forward_block = FeedForwardBlock(d_model,d_ff,dropout)
        decoder_block = DecoderBlock(decoder_self_attention_block,decoder_cross_attention_block,feed_forward_block,dropout)
        decoder_blocks.append(decoder_block)

    #creating encoder and decoder with num_blocks 
    encoder = Encoder(nn.ModuleList(encoder_block))
    decoder = Decoder(nn.ModuleList(decoder_block))

    #creating projection later
    projection_layer = ProjectionLayer(d_model,tgt_vocab_size)

    #create the transformer
    transformer = Transformer(encoder,decoder,src_embed,tgt_embed,src_pos,tgt_pos,projection_layer)

    #Initialise parameters

    for p in transformer.parameters():
        if p.dim() >1:
            nn.init.xavier_uniform_(p)

    return transformer