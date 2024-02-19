import torch
import torch.nn as nn
from math import * 

#====================== TRANSFORMER ELEMENTS =================================================

class InputEmbeddings(nn.Module):
    def __init__(self, d_model:int, vocab_size: int):
        super().__init__()
        self.d_model = d_model #length/dimension of the feature vector
        self.vocab_size = vocab_size #num of words in our vocab. Note, we're not doing Byte pair encoding
        self.embedding = nn.Embedding(vocab_size, d_model)

    def forward(self,x):#this is what passes the data across the nn layers during feedforward and backprop
        return self.embedding(x) * sqrt(self.d_model) #specified in attn is all you need paper

class PosEncodings(nn.Module):
    def __init__(self, d_model: int, seq_len:int, dropout: float):
        super().__init__()
        self.d_model = d_model
        self.seq_len = seq_len
        self.dropout = nn.Dropout(dropout)

        #create a matrix of shape (seq_len, d_model)
        pe = torch.zeros(seq_len,d_model)
        position = torch.arange(0, seq_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0,d_model,2).float() * (-log(1000.0)/d_model))
                             
        #sin for even terms
        pe[:, 0::2] = torch.sin(position * div_term)

        #cos for odd terms
        pe[:, 1::2] = torch.cos(position * div_term)

        pe = pe.unsqueeze(0) #adding extra dim for accomodating batch dim

        self.register_buffer('pe',pe) #tensor saved w weights that's not trained

    def forward(self,x):
        
        x = x+ (self.pe[:, :x.shape[1],:]).requires_grad_(False) #untrained parameter
        return self.dropout(x)

#self implementing layer norm whereas pytorch has a func for it
class LayerNormalisation(nn.Module):

    def __init__(self, eps:float = 10**-6):
        super().__init__()
        self.eps = eps 
        self.gamma = nn.Parameter(torch.ones(1)) #scaling
        self.beta = nn.Parameter(torch.zeros(1)) #shift

    def forward(self,x):
        mean = x.mean(dim = -1, keepdim=True) #along the 2nd column dim, here we take full mean not moving, if dim=0, then it's batch norm
        std = x.std(dim = -11,keepdim=True)

        return self.gamma * (x - mean)/ (std+self.eps) + self.beta #layernorm formula

 
class FeedForwardBlock(nn.Module):
    def __init__(self, d_model: int, d_ff:int, dropout: float):
        super().__init__()
        self.linear1 = nn.Linear(d_model,d_ff) #input shape = d_model, output_shape = d_ff
        self.dropout = dropout
        self.linear2 = nn.Linear(d_ff,d_model) #reshaping it back to d_model dim
        
    def forward(self,x):
        # (batch,seq_len,d_model) --> (batch,seq_len, d_ff) --> (batch,seq_len,d_model)
        return self.linear2(self.dropout(torch.relu(self.linear1(x))))


class MultiHeadAttentionBlock(nn.Module):
    def __init__(self,d_model:int,h:int,dropout:float):
        super().__init__()
        self.d_model = d_model
        self.h = h #number of heads d_model/h = dk = dv
        
        assert d_model % h == 0, "d_model is not divisible by h"

        self.dk = d_model //h

        self.wk = nn.Linear(d_model,d_model)
        self.wq = nn.Linear(d_model,d_model)
        self.wv = nn.Linear(d_model,d_model)

        self.wo = nn.Linear(d_model, d_model) 
        
        self.dropout = nn.Dropout(dropout)

    @staticmethod #so that we can call ClassName.attention without instantiation 
    def attention(key,query,value,mask,dropout: nn.Dropout): #the parameter to be passed to the dropout var here is a layer
        dk = query.shape[-1]

        attention_scores = (query @ key.transpose(-2,-1)) / sqrt(dk)

        if mask is not None: #applying mask
            attention_scores.masked_fill_(mask==0, -1e9)

        if dropout is not None: #applying dropout if it's specified after calculating attn scores
            attention_scores = dropout(attention_scores)

        return (attention_scores @ value), attention_scores

    def forward(self,k,q,v,mask):
        Q = self.wq(q) #(batch,seq_len,d_model)
        K = self.wk(k) # ""
        V = self.wv(v) #""

        # splitting into multiple heads
        # (batch, seq_len, d_model) --> (batch, seq_len, h, dk) --> (batch, h, seq_len, dk)
        Q = Q.view(Q.shape[0],Q.shape[1],self.h,self.dk).transpose(1,2)
        K = K.view(K.shape[0],Q.shape[1],self.h,self.dk).transpose(1,2)
        V = V.view(V.shape[0],V.shape[1],self.h,self.dk).transpose(1,2)

        x, self.attention_scores = MultiHeadAttentionBlock.attention(k,q,v,mask,self.dropout)

        #concat the multiple heads
        # (batch,h,seq_len,dk) --> (batch,seq_len,h,dk) --> (batch,seq_len,d_model)

        x = x.transpose(1,2).contiguous().view(x.shape[0],-1,self.h * self.dk)
        #contiguous is needed because we first use transpose to modify tensor before. This just rearrages the contents of how the tensor is stored in memory. So, newly created tensor like x != x above. contiguos makes it set correctly in memory

        return self.wo(x)
    

class ResidualConnection(nn.Module):

    def __init__(self, dropout: float): #adding a pointer -> None is the same as ignoring it. init method never returns anything and if you deliberately try to do that compiler will raise error with this formulation
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        self.norm = LayerNormalisation() #NOTE: residual connection classs comes combined with layer norm.

    def forward(self, x,sublayer):
        return x + self.dropout(sublayer(self.norm(x)))
    

#=================================== ENCODER BLOCK combining all the above elements==========================================
    
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
    

#========================DECODER=======================================================
    
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
    

#============= PUTTING EVERYTHING TOGETHER================================
    
class ProjectionLayer(nn.Module): #to change the output size of embedded values to vocab size - feedforward linear layer

    def __init__(self, d_model: int, vocab_size) -> None: #vocab size refers to how many tokens are being generated by your tokenizer; source tokenizer and target tokenizer can be combined and represented or differently represented
        super().__init__()
        self.projection = nn.Linear(d_model,vocab_size)

    def forward(self,x):
        #(batch,seq_len,d_model) --> (batch, seq_len, vocab_size)
        return torch.log_softmax(self.projection(x),dim=-1)
    
#==========================TRANSFORMERRRRRR==============================================
    
class Transformer(nn.Module):

    def __init__(self, encoder: Encoder, decoder: Decoder, src_embed: InputEmbeddings, tgt_embed: InputEmbeddings, src_pos:PosEncodings,tgt_pos:PosEncodings, projection_layer:ProjectionLayer) -> None:
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.src_embed = src_embed
        self.tgt_embed = tgt_embed
        self.src_pos = src_pos
        self.tgt_pos = tgt_pos
        self.projection_layer = projection_layer

    def encode(self,src,src_mask):
        src = self.src_embed(src)#embed the src language lines
        src = self.src_pos(src) #add postional encoding

        return self.encoder(src,src_mask)
    
    def decode(self,encoder_output,src_mask,tgt,tgt_mask):
        tgt = self.tgt_embed(tgt)
        tgt = self.tgt_pos(tgt)

        return self.decoder(tgt,encoder_output,src_mask,tgt_mask)
    
    def project(self,x):
        return self.projection_layer(x)

    

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
    




    
        

    

