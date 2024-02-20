import torch
import torch.nn as nn
from math import * 

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
    
class ProjectionLayer(nn.Module): #to change the output size of embedded values to vocab size - feedforward linear layer

    def __init__(self, d_model: int, vocab_size) -> None: #vocab size refers to how many tokens are being generated by your tokenizer; source tokenizer and target tokenizer can be combined and represented or differently represented
        super().__init__()
        self.projection = nn.Linear(d_model,vocab_size)

    def forward(self,x):
        #(batch,seq_len,d_model) --> (batch, seq_len, vocab_size)
        return torch.log_softmax(self.projection(x),dim=-1)