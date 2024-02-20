import torch
import torch.nn as nn
from torch.utils.data import Dataset,DataLoader, random_split
from datasets import load_dataset
from .tokenizer import build_tokenizer

class BilingualDataset(Dataset):

    def __init__(self,ds,tokenizer_src,tokenizer_tgt,src_lang,tgt_lang,seq_len) -> None:
        super().__init__()

        self.ds = ds
        self.tokenizer_src = tokenizer_src
        self.tokenizer_tgt = tokenizer_tgt
        self.src_lang = src_lang
        self.tgt_lang = tgt_lang
        self.seq_len = seq_len

        self.sos_token = torch.tensor([tokenizer_tgt.token_to_id("[SOS]")], dtype=torch.int64)
        self.eos_token = torch.tensor([tokenizer_tgt.token_to_id("[EOS]")], dtype=torch.int64)
        self.pad_token = torch.tensor([tokenizer_tgt.token_to_id("[PAD]")], dtype=torch.int64)

    def __getitem__(self, idx):

        src_target_pair = self.ds[idx]
        src_text = src_target_pair['translation'][self.src_lang]
        tgt_text = src_target_pair['translation'][self.tgt_lang]

        # Transform the text into tokens
        enc_input_tokens = self.tokenizer_src.encode(src_text).ids
        dec_input_tokens = self.tokenizer_tgt.encode(tgt_text).ids

        # each sentence can have different lengths st. they are always less than the seq_len. To reach seq_len, we add padding
        enc_num_padding_tokens = self.seq_len - len(enc_input_tokens) - 2  # We subtract -2 because we add start and end of sentence token
        dec_num_padding_tokens = self.seq_len - len(dec_input_tokens) - 1 #in decoder we only subtract -1 since we only add start of sentence token

        if enc_num_padding_tokens < 0 or dec_num_padding_tokens <0:
            raise ValueError('Sentence is longer than sequence length')
        
        encoder_input = torch.cat([
            self.sos_token,
            torch.tensor(enc_input_tokens,dtype=torch.int64),
            self.eos_token,
            torch.tensor([self.pad_token]*enc_input_tokens,dtype=torch.int64)
        ])

        #only start of sentence token
        decoder_input = torch.cat(
            [
                self.sos_token,
                torch.tensor(dec_input_tokens, dtype=torch.int64),
                torch.tensor([self.pad_token] * dec_num_padding_tokens, dtype=torch.int64),
            ],
            dim=0,
        )

        # only end of sentence token added
        label = torch.cat(
            [
                torch.tensor(dec_input_tokens, dtype=torch.int64),
                self.eos_token,
                torch.tensor([self.pad_token] * dec_num_padding_tokens, dtype=torch.int64),
            ],
            dim=0,
        )

        assert encoder_input.size(0) == self.seq_len
        assert decoder_input.size(0) == self.seq_len
        assert label.size(0) == self.seq_len
        
        #return a dictionary
        return{
            
            "encoder_input": encoder_input,  # (seq_len)
            "decoder_input": decoder_input,  # (seq_len)
            "encoder_mask": (encoder_input != self.pad_token).unsqueeze(0).unsqueeze(0).int(), # (1, 1, seq_len)
            "decoder_mask": (decoder_input != self.pad_token).unsqueeze(0).int() & causal_mask(decoder_input.size(0)), # (1, seq_len) & (1, seq_len, seq_len),
            "label": label,  # (seq_len)
            "src_text": src_text,
            "tgt_text": tgt_text,
        }

def causal_mask(size):
    mask = torch.triu(torch.ones(1,size,size),diagonal=1) #trui method returns only the upper triangular part of a matrix, everything below the diagonal & diagonal itself is returned as 0

    return mask == 0 #everything not zero is returns False(0) and everything true returns 1. The opp of trui is what we need

def get_ds(config):
    ds_raw = load_dataset('opus_books',f'{config['lang_src']}-{config['lang_tgt']}',split='train')

    #splitting 90%training and 10%validation
    train_ds_size = int(0.9*len(ds_raw))
    val_ds_size = len(ds_raw) - train_ds_size

    train_ds_raw, val_ds_raw = random_split(ds_raw,[train_ds_size,val_ds_size])

    #build tokenizers
    tokenizer_src = build_tokenizer(config,ds_raw,config['lang_src'])
    tokenizer_tgt = build_tokenizer(config,ds_raw,config['lang_tgt'])

    #setting-up dataset into training and val
    train_ds = BilingualDataset(train_ds_raw, tokenizer_src, tokenizer_tgt, config['lang_src'], config['lang_tgt'], config['seq_len'])
    val_ds = BilingualDataset(val_ds_raw, tokenizer_src, tokenizer_tgt, config['lang_src'], config['lang_tgt'], config['seq_len'])

    # Find the maximum length of each sentence in the source and target sentence
    max_len_src = 0
    max_len_tgt = 0

    for item in ds_raw:
        src_ids = tokenizer_src.encode(item['translation'][config['lang_src']]).ids
        tgt_ids = tokenizer_tgt.encode(item['translation'][config['lang_tgt']]).ids
        max_len_src = max(max_len_src, len(src_ids))
        max_len_tgt = max(max_len_tgt, len(tgt_ids))

    print(f'Max length of source sentence: {max_len_src}')
    print(f'Max length of target sentence: {max_len_tgt}')
    

    train_dataloader = DataLoader(train_ds, batch_size=config['batch_size'], shuffle=True)
    val_dataloader = DataLoader(val_ds, batch_size=1, shuffle=True)

    return train_dataloader, val_dataloader, tokenizer_src, tokenizer_tgt

        

