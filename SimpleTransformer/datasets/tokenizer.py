import torch
import torch.nn as nn
from pathlib import Path #useful for creating absolute paths given relative path

from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.trainers import WordLevelTrainer
from tokenizers.pre_tokenizers import Whitespace

def get_all_sentences(ds,lang): #to get all the elements from the dataset ds of a single lang
    for item in ds:
        yield item['translation'][lang]

def build_tokenizer(config,ds,lang):
    tokenizer_path = Path(config['tokenizer_file']).format(lang)

    if not Path.exists(tokenizer_path): #if there's no pre-existing file for tokenizers, we create one here

        tokenizer = Tokenizer(WordLevel(unk_token='[UNK]')) #to map unknown words to unk 
        tokenizer.pre_tokenizer = Whitespace() 
        trainer = WordLevelTrainer(special_tokens=["[UNK]","[PAD]","[SOS]","[EOS]"],min_frequency=2)
        tokenizer.train_from_iterator(get_all_sentences(ds,lang))
        tokenizer.save(str(tokenizer_path))

    else: #else we just initialise the tokenizer from the file

        tokenizer = Tokenizer.from_file(str(tokenizer_path))