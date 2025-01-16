import sys
sys.path.insert(0, '../')
from utilities.BART_utilities import *
import utilities.paper_functions as p_fct
import utilities.functions as fct

import pandas as pd
import numpy as np
import os

import time
from tqdm import tqdm

def init_bart(model_name='facebook/bart-large', finetune=False):
    global tokenizer, bart_model  # Declare global to modify the outer variables

    from transformers import BartTokenizer, BartForConditionalGeneration, AdamW, BartConfig
    tokenizer = BartTokenizer.from_pretrained(model_name, add_prefix_space=True)
    model = BartForConditionalGeneration.from_pretrained(model_name)
    
    if finetune:
        bart_model = LitModel.load_from_checkpoint("/home/pahelibhattacharya/HULK/Abhay/models/BART_large_IN_MCS.ckpt", learning_rate = 2e-5, tokenizer = tokenizer, model = model).to("cuda")
    else:
        bart_model = LitModel(learning_rate = 2e-5, tokenizer = tokenizer, model = model)

    return tokenizer, bart_model

def generate_summary_gpu(nested_sentences,p=0.2):

  '''
    Function to generate summaries from the list containing chunks of the document
    input:  nested_sentences - chunks
            p - Number of words in summaries per word in the document
    output: document summary
    '''
  device = 'cuda'
  summaries = []
  for nested in nested_sentences:
    l = int(p * len(nested.split(" ")))
    input_tokenized = tokenizer.encode(nested, truncation=True, return_tensors='pt')
    input_tokenized = input_tokenized.to(device)
    summary_ids = bart_model.model.to(device).generate(input_tokenized,
                                      length_penalty=0.01,
                                      min_length=l-5,
                                      max_length=l+5)
    output = [tokenizer.decode(g, skip_special_tokens=True, clean_up_tokenization_spaces=False) for g in summary_ids]
    summaries.append(output)
  summaries = [sentence for sublist in summaries for sentence in sublist]
  return summaries

def BART_summarize(text, tokenizer, bart_model, req_len=512):
    input_len = len(text.split(" "))
    req_len = 512 
    
    nested = p_fct.nest_sentences(text,1024)
    p = float(req_len/input_len)
    
    abs_summ = generate_summary_gpu(nested,p)
    abs_summ = " ".join(abs_summ)
    
    if len(abs_summ.split(" ")) > req_len:
        abs_summ = abs_summ.split(" ")
        abs_summ = abs_summ[:req_len]
        abs_summ = " ".join(abs_summ)

    return abs_summ
