import pandas as pd
import tokenizer
import transformers
import nltk
import openai
import tiktoken
encoding = tiktoken.get_encoding("cl100k_base")
import pickle
import numpy as np  
import os
import nltk
from dotenv import load_dotenv
import warnings


# This is a sample converter that takes CSV data from a table
# (in this case the Kaggle dataset here https://www.kaggle.com/datasets/cryptexcode/mpst-movie-plot-synopses-with-tags )
# and converts this data into a CSV that contains a single column 
# with a block of text that we want to make accessible on our Pinecone database


def num_tokens_from_string(string, encoding_name):
    """Returns the number of tokens in a text string."""
    encoding = tiktoken.get_encoding(encoding_name)
    num_tokens = len(encoding.encode(string))
    return num_tokens

warnings.filterwarnings("ignore", category=DeprecationWarning) 
warnings.filterwarnings("ignore", category=FutureWarning) 


# original loader function for full file, not shared in repo because it's too large
#df = pd.read_csv(filepath_or_buffer='data/mpst_full_data.csv', sep="," , header=0, dtype="string", encoding="utf-8" )

# loader function for sample data
df = pd.read_csv(filepath_or_buffer='data/mpst_full_data.csv', sep="," , header=0, dtype="string", encoding="utf-8" )


df["gpttext"] = "Title: " + df["title"].astype(str) + \
                " tags: " + df["tags"].astype(str) +\
                " Plot / story / about: " + df["plot_synopsis"].astype(str)

df = df.drop(df.columns[[0, 1, 2, 3, 4, 5]], axis=1)   
df.to_csv('data/mpst_converted.csv')

