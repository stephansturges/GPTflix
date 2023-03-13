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
# (in this case the Kaggle dataset here https://www.kaggle.com/datasets/jrobischon/wikipedia-movie-plots )
# and converts this data into a CSV that contains a single column 
# with a block of text that we want to make accessible on our Pinecone database


def num_tokens_from_string(string, encoding_name):
    """Returns the number of tokens in a text string."""
    encoding = tiktoken.get_encoding(encoding_name)
    num_tokens = len(encoding.encode(string))
    return num_tokens

warnings.filterwarnings("ignore", category=DeprecationWarning) 
warnings.filterwarnings("ignore", category=FutureWarning) 


openai.api_key =  #### SET YOUR API KEY HER



COMPLETIONS_MODEL = "text-davinci-003"
EMBEDDING_MODEL = "text-embedding-ada-002"


df = pd.read_csv(filepath_or_buffer='data_sample/wiki_movie_plots_small.csv', sep="," , header=0, dtype="string", encoding="utf-8" )

df["gpttext"] = "Title: " + df["Title"].astype(str) + \
                " Year: " + df["Release Year"].astype(str) + \
                " Cast: " + df["Cast"].astype(str).replace("<NA>", "unknown") +    \
                " Director: " + df["Director"].astype(str) + \
                " Country of production: " + df["Origin/Ethnicity"].astype(str) +\
                " Genre: " + df["Genre"].astype(str) +\
                " wiki: " + df["Wiki Page"].astype(str) +\
                " Plot / story / about: " + df["Plot"].astype(str).replace("[1]", "FUCK")

df = df.drop(df.columns[[0, 1, 2, 3, 4, 5, 6, 7]], axis=1)   
df.to_csv('data/wiki_plots_small.csv')


