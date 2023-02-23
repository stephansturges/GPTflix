import json
import pandas as pd
import numpy as np


# This is a sample converter that takes CSV data from a CSV table
# (in this case the Kaggle dataset here https://www.kaggle.com/datasets/cryptexcode/mpst-movie-plot-synopses-with-tags )
# and iterates through the rows of this data to make a JSONL file
# which can be loaded and processed by api_request_parallel_processor.py
# to generate the embeddings which we will use for vector search!

df2 = pd.read_csv('data_sample/mpst_5k.csv')


filename = "data_sample/all_plots_embeddings_maker.jsonl"
jobs = [{"model": "text-embedding-ada-002", "input": str(row[1])} for index, row in df2.iterrows()]
with open(filename, "w") as f:
    for job in jobs:
        json_string = json.dumps(job)
        f.write(json_string + "\n")
