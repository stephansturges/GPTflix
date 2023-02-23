import json
import pandas as pd
import numpy as np
import os


filename = "data/all_plots_embeddings_COMPLETE.jsonl"

with open(os.path.abspath(filename), 'r', encoding='utf-8') as f:
  data = [json.loads(line) for line in open(os.path.abspath(filename), 'r', encoding='utf-8')]


print("OPENED JSONL FILE WITH EMBEDDINS")

def flattenizer(a):
    return (a[0],) + tuple(a[1])

dataframe_with_text_and_embeddings = pd.DataFrame()

mydata_expanded_flat = [flattenizer([json.loads(json.dumps(line))[0]['input'],json.loads(json.dumps(line))[1]['data'][0]['embedding']]) for line in data]

print("CONVERTED JSONL FLAT ARRAY")


def columns_index_maker(): 
	column_names =[]
	column_names.append('gpttext')
	for _ in range(1536):
		column_names.append(str(_))

	return column_names

all_the_columns = columns_index_maker()

df = pd.DataFrame(mydata_expanded_flat , columns=all_the_columns)

print("CONVERTED BIG ARRAY TO DATAFRAME")


def chunker(seq, size):
    return (seq[pos:pos + size] for pos in range(0, len(seq), size))


# if you are working on very large files, for example uploading all of wikipedia
# these indexes can get very very chonky with the embeddings appended (like >400Gb)
# This is why we chunk through the dataframe and append pieces to the CSV to avoid
# running out of memory

def chonk_dataframe_and_make_csv_with_embeds(pddf, outputfile, chunks):
    for i, chunk in enumerate(chunker(pddf, chunks)):
        print("CHONKING TO CSV No: " + str(i))
        document_embeddings_i = pd.DataFrame(chunk)
        document_embeddings_i.to_csv(
            outputfile, mode='a', index=False, header=False if i > 0 else True
        )

chonk_dataframe_and_make_csv_with_embeds(df,'data/all_plots_embeddings_COMPLETE.csv', 1000)


