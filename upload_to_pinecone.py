import pinecone
import csv
import numpy as np
from multiprocessing import Pool, cpu_count


pinecone_api_key = "" # SET YOUR PINECONE API KEY HERE
pinecone.init(api_key=pinecone_api_key, environment="us-east1-gcp")

# Define the name of the index and the dimensionality of the embeddings
index_name = "400kmovies"
dimension = 1536

# Create an empty index if required
#   pinecone.create_index(name=index_name, dimension=dimension)

index = pinecone.Index(index_name)

print("Pinecone index info:")
print(pinecone.whoami())

print("How many vectors are in the index:")
print(index.describe_index_stats()['total_vector_count'])


# We are using a function to limit the metadata character length to 4000 here. 
# The reason is that there is a limit on the size of metadata that you can append 
# to a vector on pinecone, currently I believe it's 10Kb... You could go with more
# than 4000 and give it a try.

def get_first_4000_chars(s):
    if len(s) > 4000:
        return s[:4000]
    else:
        return s

# Define a function to upsert embeddings in batches

def upsert_embeddings_batch(starting_index, data_batch, index_offset):
    # Convert the data to a list of Pinecone upsert requests
    upsert_requests = [
        (str(starting_index + i + index_offset), embedding, {'text': get_first_4000_chars(row[0])})   # taking 1500 first characters because of meta size limit
        for i, row in enumerate(data_batch)
        for embedding in [np.array([float(x) for x in row[1:]]).tolist()]
    ]

    # Upsert the embeddings in batch
    upsert_response = index.upsert(
        vectors=upsert_requests,
        namespace='movies'
    )

    return upsert_response

# Load the data from the CSV file
with open("data/embedded_converted_flix_complete_big_first_try_good.csv") as f:
    reader = csv.reader(f)
    next(reader)  # skip header row
    data = list(reader)

# Upsert the embeddings in batches
batch_size = 100
index_offset = 0
while index_offset < len(data):
    batch = data[index_offset:index_offset + batch_size]

    ## APPEND VECTORS TO INDEX AFTER LAST ENTRIES
    #upsert_embeddings_batch( int(index.describe_index_stats()['total_vector_count'] +1) ,batch, index_offset)

    ## REPLACE VECTORS STATING AT 0
    upsert_embeddings_batch( 0,batch, index_offset)
    print("batch " + str(index_offset))
    index_offset += batch_size

