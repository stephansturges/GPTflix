# GPTflix source code for deployment on Streamlit

This is the source code of www.gptflix.ai

</br>

## What is the point?

This is meant as a basic scaffolding to build your own knowledge-retrieval systems, it's super basic for now! 

This repo contains the GPTflix source code and a Streamlit deployment guide.

</br>

## Setup prerequisites

This repo is set up for deployment on Streamlit, you will want to set your environment variables in streamlit like this:

1. Set up an account on [Streamlit cloud](https://share.streamlit.io/signup)
2. Set up an account on [Pinecone.io](https://app.pinecone.io/)
2. Link to github, find your repo, create an app pointing to /chat/main.py as the executable
3. Go to your app settings, and navigate to secrets. Set up the secret like this:



[//]: # ([API_KEYS]
pinecone = "xxxxxxx"
openai = "xxxxx")


Those need to be your pinecone and openai API keys of course ;)

</br>

## How to add data?
This repo is set up to walk through a demo using the MPST data in /data_samples
These are the steps:

1. Run generate_index_mpst.py to convert the .csv file in data_samples into a format with just the text we want to inject.

2. Run make_jsonl_for_requests_mpst.py to use your new csv file to make a jsonl file with instructions to run the embeddings requests against the OpenAI API.

3. Run api_request_parallel_processor.py using the docs inside the file (add the tag to point to your API key) on the JSONL file from (2) to get embeddings

4. Run convert_jsonl_with_embeddings_to_csv.py with the new jsonl file to make a pretty CSV with the text and embeddings. This is cosmetic and a bit of a waste of time in the process, feel free to clean it up.

5. Run upload_to_pinecone.py with your api key and database settings to upload all that text data and embeddings.

You can run the app locally but you'll need to remove the images (the paths are different on streamlit cloud)

</br>

## What is included?

At the moment there is some data in sample_data, all taken from Kaggle as examples. 

</br>

## To do

[] Add memory: summarize previous questions / answers and prepend to prompt </br>
[] Add different modes: wider search in database </br>
[] Add different modes: AI tones / characters for responses </br>
[] Better docs </br>


BETTER DOCS COMING SOON! Feel free to contribute them :)
