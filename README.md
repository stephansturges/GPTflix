# GPTflix source code for deployment on Streamlit

## What are we going to build?


This is the source code of www.gptflix.ai

We will build a GPTflix QA bot with OpenAI, Pinecone DB and Streamlit. You will learn how to prepare text to send to an embedding model. You will capture the embeddings and text returned from the model for upload to Pinecone DB. Afterwards you will setup a Pinecone DB index and upload the OpenAI embeddings to the DB for the bot to search over the embeddings.

Finally, we will setup a QA bot frontend chat app with Streamlit. When the user asks the bot a question, the bot will search over the movie text in your Pinecone DB. It will answer your question about a movie based on text from the DB.

</br>

## What is the point?

This is meant as a basic scaffolding to build your own knowledge-retrieval systems, it's super basic for now! 

This repo contains the GPTflix source code and a Streamlit deployment guide.

</br>

## Setup prerequisites

This repo is set up for deployment on Streamlit, you will want to set your environment variables in streamlit like this:

1. Fork the [GPTflix](https://github.com/stephansturges/GPTflix/fork) repo to your GitHub account. 

2. Set up an account on [Pinecone.io](https://app.pinecone.io/)

3. Set up an account on [Streamlit cloud](https://share.streamlit.io/signup)

4. Create a new app on Streamlit. Link it to your fork of the repo on Github then point the app to `/chat/main.py` as the main executable.

5. Go to your app settings, and navigate to Secrets. Set up the secret like this:

[//]: # 

    [API_KEYS]
    pinecone = "xxxxxxxx-xxxx-xxxx-xxxx-xxxxxxxx"
    openai = "sk-xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx"

6. Make a `.env` file in the the root of the project with your OpenAI API Key on your local machine.

[//]: # 

    PINECONE_API_KEY=xxxxxxxx-xxxx-xxxx-xxxx-xxxxxxxx
    OPENAI_API_KEY=sk-xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx



Those need to be your pinecone and openai API keys of course ;)

</br>

## How to add data?
This repo is set up to walk through a demo using the MPST data in /data_samples
These are the steps:

1. Run `p1.generate_index_mpst.py` to prepare the text from`./data_sample/d0.mpst_1k_raw.csv` into a format we can inject into a model and get its embedding.

[//]: # 

        python p1.generate_index_mpst.py

2. Run `p2.make_jsonl_for_requests_mpst.py` to convert your new `d1.mpst_1k_converted.csv` file to a jsonl file with instructions to run the embeddings requests against the OpenAI API.

[//]: # 

        python p2.make_jsonl_for_requests_mpst.py

3. Run `p3.api_request_parallel_processor.py` on the JSONL file from (2) to get embeddings.

[//]: # 

    python src/p3.api_request_parallel_processor.py \
      --requests_filepath data_sample/d2.embeddings_maker.jsonl \
      --save_filepath data_sample/d3.embeddings_maker_results.jsonl \
      --request_url https://api.openai.com/v1/embeddings \
      --max_requests_per_minute 1500 \
      --max_tokens_per_minute 6250000 \
      --token_encoding_name cl100k_base \
      --max_attempts 5 \
      --logging_level 20

4. Run `p4.convert_jsonl_with_embeddings_to_csv.py` with the new jsonl file to make a pretty CSV with the text and embeddings. 
~~This is cosmetic and a bit of a waste of time in the process, feel free to clean it up.~~.  -> actually that's not quite true: you don't care about making the CSV because you don't need to care about the index of the embeddings **if you are only going to upload data to the index once**, if you are going to be updating the indexing and adding more data, or need an offline / readable format to keep track of things then making the CSV kinda makes sense :)

[//]: # 

        python p4.convert_jsonl_with_embeddings_to_csv.py

5. Run `p5.upload_to_pinecone.py` with your api key and database settings to upload all that text data and embeddings.

[//]: # 

        python p5.upload_to_pinecone.py

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
