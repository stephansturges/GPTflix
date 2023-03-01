import openai
import tiktoken
import numpy as np  
import os
import streamlit as st
import json
from streamlit_chat import message
import pinecone
import random

from PIL import Image

pinecone_api_key = st.secrets["API_KEYS"]["pinecone"]

pinecone.init(api_key=pinecone_api_key, environment="us-east1-gcp")

openai.api_key = st.secrets["API_KEYS"]["openai"]


#gptflix_logo = Image.open('./chat/logo.png')

bens_bites_logo = Image.open('./chat/Bens_Bites_Logo.jpg')

# random user picture
user_av = random.randint(0, 100)

# random bott picture
bott_av = random.randint(0, 100)

def randomize_array(arr):
    sampled_arr = []
    while arr:
        elem = random.choice(arr)
        sampled_arr.append(elem)
        arr.remove(elem)
    return sampled_arr

st.set_page_config(page_title="GPTflix", page_icon="🍿", layout="wide")

# st.header("GPTflix is like chatGPT for movie reviews!🍿\n")


st.header("Thanks for visiting GPTflix! It's been a fun experiment, with over 2500 unique users over two weeks the site was online. Perhaps we will be back some time...🍿\n")

# Define the name of the index and the dimensionality of the embeddings
index_name = "400kmovies"
dimension = 1536

pineconeindex = pinecone.Index(index_name)


######################################
#######
#######   OPEN AI SETTINGS !!!
#######
#######
######################################


COMPLETIONS_MODEL = "text-davinci-003"
#COMPLETIONS_MODEL = "text-curie-001"
EMBEDDING_MODEL = "text-embedding-ada-002"

COMPLETIONS_API_PARAMS = {
    # We use temperature of 0.0 because it gives the most predictable, factual answer.
    "temperature": 0.0,  
    "max_tokens": 400,
    "model": COMPLETIONS_MODEL,
}


feedback_url = "https://forms.gle/YMTtGK1zXdCRzRaj6"
bb_url ="https://www.bensbites.co/?utm_source=gptflix"
tech_url = "https://news.ycombinator.com/item?id=34802625"
github_url = "https://github.com/stephansturges/GPTflix"

with st.sidebar:
    st.markdown("# About 🙌")
    st.markdown(
        "GPTflix allows you to talk to version of chatGPT \n"
        "that has access to reviews of about 10 000 movies! 🎬 \n"
        "It's a little stupid at the moment because it knows 30k movies but only has reviews for 10k 😝\n"
        )
    st.markdown(
        "Unline chatGPT, GPTflix can't make stuff up\n"
        "and will only answer from injected knowlege 👩‍🏫 \n"
    )
    st.markdown("---")
    st.markdown("A side project by Stephan Sturges")
    st.markdown("Kept online by [Ben's Bites](%s)!" %bb_url)
    st.image(bens_bites_logo, width=60)

    st.markdown("---")
    st.markdown("Tech [info](%s) for you nerds out there!" %tech_url)
    st.markdown("Give feedback [here](%s)" %feedback_url)
    st.markdown("---")
    st.markdown("Code open-sourced [here](%s)" %github_url)
    st.markdown("---")


# MAIN FUNCTIONS




def num_tokens_from_string(string, encoding_name):
    """Returns the number of tokens in a text string."""
    encoding = tiktoken.get_encoding(encoding_name)
    num_tokens = len(encoding.encode(string))
    return num_tokens



def get_embedding(text, model):
    result = openai.Embedding.create(
      model=model,
      input=text
    )
    return result["data"][0]["embedding"]



MAX_SECTION_LEN = 2000 #in tokens
SEPARATOR = "\n"
ENCODING = "cl100k_base"  # encoding for text-embedding-ada-002

encoding = tiktoken.get_encoding(ENCODING)
separator_len = len(encoding.encode(SEPARATOR))



def construct_prompt_pinecone(question):
    """
    Fetch relevant information from pinecone DB
    """
    xq = get_embedding(question , EMBEDDING_MODEL)

    #print(xq)

    res = pineconeindex.query([xq], top_k=30, include_metadata=True, namespace="movies")

    #print(res)
    # print(most_relevant_document_sections[:2])

    chosen_sections = []    
    chosen_sections_length = 0

    for match in res['matches'][:12]:
        #print(f"{match['score']:.2f}: {match['metadata']['text']}")
        if chosen_sections_length <= MAX_SECTION_LEN:
            document_section = match['metadata']['text']

            #   document_section = str(_[0] + _[1])      
            chosen_sections.append(SEPARATOR + document_section)

            chosen_sections_length += num_tokens_from_string(str(document_section), "gpt2")

    for match in randomize_array(res['matches'][-18:]):
        #print(f"{match['score']:.2f}: {match['metadata']['text']}")
        if chosen_sections_length <= MAX_SECTION_LEN:
            document_section = match['metadata']['text']

            #   document_section = str(_[0] + _[1])      
            chosen_sections.append(SEPARATOR + document_section)

            chosen_sections_length += num_tokens_from_string(str(document_section), "gpt2")


    # Useful diagnostic information
    #print(f"Selected {len(chosen_sections)} document sections:")
    
    header = """Answer the question as truthfully as possible using the provided context, 
    and if the answer is not contained within the text below, say "I don't know."
    Answer in a very sarcastic tone and make it fun! Surpise the user with your answers!\n
    Context:\n
    You are GPTflix, a AI movie-buff that loves talking about movies!\n
    Movie references:\n
    """ 
    return header + "".join(chosen_sections) 



# TO BE ADDED: memory with summary of past discussions

# def summarize_past_conversation(content):

#     APPEND_COMPLETION_PARAMS = {
#         "temperature": 0.0,
#         "max_tokens": 300,
#         "model": COMPLETIONS_MODEL,
#     }

#     prompt = "Summarize this discussion into a single paragraph keeping the titles of any movies mentioned: \n" + content

#     try:
#         response = openai.Completion.create(
#                     prompt=prompt,
#                     **APPEND_COMPLETION_PARAMS
#                 )
#     except Exception as e:
#         print("I'm afraid your question failed! This is the error: ")
#         print(e)
#         return None

#     choices = response.get("choices", [])
#     if len(choices) > 0:
#         return choices[0]["text"].strip(" \n")
#     else:
#         return None





COMPLETIONS_API_PARAMS = {
        "temperature": 0.0,
        "max_tokens": 500,
        "model": COMPLETIONS_MODEL,
    }


def answer_query_with_context_pinecone(query):
    prompt = construct_prompt_pinecone(query) + "\n\n Q: " + query + "\n A:"
    
    print("---------------------------------------------")
    print("prompt:")
    print(prompt)
    print("---------------------------------------------")
    try:
        response = openai.Completion.create(
                    prompt=prompt,
                    **COMPLETIONS_API_PARAMS
                )
    except Exception as e:
        print("I'm afraid your question failed! This is the error: ")
        print(e)
        return None

    choices = response.get("choices", [])
    if len(choices) > 0:
        return choices[0]["text"].strip(" \n")
    else:
        return None



# Storing the chat
if 'generated' not in st.session_state:
    st.session_state['generated'] = []

if 'past' not in st.session_state:
    st.session_state['past'] = []

# def clear_text():
#     st.session_state["input"] = ""

# We will get the user's input by calling the get_text function
def get_text():
    input_text = st.text_input("Input a question here! For example: \"Is X movie good?\". \n It works best if your question contains the title of a movie! You might want to be really specific, like talking about Pixar's Brave rather than just Brave. Also, I have no memory of previous questions!😅😊","Who are you?", key="input")
    return input_text



# user_input = get_text()


# if user_input:
#     output = answer_query_with_context_pinecone(user_input)

#     # store the output 
#     st.session_state.past.append(user_input)
#     st.session_state.generated.append(output)


# if st.session_state['generated']:
#     for i in range(len(st.session_state['generated'])-1, -1, -1):
#         message(st.session_state["generated"][i],seed=bott_av , key=str(i))
#         message(st.session_state['past'][i], is_user=True,avatar_style="adventurer",seed=user_av, key=str(i) + '_user')



