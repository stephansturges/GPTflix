# Python Standard Library
import warnings

# Third Party Libraries
import pandas as pd
import tiktoken


def num_tokens_from_string(string, encoding_name: str = "cl100k_base"):
    """Returns the number of tokens in a text string."""
    encoding = tiktoken.get_encoding(encoding_name)
    num_tokens = len(encoding.encode(string))
    return num_tokens

def combine_text_to_one_column(df):
    df["gpttext"] = "Title: " + df["title"].astype(str) + \
                    " tags: " + df["tags"].astype(str) + \
                    " Plot / story / about: " + df["plot_synopsis"].astype(str)

    df = df.drop(df.columns[[0, 1, 2, 3, 4, 5]], axis=1)   

    df.to_csv("data_sample/mpst_5k_converted.csv")


if __name__ == "__main__":
    # This is a sample converter that takes CSV data from a table
    # (in this case the Kaggle dataset here https://www.kaggle.com/datasets/cryptexcode/mpst-movie-plot-synopses-with-tags )
    # and converts this data into a CSV that contains a single column 
    # with a block of text that we want to make accessible on our Pinecone database

    warnings.filterwarnings("ignore", category=DeprecationWarning) 
    warnings.filterwarnings("ignore", category=FutureWarning) 

    # read sample data
    df = pd.read_csv(filepath_or_buffer='data_sample/mpst_5k.csv', sep="," , header=0, dtype="string", encoding="utf-8" )

    combine_text_to_one_column(df=df)

