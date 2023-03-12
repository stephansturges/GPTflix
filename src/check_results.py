import os

import json
from dotenv import load_dotenv
import pinecone


def check_p3_results_():
    """Counts how many embeddings are in the .jsonl file."""
    with open("data_sample/d3.embeddings_maker_results.jsonl", encoding="utf8") as f:
        lines = f.readlines()
        total_lines = len(lines)
        print(f"Total Embeddings: {total_lines}")


def check_p5_results_query_pinecone(
    ids: list,
    index_name: str,
    namespace: str = "movies",
):
    """This function will return a specific vector id back from the index."""
    dotenv_path = os.path.join(os.path.dirname(__file__), ".env")
    load_dotenv(dotenv_path)

    pinecone.init(
        api_key=os.environ.get("PINECONE_API_KEY"), environment="us-east1-gcp"
    )
    index = pinecone.Index(index_name)

    fetch_response = index.fetch(ids=ids, namespace=namespace)
    for i, id in enumerate(ids):
        print(
            f'Vector Id: {ids[i]}\n{fetch_response["vectors"][ids[i]]["metadata"]["text"]}\n\n'
        )


if __name__ == "__main__":
    check_p5_results_query_pinecone(ids=["99"], index_name="YOUR-PINECONE-INDEX-NAME")
