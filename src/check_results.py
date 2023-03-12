import os

import json
from dotenv import load_dotenv
import pinecone


def check_p3_results():
    """Counts how many embeddings are in the .jsonl file."""

    total_errors = 0
    list_of_indices_with_errors = list()
    with open("data_sample/d3.embeddings_maker_results.jsonl", encoding="utf8") as f:
        lines = f.readlines()
        total_lines = len(lines)
        for i in range(0, total_lines):
            data = json.loads(lines[i])
            if isinstance(data[1], list):
                list_of_indices_with_errors.append(i + 1)
                total_errors += 1
    
    complete_embeddings = total_lines - total_errors
    success_rate = (complete_embeddings / total_lines) * 100
    
    print(
        f"\nIndices with error: {list_of_indices_with_errors}\n"
        f"\nTotal elements with embedding error: {total_errors}"
        f"\nTotal embeddings made from elements: {complete_embeddings}"
        f"\nTotal percentage of elements successfully embedded from corpus: {success_rate:.2f}%"
        f"\nTotal elements processed by OpenAI: {total_lines}\n"
    )


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

    check_p3_results()

    # check_p5_results_query_pinecone(ids=["99"], index_name="YOUR-PINECONE-INDEX-NAME")
