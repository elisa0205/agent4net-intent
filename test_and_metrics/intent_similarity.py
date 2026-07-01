import numpy as np
import csv
import argparse
import re
from pathlib import Path
from collections import defaultdict
from sentence_transformers import SentenceTransformer


CSV_PATH = Path("expanded-dataset(in).csv")
RESULTS_DIR = Path("test_and_metrics/results")


#Simple tokenization to compute similarity 
def tokenize(text):
    tokens = re.findall(r"[a-zA-Z0-9']+", text.lower())
    return set(tokens)

#Jaccard similarity between two sets of tokens
def jaccard(a, b):
    return len(a & b) / len(a | b)

#Load intents from the CSV file and group them by example name
def load_intents(csv_path: Path) -> dict[str, list[str]]:
    grouped = defaultdict(list)

    with csv_path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            example_name = row["example"].strip()
            intent_text = row["text"].strip()
            grouped[example_name].append(intent_text)
    return dict(grouped)

#Build similarity matrices for each example
def build_similarity_matrices(intents_set: dict[str, list[str]]) -> dict[str, np.ndarray]:
    matrices = {}

    for example, intents in intents_set.items():
        tokenized_intents = [tokenize(intent) for intent in intents]
        n = len(tokenized_intents)
        matrix = np.zeros((n, n), dtype=float)

        for i in range(n):
            for j in range(n):
                matrix[i, j] = jaccard(tokenized_intents[i], tokenized_intents[j])

        matrices[example] = matrix

    return matrices

#Save the similarity matrices to text files for later analysis
def save_similarity_matrices(similarity_matrices: dict[str, np.ndarray], method: str) -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    for example, matrix in similarity_matrices.items():
        output_path = RESULTS_DIR / f"{method}_similarity" / f"{example}_{method}_matrix.txt"
        np.savetxt(output_path, matrix, fmt="%.4f", delimiter="\t")

#Build similarity matrices using SentenceTransformer embeddings
def build_sentence_transformer_similarity_matrices(intents_set: dict[str, list[str]], model_name: str) -> dict[str, np.ndarray]:
    model = SentenceTransformer(model_name)
    matrices = {}

    for example, intents in intents_set.items():
        embeddings = model.encode(intents)
        # By default, the similarity method in SentenceTransformer computes cosine similarity
        similarity_matrix = model.similarity(embeddings, embeddings)
        matrices[example] = similarity_matrix.cpu().numpy()

    return matrices
       


if __name__ == "__main__":

    # py .\test_and_metrics\intent_similarity.py --method st

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--method",
        choices=["jaccard", "st", "all"],
        default="all",
        help="Similarity method to use (jaccard, sentence_transformer (st), or all)"
    )

    args = parser.parse_args()

    intents_by_example = load_intents(CSV_PATH)

    if args.method in ["jaccard", "all"]:
        #Jaccard similarity
        jaccard_similarity_matrices = build_similarity_matrices(intents_by_example)
        save_similarity_matrices(jaccard_similarity_matrices, "jaccard")
        print("Jaccard similarity matrices saved.")

    if args.method in ["st", "all"]:
        #Sentence Transformer similarity
        st_similarity_matrices = build_sentence_transformer_similarity_matrices(intents_by_example, "all-MiniLM-L6-v2")
        save_similarity_matrices(st_similarity_matrices, "sentence_transformer")
        print("Sentence Transformer similarity matrices saved.")
