import sys
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path


def load_matrix(file_path: Path) -> np.ndarray:
    matrix = np.loadtxt(file_path)
    if matrix.shape[0] != matrix.shape[1]:
        raise ValueError(f"Matrix is not square: shape={matrix.shape}")
    return matrix


def create_heatmap(matrix: np.ndarray, output_path: str = "heatmap.png", title: str = "Intent Similarity"):
    n = matrix.shape[0]

    # Figure size scaled for a 60x60 matrix
    fig, ax = plt.subplots(figsize=(16, 14))

    im = ax.imshow(matrix, cmap="viridis", vmin=matrix.min(), vmax=matrix.max())

    # Colorbar
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("Similarity", rotation=270, labelpad=20, fontsize=12)

    # Axis labels: since no names are available, use numeric indices (0-59)
    labels = [str(i) for i in range(n)]
    ax.set_xticks(np.arange(n))
    ax.set_yticks(np.arange(n))
    ax.set_xticklabels(labels, fontsize=6, rotation=90)
    ax.set_yticklabels(labels, fontsize=6)

    ax.set_title(title, fontsize=16, pad=20)
    ax.set_xlabel("Intent", fontsize=12)
    ax.set_ylabel("Intent", fontsize=12)

    plt.tight_layout()
    plt.savefig(output_path, dpi=200, bbox_inches="tight")
    print(f"Heatmap saved to: {output_path}")
    plt.close()


if __name__ == "__main__":

    base_dir = Path(__file__).resolve().parent

    input_path = base_dir / "sentence_transformer_similarity" / "frontend-backend-app_sentence_transformer_matrix.txt"
    output_path = base_dir / "sentence_transformer_similarity" / "frontend-backend-app_sentence_transformer_heatmap.png"

    matrix = load_matrix(input_path)
    create_heatmap(matrix, output_path=output_path, title="Sentence Transformer Similarity Heatmap")