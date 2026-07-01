import heapq
import csv
from collections import defaultdict
from pathlib import Path


def load_matrix(filepath: str) -> list[list[float]]:
    path = Path(filepath)
    if not path.exists():
        raise FileNotFoundError(f"File non trovato: {filepath}")

    lines = path.read_text(encoding="utf-8").strip().splitlines()
    lines = [l.strip() for l in lines if l.strip()]

    def parse_line(line: str) -> list[float]:
        
        parts = line.split('\t')
        parts = [p.strip() for p in parts if p.strip()]
        if len(parts) > 1:
            return [float(p) for p in parts]
        return [float(p.strip()) for p in parts if p.strip()]

    matrix = [parse_line(l) for l in lines]

    # Validazione: deve essere quadrata
    n = len(matrix)
    for i, row in enumerate(matrix):
        if len(row) != n:
            raise ValueError(
                f"Riga {i+1}: attese {n} colonne, trovate {len(row)}. "
                "La matrice deve essere quadrata."
            )
    return matrix

def top_pairs(matrix: list[list[float]], top_n: int) -> list[tuple[float, int, int]]:
    n = len(matrix)
    heap = []
    for i in range(n):
        for j in range(i + 1, n):
            score = matrix[i][j]
            if len(heap) < top_n:
                heapq.heappush(heap, (score, i, j))
            elif score > heap[0][0]:
                heapq.heapreplace(heap, (score, i, j))

    # descending order
    results = sorted(heap, reverse=True)
    return results

def minor_pairs(matrix: list[list[float]], minor_n: int) -> list[tuple[float, int, int]]:
    n = len(matrix)
    heap = []
    for i in range(n):
        for j in range(i + 1, n):
            score = matrix[i][j]
            neg = -score
            if len(heap) < minor_n:
                heapq.heappush(heap, (neg, i, j))
            elif neg > heap[0][0]:
                heapq.heapreplace(heap, (neg, i, j))

    return sorted((-neg, i, j) for neg, i, j in heap)


def main():
    BASE = Path(__file__).parent

    for file in BASE.rglob("*.txt"):

        matrix = load_matrix(file)
        # More similar and less similar pairs
        top = top_pairs(matrix, 5)
        minor = minor_pairs(matrix, 5)

        out_file = file.with_suffix(".stats")
        with open(out_file, "w", encoding="utf-8") as f:
            for rank, (score, i, j) in enumerate(top, start=1):
                f.write(f"{rank}. Intent {i+1} — Intent {j+1}: {score:.4f}\n")
            f.write("\n")
            for rank, (score, i, j) in enumerate(minor, start=1):
                f.write(f"{rank}. Intent {i+1} — Intent {j+1}: {score:.4f}\n")
    

if __name__ == "__main__":
    main()
    
