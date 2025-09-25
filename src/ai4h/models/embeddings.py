from pathlib import Path

from numpy.typing import NDArray
import numpy as np


def load_txt_embeddings(path: Path | str, max_vocab: int | None = None):
    vocab: list[str] = []
    vectors = []

    with open(path, "r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            parts = line.rstrip().split()
            word, vec = parts[:-50], parts[-50:]
            vocab.append("".join(word))
            vectors.append([float(x) for x in vec])
            if max_vocab and i + 1 >= max_vocab:
                break

    vectors = np.array(vectors, dtype=np.float32)
    return vocab, vectors

class WordEmbeddings:
    def __init__(self, unit_vectors: NDArray[np.float32], vocab: list[str]):
        self.unit_vectors = unit_vectors
        self.vocab = vocab
        self.word2idx = {word: i for i, word in enumerate(vocab)}

    def vec(self, word: str) -> NDArray[np.float32] | None:
        idx = self.word2idx.get(word)
        if idx is not None:
            return self.unit_vectors[idx]
        return None
    
    def most_similar(
        self,
        query_vec: NDArray[np.float32],
        topk: int = 10,
        exclude: tuple[str, ...] | None = None,
    ) -> list[tuple[str, float]]:
        
        # Normalize the query vector to unit length 
        query_vec_norm = query_vec / np.linalg.norm(query_vec)

        # Compute cosine similarities (dot products with unit vectors)
        score = np.dot(self.unit_vectors, query_vec_norm)

        # Set all scores of excluded words to -inf
        if exclude:
            for word in exclude:
                idx = self.word2idx.get(word)
                if idx is not None:
                    score[idx] = -np.inf

        # Get the indices of the top-k most similar words
        topk_indices = np.argpartition(-score, topk)[:topk]

        # Return a list of tuples with (word, score) for the top-k words
        return [(self.vocab[i], float(score[i])) for i in topk_indices]
