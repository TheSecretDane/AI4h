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