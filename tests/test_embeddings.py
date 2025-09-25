"""
Some (probably too) simple tests for the exercise class.
"""

import numpy as np

from ai4h.models import embeddings

# Load data
vocab, vectors = embeddings.load_txt_embeddings(
    "src/data/wiki_giga_2024_50_MFT20_vectors_seed_123_alpha_0.75_eta_0.075_combined.txt",
    max_vocab=100_000,
)

norms = np.linalg.norm(vectors, axis=1, keepdims=True)
unit_vectors = vectors / np.clip(norms, a_min=1e-12, a_max=None)
we = embeddings.WordEmbeddings(unit_vectors, vocab)


def test_vec():
    assert np.allclose(
        we.vec("king").round(4),
        np.array(
            [
                [-0.0643, 0.1432, 0.1129, -0.125, 0.2542],
                [-0.3309, -0.0178, 0.2079, 0.0429, -0.0968],
                [0.1925, 0.0357, 0.1176, -0.0849, -0.0877],
                [-0.0413, 0.0449, 0.0693, 0.0696, 0.0448],
                [0.0468, 0.0343, -0.1667, 0.0297, 0.0037],
                [0.0315, 0.0051, -0.011, -0.1717, -0.0342],
                [-0.0995, -0.0312, 0.1098, 0.0201, -0.0616],
                [0.0897, -0.0427, 0.656, 0.0866, 0.1529],
                [-0.1004, -0.0913, -0.049, 0.0448, 0.0679],
                [-0.0091, -0.0921, -0.0107, 0.1925, 0.0218],
            ],
            dtype=np.float32,
        ).ravel(),
    )


def test_most_similar():
    vec = we.vec
    most_similar = we.most_similar

    (c1, c2, c3, _) = ("king", "man", "woman", "queen")
    q = vec(c1) - vec(c2) + vec(c3)
    output = most_similar(q, topk=10, exclude=(c1, c2, c3))

    words = [o[0] for o in output]
    scores = np.array([o[1] for o in output])

    assert words == [
        "queen",
        "daughter",
        "throne",
        "eldest",
        "elizabeth",
        "princess",
        "marriage",
        "mother",
        "granddaughter",
        "father",
    ]
    assert np.allclose(
        scores.round(4),
        np.array(
            [
                0.8659,
                0.7968,
                0.7833,
                0.7757,
                0.7756,
                0.7643,
                0.7618,
                0.758,
                0.7551,
                0.7543,
            ]
        ),
    )
