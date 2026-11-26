
import time
import numpy as np
from scipy import sparse

THRESHOLD = 0.5


def get_user_movies(matrix, user_id):
    """
    Retrieve movie indices rated by a user.

    Candidate user IDs are assumed to be original 1-based IDs.
    The matrix uses 0-based column indices.
    """
    user_index = int(user_id) - 1

    if user_index < 0 or user_index >= matrix.shape[1]:
        raise ValueError(f"Invalid user ID: {user_id}")

    start = matrix.indptr[user_index]
    end = matrix.indptr[user_index + 1]

    return matrix.indices[start:end]


def jaccard_similarity(movies_a, movies_b):
    """
    Calculate exact Jaccard similarity.

    Movie lists are sorted, so intersection is calculated
    using NumPy.
    """
    intersection = np.intersect1d(movies_a, movies_b).size
    union = len(movies_a) + len(movies_b) - intersection

    if union == 0:
        return 0.0

    return intersection / union


def verify_candidates(candidate_pairs, user_movies, threshold=THRESHOLD):
    """
    Verify candidate pairs and keep only pairs whose
    Jaccard similarity is strictly greater than threshold.
    """
    similar_pairs = []
    checked = 0

    # Remove duplicate candidate pairs and normalize their order
    unique_pairs = set()

    for user_a, user_b in candidate_pairs:
        user_a = int(user_a)
        user_b = int(user_b)

        if user_a == user_b:
            continue

        pair = tuple(sorted((user_a, user_b)))
        unique_pairs.add(pair)

    for user_a, user_b in unique_pairs:
        movies_a = user_movies[user_a]
        movies_b = user_movies[user_b]

        similarity = jaccard_similarity(movies_a, movies_b)
        checked += 1

        if similarity > threshold:
            similar_pairs.append((user_a, user_b, similarity))

    return similar_pairs, checked


def save_results(similar_pairs, output_file="result.txt"):
    """
    Save final user pairs in the required format:
    user1,user2
    """
    with open(output_file, "w") as f:
        for user_a, user_b, similarity in sorted(similar_pairs):
            f.write(f"{user_a},{user_b}\n")


def main():
    start_time = time.time()

    print("Loading movie-user matrix...")
    matrix = sparse.load_npz("movie_user_matrix.npz").tocsc()

    print("Loading candidate pairs...")
    candidate_pairs = np.load("candidates.npy")

    print("Matrix shape:", matrix.shape)
    print("Candidate pairs:", len(candidate_pairs))

    print("Extracting user movie lists...")
    user_movies = {}

    # Extract movies only for users appearing in candidates
    candidate_users = np.unique(candidate_pairs)

    for user_id in candidate_users:
        user_movies[int(user_id)] = get_user_movies(matrix, int(user_id))

    print("Users extracted:", len(user_movies))

    print("Verifying candidate pairs...")
    similar_pairs, checked = verify_candidates(
        candidate_pairs,
        user_movies,
        threshold=THRESHOLD
    )

    print("\n--- Results ---")
    print("Candidate pairs:", len(candidate_pairs))
    print("Unique pairs checked:", checked)
    print("Pairs above 0.5:", len(similar_pairs))

    save_results(similar_pairs, "result.txt")

    elapsed = time.time() - start_time

    print("Saved to result.txt")
    print(f"Total verification runtime: {elapsed:.2f} seconds")


if __name__ == "__main__":
    main()
