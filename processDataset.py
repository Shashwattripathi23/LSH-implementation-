import numpy as np
from scipy.sparse import csr_matrix, save_npz

data = np.load("user_movie_rating.npy")


user_ids  = data[:, 0]  # 1-indexed
movie_ids = data[:, 1]  # 1-indexed

# getting dimensions of the sparse matrix
n_users  = int(user_ids.max())
n_movies = int(movie_ids.max())

print(f"Users : {n_users}")
print(f"Movies: {n_movies}")
print(f"Records: {len(user_ids)}")

# build binary sparse matrix 

values = np.ones(len(user_ids), dtype=np.uint8)  # binary: 1 for every (movi, user) pair

movie_user_matrix = csr_matrix(
    (values, (movie_ids - 1, user_ids - 1)),
    shape=(n_movies, n_users)
)

#preventing duplicates
movie_user_matrix = movie_user_matrix.astype(bool).astype(np.uint8) 

# print(movie_user_matrix[0:25, 0:25].toarray())  # print first 5 rows and columns

save_npz("movie_user_matrix.npz", movie_user_matrix)

