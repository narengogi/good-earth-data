import pycuda.autoinit
import pycuda.driver as cuda
from pycuda.compiler import SourceModule
import numpy as np
import pandas as pd
import os

base_dir = "/home/naren/Code/map-of-goodreads/data-crunching-final/data"

kernel_code = """
__global__ void compute_similarity(int *book_user, float *output, int num_books, int num_users) {
    int i = blockIdx.x;
    int j = threadIdx.x;
    if (i >= num_books || j >= num_books || i == j) return;

    int intersection = 0;
    int total = 0;
    for (int u = 0; u < num_users; ++u) {
        int a = book_user[i * num_users + u];
        int b = book_user[j * num_users + u];
        intersection += (a & b);
        total += a + b;
    }

    if (intersection > 0 && total > 0) {
        output[i * num_books + j] = ((float)intersection) / total;
    }
}
"""

def create_edges_cuda():
    # Step 1: Read data
    books_df = pd.read_csv(f"{base_dir}/books/merged_final.csv", usecols=['book_id']).sort_values(by='book_id')
    book_ids = books_df['book_id'].tolist()
    num_books = len(book_ids)

    for file in os.listdir(f"{base_dir}/behaviour"):
        print(f"Processing {file}")
        df = pd.read_csv(f"{base_dir}/behaviour/{file}", usecols=['book_id', 'user_id'])
        df['presence'] = 1
        pivot = df.pivot_table(index='book_id', columns='user_id', values='presence', fill_value=0)

        book_index_map = {bid: i for i, bid in enumerate(book_ids)}
        pivot = pivot.reindex(book_ids).fillna(0).astype(np.int32)
        book_user_matrix = pivot.to_numpy()
        num_users = book_user_matrix.shape[1]

        # Step 2: Allocate device memory
        flat_book_user = book_user_matrix.flatten()
        d_book_user = cuda.mem_alloc(flat_book_user.nbytes)
        cuda.memcpy_htod(d_book_user, flat_book_user)

        output_matrix = np.zeros((num_books, num_books), dtype=np.float32)
        d_output = cuda.mem_alloc(output_matrix.nbytes)

        # Step 3: Compile and launch kernel
        mod = SourceModule(kernel_code)
        func = mod.get_function("compute_similarity")
        func(d_book_user, d_output, np.int32(num_books), np.int32(num_users),
             block=(num_books, 1, 1), grid=(num_books, 1, 1))

        cuda.memcpy_dtoh(output_matrix, d_output)

        # Step 4: Save top 30 similar books
        with open(f"{base_dir}/edges/edges_{file}.csv", "w") as f:
            f.write("source,target,similarity_score\n")
            for i, book_id_1 in enumerate(book_ids):
                sim_scores = [(book_ids[j], output_matrix[i][j]) for j in range(num_books) if i != j]
                top_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)[:30]
                for book_id_2, score in top_scores:
                    if score > 0:
                        f.write(f"{book_id_1},{book_id_2},{score}\n")

if __name__ == "__main__":
    create_edges_cuda()