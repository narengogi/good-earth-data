import pandas as pd
from scipy.sparse import csr_matrix, coo_matrix
import gc
import os
import numpy as np
from scipy.sparse import find
import pickle
import igraph as ig
import leidenalg

def keep_top_k_per_row(matrix,   k: int):
    """Keeps only top-k values per row in a matrix."""
    print(f"keeping top {k} values per row")
        
    # Process each row of the CSR matrix directly
    print(f"processing CSR matrix directly")
    rows, cols, data = [], [], []
    
    for row_idx in range(matrix.shape[0]):
        print("doing row", row_idx)
        # Get the row data
        row_start = matrix.indptr[row_idx]
        row_end = matrix.indptr[row_idx + 1]
        row_data = matrix.data[row_start:row_end]
        row_cols = matrix.indices[row_start:row_end]
        
        if len(row_data) > k:
            # Get indices of top-k values
            top_k_idx = np.argsort(row_data)[-k:]
            # Keep only the top k values
            for idx in top_k_idx:
                rows.append(row_idx)
                cols.append(row_cols[idx])
                data.append(row_data[idx])
        else:
            # Keep all values if there are k or fewer
            for i in range(len(row_data)):
                rows.append(row_idx)
                cols.append(row_cols[i])
                data.append(row_data[i])
    
    return coo_matrix((data, (rows, cols)), shape=matrix.shape)
    # # Create a new sparse matrix with only the top-k values per row
    # matrix = csr_matrix((data, (rows, cols)), shape=matrix.shape)
    # print("converting to coo matrix")

    # return matrix.tocoo()

def add_postions_to_vertices(partition):
    # compute global positions of books using subgraph size and treat it as a packing problem
    # tuple is (subgraph, size, x, y)
    squares = [(subgraph, subgraph.vcount(), None, None) for subgraph in partition.subgraphs()]
    # pack squares into a rectangle with height 1 and width 3
    squares.sort(key=lambda s: s[1], reverse=True)

    packed = []
    current_x = 0
    current_y = 0
    shelf_height = 0
    container_width = 10e7

    for item, size, _, _ in squares:
        if current_x + size > container_width:
            # Move to next shelf
            current_y += shelf_height
            current_x = 0
            shelf_height = 0

        # Place the square
        packed.append((item, size, current_x, current_y))

        # Update position and shelf height
        current_x += size
        shelf_height = max(shelf_height, size)

    for subgraph, _, x, y in packed:
        pos = subgraph.layout_random()
        updated_pos = [(pos[0] + x, pos[1] + y) for pos in pos]
        subgraph.vs['pos'] = updated_pos

    return [item for item, _, _, _ in packed]



def get_cooccurence_matrix():
    book_ids_map = {}
    id_to_book_map = {}
    user_ids_map = {}
    curr_user_id = 0
    curr_book_id = 0

    user_rows = []
    book_cols = []

    # compute cooccurence matrix for all books
    print("loading behaviour data...")
    for file in os.listdir("data/behaviour"):
        print(f"loading {file}...")
        df = pd.read_parquet(f"data/behaviour/{file}", columns=['user_id', 'book_id'])
        print(f"loaded {len(df)} rows from {file}")
        print(f"processing {file}...")
        print(f"grouping by user_id...")
        df = df.groupby('user_id')['book_id'].apply(lambda x: sorted(list(x)))
        print(f"grouped by user_id")
        for user_id, books in df.items():
            if (len(books) > 1000):
                print(f"skipping {user_id} because they have more than 1000 books")
                continue
            if user_id not in user_ids_map:
                user_ids_map[user_id] = curr_user_id
                curr_user_id += 1
            row_index = user_ids_map[user_id]
            for book in books:
                if book not in book_ids_map:
                    book_ids_map[book] = curr_book_id
                    id_to_book_map[curr_book_id] = book
                    curr_book_id += 1
                col_index = book_ids_map[book]
                user_rows.append(row_index)
                book_cols.append(col_index)
        print(f"processed {len(user_rows)} rows and {len(book_cols)} columns")
        # only process one file for testing
        break
    data = np.ones(len(user_rows),dtype=np.uint8)
    print("creating a sparse matrix...")
    A = csr_matrix((data, (user_rows, book_cols)), shape=(curr_user_id, curr_book_id))
    print("sparse matrix created")
    print("computing cooccurence matrix...")
    cooccurence = A.T @ A
    print(f"cooccurence matrix computed")
    return (cooccurence, id_to_book_map)


def create_graph(cooccurence, id_to_book_map):
    counts = cooccurence.diagonal()

    #compute similarity matrix
    i, j, inter = find(cooccurence)
    diagonal = cooccurence.diagonal()
    jaccard_similarities = inter / (diagonal[i] + diagonal[j] - inter) 
    A = csr_matrix((jaccard_similarities, (i, j)), shape=(len(id_to_book_map), len(id_to_book_map)))
    #set self similarity to 0
    A.setdiag(0)
    
    # Print the number of non-zero entries before filtering
    print(f"Number of non-zero entries in original similarity matrix: {len(jaccard_similarities)}")
    
    # create edge list of top 15 most similar books for each book
    jaccard_similarities = keep_top_k_per_row(A, 5)
    del A
    gc.collect()
    # replace inf with 0
    non_zero_rows = jaccard_similarities.row
    non_zero_cols = jaccard_similarities.col
    non_zero_values = jaccard_similarities.data
    print(f"Number of non-zero entries in filtered similarity matrix: {len(non_zero_values)}")
    
    # save graph
    graph = ig.Graph()
    graph.add_vertices(len(id_to_book_map), attributes={'count': counts, 'book_id': list(id_to_book_map.values())})
    graph.add_edges(zip(non_zero_rows, non_zero_cols))
    pickle.dump(graph, open(f"data/graph.pkl", "wb"))
    pickle.dump(non_zero_values, open(f"data/weights.pkl", "wb"))

if __name__ == "__main__":
    cooccurence, id_to_book_map = get_cooccurence_matrix()
    create_graph(cooccurence, id_to_book_map)