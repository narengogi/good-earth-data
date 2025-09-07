import pandas as pd
from scipy.sparse import csr_matrix, coo_matrix
import gc
import os
import numpy as np
from scipy.sparse import find
import pickle
import igraph as ig
import leidenalg
import psutil

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



def print_memory_usage():
    """Print current memory usage"""
    process = psutil.Process(os.getpid())
    memory_info = process.memory_info()
    print(f"Memory usage: {memory_info.rss / 1024 / 1024:.2f} MB")

def get_cooccurence_matrix():
    book_ids_map = {}
    id_to_book_map = {}
    curr_book_id = 0
    
    # Initialize cooccurrence matrix as empty
    cooccurence = None
    
    print("loading behaviour data...")
    print_memory_usage()
    
    for file in os.listdir("data/behaviour"):
        print(f"loading {file}...")
        df = pd.read_parquet(f"data/behaviour/{file}", columns=['user_id', 'book_id'])
        print(f"loaded {len(df)} rows from {file}")
        print_memory_usage()
        
        # Process in chunks to avoid memory issues
        chunk_size = 5000  # Reduced chunk size for better memory management
        user_groups = df.groupby('user_id')['book_id'].apply(lambda x: sorted(list(x)))
        
        print(f"Total users in file: {len(user_groups)}")
        del df  # Free memory immediately after grouping
        gc.collect()
        
        for i in range(0, len(user_groups), chunk_size):
            chunk = user_groups.iloc[i:i+chunk_size]
            print(f"processing chunk {i//chunk_size + 1} with {len(chunk)} users...")
            
            # Build book mappings for this chunk
            chunk_books = set()
            for books in chunk.values:
                if len(books) <= 1000:  # Skip users with too many books
                    chunk_books.update(books)
            
            # Add new books to global mapping
            for book in chunk_books:
                if book not in book_ids_map:
                    book_ids_map[book] = curr_book_id
                    id_to_book_map[curr_book_id] = book
                    curr_book_id += 1
            
            # Process cooccurrences for this chunk
            chunk_cooccurrence = process_chunk_cooccurrences(chunk, book_ids_map, curr_book_id)
            
            # Add to global cooccurrence matrix
            if cooccurence is None:
                cooccurence = chunk_cooccurrence
            else:
                cooccurence = cooccurence + chunk_cooccurrence
            
            # Clean up memory
            del chunk_cooccurrence
            del chunk_books
            gc.collect()
            
            if (i // chunk_size + 1) % 10 == 0:  # Print memory every 10 chunks
                print_memory_usage()
        
        del user_groups  # Free memory after processing file
        gc.collect()
        print_memory_usage()
        
        # Process all files, not just one
        break
    
    print(f"cooccurence matrix computed")
    print_memory_usage()
    return (cooccurence, id_to_book_map)

def process_chunk_cooccurrences(user_groups, book_ids_map, total_books):
    """Process cooccurrences for a chunk of users"""
    book_pairs = {}
    
    for user_id, books in user_groups.items():
        if len(books) > 1000:
            continue
            
        # Get book indices
        book_indices = [book_ids_map[book] for book in books if book in book_ids_map]
        
        # Generate all pairs of books for this user
        for i in range(len(book_indices)):
            for j in range(i, len(book_indices)):
                pair = (book_indices[i], book_indices[j])
                book_pairs[pair] = book_pairs.get(pair, 0) + 1
    
    # Convert to sparse matrix
    if not book_pairs:
        return csr_matrix((total_books, total_books))
    
    rows, cols, data = zip(*[(i, j, count) for (i, j), count in book_pairs.items()])
    
    # Make matrix symmetric
    sym_rows = list(rows) + list(cols)
    sym_cols = list(cols) + list(rows)
    sym_data = list(data) + list(data)
    
    # Remove duplicates on diagonal
    pairs_dict = {}
    for r, c, d in zip(sym_rows, sym_cols, sym_data):
        if (r, c) in pairs_dict:
            if r == c:  # Diagonal element, don't double count
                pairs_dict[(r, c)] = d // 2
            else:
                pairs_dict[(r, c)] = d
        else:
            pairs_dict[(r, c)] = d
    
    if pairs_dict:
        final_rows, final_cols, final_data = zip(*[(r, c, d) for (r, c), d in pairs_dict.items()])
        return csr_matrix((final_data, (final_rows, final_cols)), shape=(total_books, total_books))
    else:
        return csr_matrix((total_books, total_books))


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