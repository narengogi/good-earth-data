import pandas as pd
import os
from concurrent.futures import ThreadPoolExecutor
import gc

base_dir = "/home/naren/Code/map-of-goodreads/data-crunching-final/data"

def process_batch(book_to_users, df, book_id_1):
    if book_id_1 not in book_to_users:
        return {}
    co_counts = df[df['user_id'].isin(book_to_users.get(book_id_1))].groupby('book_id')['user_id'].count()
    similarity_scores = {book_id_2: co_counts[book_id_2] / (len(book_to_users.get(book_id_1)) + len(book_to_users.get(book_id_2))) for book_id_2 in co_counts.index}
    top_10_similarity_scores = sorted(similarity_scores.items(), key=lambda x: x[1], reverse=True)[1:11]
    return top_10_similarity_scores

def create_edges_csv():
    print("Reading book counts")
    books = pd.read_csv(f"{base_dir}/books/merged_final.csv", usecols=['book_id']).sort_values(by='book_id')
    num_books = len(books)
    print(f"Loading behaviour data")
    df = pd.read_csv(f"{base_dir}/behaviour/train-00000-of-00009.csv", usecols=['book_id', 'user_id'])
    print(f"Number of rows in behaviour data: {len(df)}")
    print("grouping by book_id")
    book_to_users = df.groupby('book_id')['user_id'].apply(set).to_dict()
    print("grouping by book_id done")
    with open(f"{base_dir}/edges.csv", 'w') as f:
        f.write("source,target,similarity_score\n")
        with ThreadPoolExecutor(max_workers=64) as executor:
            futures = []
            for i, book_id_1 in enumerate(books['book_id']):
                if i == 300:
                    break
                print(f"Processing book {i} of {num_books}")
                futures.append(executor.submit(process_batch, book_to_users, df, book_id_1))
            # write results async as they are done
            for future in futures:
                top_10_similarity_scores = future.result()
                for book_id_2, similarity_score in top_10_similarity_scores:
                    f.write(f"{book_id_1},{book_id_2},{similarity_score}\n")


if __name__ == "__main__":
    create_edges_csv()