import pandas as pd
from concurrent.futures import ThreadPoolExecutor
import gc

def process_batch(book_to_users, books, book_id_1, start_index, end_index):
    similarity_scores = {}
    for book_id_2 in books['book_id'][start_index:end_index]:
        # find number of users that have both books in their behaviour
        book_1_users = book_to_users.get(book_id_1, set())
        book_2_users = book_to_users.get(book_id_2, set())
        num_users = len(book_1_users & book_2_users)
        if num_users > 0:
            similarity_score = num_users / (len(book_1_users) + len(book_2_users))
            similarity_scores[book_id_2] = similarity_score
    return similarity_scores

def create_edges_csv():
    print("Reading book counts")
    books = pd.read_csv(f"./books.csv", usecols=['book_id']).sort_values(by='book_id')
    num_books = len(books)
    print(f"Loading behaviour data")
    df = pd.read_csv(f"./behaviour.csv", usecols=['book_id', 'user_id'])
    print(f"Number of rows in behaviour data: {len(df)}")
    print("grouping by book_id")
    book_to_users = df.groupby('book_id')['user_id'].apply(set).to_dict()
    print("grouping by book_id done")
    del df
    gc.collect()
    with open(f"./edges.csv", 'w') as f:
        f.write("source,target,similarity_score\n")
        edges = {}
        for i, book_id_1 in enumerate(books['book_id']):
            edges[book_id_1] = {}
            print(f"Processing book {i} of {num_books}")
            with ThreadPoolExecutor(max_workers=1000) as executor:
                futures = []
                batch_size = 10000
                for j in range(i+1, num_books, batch_size):
                    futures.append(executor.submit(process_batch, book_to_users, books, book_id_1, j, j + batch_size))
                results = {}
                for future in futures:
                    results.update(future.result())
                if book_id_1 in edges:
                    results.update(edges[book_id_1])

                if book_id_1 in results:
                    results.pop(book_id_1)
                top_10_similarity_scores = sorted(results.items(), key=lambda x: x[1], reverse=True)[:10]
                for book_id_2, similarity_score in top_10_similarity_scores:
                    f.write(f"{book_id_1},{book_id_2},{similarity_score}\n")
                for key, value in results.items():
                    if key not in edges:
                        edges[key] = {book_id_1: value}
                    else:
                        edges[key][book_id_1] = value
                edges.pop(book_id_1)


if __name__ == "__main__":
    create_edges_csv()