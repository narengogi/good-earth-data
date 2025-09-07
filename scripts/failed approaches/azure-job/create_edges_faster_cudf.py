import cudf as pd
import collections
import gc

def create_edges_csv():
    print("Reading book counts")
    books = pd.read_csv(f"./books.csv", usecols=['book_id']).sort_values(by='book_id')
    num_books = len(books)
    print(f"Loading behaviour data")
    df = pd.read_csv(f"./behaviour.csv", usecols=['book_id', 'user_id'])
    print(f"Number of rows in behaviour data: {len(df)}")
    print("grouping by book_id")
    book_to_users = df.groupby('book_id')['user_id'].apply(set).to_dict()
    print("grouping by user_id")
    user_to_books = df.groupby('user_id')['book_id'].apply(set).to_dict()
    del books
    del df
    gc.collect()
    edges = {}
    for i, book_id_1 in enumerate(book_to_users.keys()):
        if i == 300:
            break
        print(f"Processing book {i} of {num_books}")
        co_counts = pd.Series(collections.Counter(book_id_2 for user_id in book_to_users.get(book_id_1, set()) for book_id_2 in user_to_books.get(user_id, set())))
        # co_counts = df[df['user_id'].isin(book_to_users.get(book_id_1))].groupby('book_id')['user_id'].count()
        similarity_scores = {book_id_2: co_counts[book_id_2] / (len(book_to_users.get(book_id_1)) + len(book_to_users.get(book_id_2))) for book_id_2 in co_counts.index}
        top_10_similarity_scores = sorted(similarity_scores.items(), key=lambda x: x[1], reverse=True)[1:11]
        edges[book_id_1] = top_10_similarity_scores
    with open(f"./edges.csv", 'w') as f:
        f.write("source,target,similarity_score\n")
        for book_id_1, top_10_similarity_scores in edges.items():
            for book_id_2, similarity_score in top_10_similarity_scores:
                f.write(f"{book_id_1},{book_id_2},{similarity_score}\n")


if __name__ == "__main__":
    create_edges_csv()