import pandas as pd

def create_edges_csv():
    print("Reading book counts")
    books = pd.read_csv("./data/books/merged_final.csv", usecols=['book_id']).sort_values(by='book_id')
    print("Loading behaviour data")
    df = pd.read_csv("./data/behaviour/train-00000-of-00009.csv", usecols=['book_id', 'user_id'], nrows=10000)
    print(f"Number of rows in behaviour data: {len(df)}")

    # Create user -> books mapping
    user_to_books = df.groupby("user_id").agg({"book_id": "collect"}).reset_index()
    user_to_books = user_to_books.set_index("user_id")
    
    # Create book -> users mapping
    book_to_users = df.groupby("book_id").agg({"user_id": "collect"}).reset_index()
    book_to_users = book_to_users.set_index("book_id")

    edges = []

    for i, book_id_1 in enumerate(book_to_users.index[:300]):
        if i % 10 == 0:
            print(f"Processing book {i} of {len(book_to_users)}")
        
        users_1 = set(book_to_users.loc[book_id_1].user_id.to_pandas())
        books_seen = []

        for user in users_1:
            try:
                books_seen.extend(user_to_books.loc[user].book_id.to_pandas())
            except KeyError:
                continue

        co_counts = pd.Series(books_seen).value_counts()

        similarity_scores = {}
        len_u1 = len(users_1)
        for book_id_2 in co_counts.index.to_pandas():
            try:
                users_2 = set(book_to_users.loc[book_id_2].user_id.to_pandas())
                len_u2 = len(users_2)
                score = co_counts[book_id_2] / (len_u1 + len_u2)
                similarity_scores[book_id_2] = score
            except KeyError:
                continue

        # Get top 10, excluding self
        top_10 = sorted(similarity_scores.items(), key=lambda x: x[1], reverse=True)
        top_10 = [(b2, score) for b2, score in top_10 if b2 != book_id_1][:10]

        for book_id_2, sim_score in top_10:
            edges.append((book_id_1, book_id_2, sim_score))

    # Save to CSV
    pd.DataFrame(edges, columns=["source", "target", "similarity_score"]).to_csv("./data/edges.csv", index=False)

if __name__ == "__main__":
    create_edges_csv()
