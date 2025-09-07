import cudf as pd

def create_edges():
    print("Reading behaviour data...")
    df = pd.read_csv("./behaviour.csv", usecols=["user_id", "book_id"])
    print("Computing per-book user counts...")
    book_users = df.groupby("book_id")["user_id"]
    book_counts = df.groupby("book_id").size().reset_index(name="count")

    results = {}
    for book_id, book_users in book_users:
        current_book_count = book_counts[book_counts["book_id"] == book_id]["count"].iloc[0]
        print(f"Processing book {book_id} with {len(book_users)} users")
        co_counts = df[df["user_id"].isin(book_users)].groupby("book_id").size().reset_index(name="co_count")
        co_counts = co_counts.merge(book_counts, on="book_id", how="left")
        co_counts["count"] = co_counts["count"] + current_book_count
        co_counts["similarity_score"] = co_counts["co_count"] / co_counts["count"]
        co_counts = co_counts.sort_values(by="similarity_score", ascending=False)[1:11]
        results[book_id] = co_counts
    with open("./edges.csv", "w") as f:
        f.write("src,target,similarity_score\n")
        for book_id, co_counts in results.items():
            for _, index in co_counts.to_pandas().iterrows():
                f.write(f"{int(book_id)},{int(index['book_id'])},{index['similarity_score']}\n")

if __name__ == "__main__":
    create_edges()


