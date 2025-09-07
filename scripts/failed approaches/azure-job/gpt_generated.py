import cudf

def create_edges_batchwise_cudf_compatible(
    behavior_csv="./behaviour.csv",
    output_csv="./edges.csv",
    top_k=10
):
    print("Reading behavior data...")
    df = cudf.read_csv(behavior_csv, usecols=["book_id", "user_id"], nrows=100000)

    print("Computing per-book user counts...")
    book_user_counts = df.groupby("book_id").size().reset_index(name="user_count")

    results = []

    unique_books = df["book_id"].drop_duplicates().to_arrow().to_pylist()
    for i, book_id_1 in enumerate(unique_books):
        # Users who read this book
        users = df[df["book_id"] == book_id_1]["user_id"]

        if users.empty:
            continue

        # Books those users have read
        df_user_books = df[df["user_id"].isin(users)]

        # Count co-occurrence with other books
        co_counts = (
            df_user_books[df_user_books["book_id"] != book_id_1]
            .groupby("book_id")
            .size()
            .reset_index(name="co_count")
            .rename(columns={"book_id": "book_id_2"})
        )

        if co_counts.empty:
            continue

        # Add user counts for book1 and book2
        b1_count = book_user_counts[book_user_counts["book_id"] == book_id_1]["user_count"].iloc[0]
        b2_counts = book_user_counts.rename(columns={
            "book_id": "book_id_2", "user_count": "user_count_2"
        })
        co_counts = co_counts.merge(b2_counts, on="book_id_2", how="left")

        # Compute similarity
        co_counts["similarity"] = co_counts["co_count"] / (b1_count + co_counts["user_count_2"])

        # Select top-k
        top_k_sim = co_counts.nlargest(top_k, "similarity")

        top_k_sim["source"] = book_id_1
        top_k_sim = top_k_sim[["source", "book_id_2", "similarity"]]
        top_k_sim = top_k_sim.rename(columns={"book_id_2": "target"})

        results.append(top_k_sim)

    print("Combining results...")
    all_edges = cudf.concat(results)
    print(f"Writing {len(all_edges):,} edges to {output_csv}")
    all_edges.to_csv(output_csv, index=False)
    print("Done.")

if __name__ == "__main__":
    create_edges_batchwise_cudf_compatible()
