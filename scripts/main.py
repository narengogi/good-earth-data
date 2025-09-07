import json
import pandas as pd
import os
import networkx as nx
import pickle
import numpy as np
# from sklearn.cluster import 

def cluster_books_and_create_geojson():
    # if os.path.exists("./subgraphs.pkl"):
    #     subgraphs = pickle.load(open("./subgraphs.pkl", "rb"))
    # else:
    print("Reading edges")
    edges = pd.read_parquet("./edges.parquet").astype({"book_id_x": str, "book_id_y": str})
    
    print("creating graph")
    G = nx.Graph(directed=False)
    G.add_weighted_edges_from(edges[["book_id_x", "book_id_y", "similarity_score"]].values)
    print("Reading books into memory")
    books_df = pd.read_parquet("./books.parquet")

    print("Clustering books")
    subgraphs = nx.community.louvain_communities(G, weight="similarity_score")
    # subgraphs = nx.louvain(G, weight="similarity_score")
    # pickle.dump(subgraphs, open("./subgraphs.pkl", "wb"))

    print("Creating geojson")
    total_size = 0
    # pack into a rectangle
    total_size = sum(len(subgraph) for subgraph in subgraphs)
    container_width = total_size / 10
    container_height = total_size / 10
    packed_dict = {i: (0, 0, container_width, container_height) for i in range(len(subgraphs))}
    curr_x, curr_y, max_y = 0, 0, 0
    for i, subgraph in enumerate(subgraphs):
        if (curr_x + len(subgraph) > container_width):
            curr_x = 0
            curr_y += max_y
        max_y = max(max_y, len(subgraph))
        packed_dict[i] = (curr_x, curr_y, len(subgraph), len(subgraph))
        curr_x += len(subgraph)

    vertex_features = []
    for i, subgraph in enumerate(subgraphs):
        print(f"Layouting subgraph {i}")
        x, y, width, height = packed_dict[i]
        pos = nx.random_layout(subgraph)
        print("adjusting positions based on container size")
        for id, pos_tuple in pos.items():
            pos[id] = (float(pos_tuple[0]*360*width/total_size + 360*x/container_width - 180), float(pos_tuple[1]*120*height/total_size + 120*y/container_height - 60))
        for id, p in pos.items():
            book = books_df[books_df["book_id"] == id].iloc[0]
            book_properties = {
                   "title": str(book["title"]),
                    "size": book["count"] or 1,
                    "author_names": ", ".join(book["author_names"]) if isinstance(book["author_names"], np.ndarray) else book["author_names"] or "",
                    "author_ids": ", ".join(book["author_ids"]) if isinstance(book["author_ids"], np.ndarray) else book["author_ids"] or "",
                    "publisher": book["publisher"] or "",
                    "publication_year": book["publication_year"] or "",
                    "average_rating": book["average_rating"] or "",
                    "description": book["description"] or "",
            } if book is not None else {}
            lat = p[1]
            lon = p[0]
            vertex_features.append({
                "type": "Feature",
                "geometry": {
                    "type": "Point",
                    "coordinates": [lon, lat],
                },
                "properties": {
                    "id": str(id),
                    "groupId": i,
                    **book_properties
                }
            })
            edge_features = []
            for edge in G.edges(pos.keys()):
                edge_features.append({
                    "type": "Feature",
                    "geometry": {
                        "type": "LineString",
                        "coordinates": [
                            [pos[edge[0]][0], pos[edge[0]][1]],
                            [pos[edge[1]][0], pos[edge[1]][1]]
                        ]
                    },
                    "properties": {
                        "source": str(edge[0]),
                        "target": str(edge[1]),
                    }
                })
            edges_geojson = {
                "type": "FeatureCollection",
                "features": edge_features
            }
            with open(f"./edges/subgraph_{i}.geojson", "w") as f:
                json.dump(edges_geojson, f)
    vertex_geojson = {
        "type": "FeatureCollection",
        "features": vertex_features
    }
    with open(f"./vertices.geojson", "w") as f:
        json.dump(vertex_geojson, f)

def get_occurences_by_book_id():
    behaviour = pd.DataFrame()
    for file in os.listdir("./behaviour"):
        print(f"Reading {file}")
        behaviour = pd.concat([behaviour, pd.read_parquet(f"./behaviour/{file}")])
    print("filtering out users who've read more than 300 books")
    book_counts_per_user = behaviour.groupby("user_id")["book_id"].size().reset_index(name="book_count")
    filtered_users = book_counts_per_user[book_counts_per_user["book_count"] <= 300]
    behaviour = behaviour.merge(filtered_users[["user_id"]], on="user_id", how="inner")
    print("grouping by book_id and counting occurences")
    behaviour = behaviour.groupby("book_id").size().reset_index(name="count")
    behaviour.to_parquet("./occurences_by_book_id.parquet", index=False)
    books_df = pd.read_parquet("./books.parquet")
    books_df = books_df.drop(columns=["count"])
    books_df = books_df.merge(behaviour, on="book_id", how="left")
    books_df.to_parquet("./books_with_occurences.parquet", index=False)


def merge_co_counts_and_compute_top_10_similarities():
    co_counts = pd.DataFrame()
    for file in os.listdir("./co_counts"):
        print(f"Reading {file}")
        co_counts = pd.concat([co_counts, pd.read_parquet(f"./co_counts/{file}")])
    
    print("dropping duplicates")
    co_counts = co_counts.drop_duplicates()

    print("aggregating co-counts")
    co_counts = co_counts.groupby(["book_id_x", "book_id_y"]).agg({"co_count": "sum"}).reset_index()
    
    print("loading books")
    aggrgated_user_counts_by_book = pd.read_parquet("./books.parquet", columns=["book_id", "count"]) # colums: book_id, count
    print("Updating user_count_x and user_count_y with the aggregated user counts by book")
    co_counts = co_counts.merge(aggrgated_user_counts_by_book, left_on="book_id_x", right_on="book_id", how="left")
    co_counts = co_counts.drop(columns=["book_id"])
    co_counts = co_counts.merge(aggrgated_user_counts_by_book, left_on="book_id_y", right_on="book_id", how="left")
    co_counts = co_counts.drop(columns=["book_id"])

    print("Computing similarity scores")
    co_counts["similarity_score"] = co_counts["co_count"] / (co_counts["count_x"] + co_counts["count_y"] - co_counts["co_count"])

    # replace infinity with 1
    co_counts["similarity_score"] = co_counts["similarity_score"].replace([np.inf, -np.inf], 1)

    print("Keeping top 10 most similar books for each book")
    co_counts = co_counts.sort_values(by="similarity_score", ascending=False).groupby("book_id_x").head(10)

    print("Saving results")
    co_counts.to_csv("./edges.csv", index=False, columns=["book_id_x", "book_id_y", "similarity_score", "co_count", "count_x", "count_y"])    
    co_counts.to_parquet("./edges.parquet", index=False)

def compute_co_counts_per_chunk_and_save_top_30_for_each_book():
    for file in os.listdir("./behaviour"):
        print("Reading data")
        df = pd.read_parquet(f"./behaviour/{file}", columns=["user_id", "book_id"])

        print("computing number of users who've read each book")
        book_counts = df.groupby("book_id").size().reset_index(name="user_count")

        print("Computing book counts per user and dropping users who've read more than 300 books")
        book_counts_per_user = df.groupby("user_id")["book_id"].size().reset_index(name="book_count")
        filtered_users = book_counts_per_user[book_counts_per_user["book_count"] <= 300]
        df = df.merge(filtered_users[["user_id"]], on="user_id", how="inner")

        print("Self join to get all pairs of books read by the same user")
        pairs = df.merge(df, on="user_id", how="inner")
        print("avoid duplicate and symmetric pairs")
        pairs = pairs[pairs["book_id_x"] < pairs["book_id_y"]]
        
        print("Computing co-occurrence counts")
        co_counts = pairs.groupby(["book_id_x", "book_id_y"]).size().reset_index(name="co_count")
        co_counts = co_counts.merge(
            book_counts.rename(columns={"book_id": "book_id_x", "user_count": "user_count_x"}),
            on="book_id_x", how="left"
        )
        co_counts = co_counts.merge(
            book_counts.rename(columns={"book_id": "book_id_y", "user_count": "user_count_y"}),
            on="book_id_y", how="left"
        )

        print("Saving top 100 co-counts for each book")
        co_counts_x = co_counts.sort_values(by="co_count", ascending=False).groupby("book_id_x").head(30)
        co_counts_y = co_counts.sort_values(by="co_count", ascending=False).groupby("book_id_y").head(30)
        co_counts = pd.concat([co_counts_x, co_counts_y])
        co_counts.to_parquet(f"./co_counts/{file}", index=False)


if __name__ == "__main__":
    get_occurences_by_book_id()
    compute_co_counts_per_chunk_and_save_top_30_for_each_book()
    merge_co_counts_and_compute_top_10_similarities()
    # cluster_books_and_create_geojson()