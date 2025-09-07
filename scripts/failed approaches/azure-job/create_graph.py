import igraph as ig
import pandas as pd


if __name__ == "__main__":
    print("loading edges")
    edges = pd.read_csv("edges.csv", usecols=['source', 'target', 'similarity_score'])
    print("creating graph")
    G = ig.Graph.TupleList(edges.values, directed=True, edge_attrs=['similarity_score'])
    # explore_graph(G)

    print("loading books")
    books_df = pd.read_csv("books.csv", usecols=['book_id', 'title'])

    print("adding titles to vertices")
    G.vs['title'] = [books_df.loc[books_df['book_id'] == v['name'], 'title'].values[0] for v in G.vs]
    G.write_pickle("graph.pickle")