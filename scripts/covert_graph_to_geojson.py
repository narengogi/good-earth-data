import pickle
import igraph as ig
import json
import leidenalg
from rectpack import newPacker
import pandas as pd

def add_postions_to_vertices(partition):
    # compute global positions of books using subgraph size and treat it as a packing problem
    # tuple is (subgraph, size, x, y)
    squares = [(subgraph, subgraph.vcount(), None, None) for subgraph in partition.subgraphs()]

    packer = newPacker()
    total_size = 0
    for i, (subgraph, size, _, _) in enumerate(squares):
        packer.add_rect(size, size, rid=i)
        total_size += size
    container_width = total_size/10
    container_height = total_size/10
    packer.add_bin(container_width, container_height)
    packer.pack()
    packed = packer.rect_list()

    list_of_subgraphs = []
    for rect in packed:
        _, x, y, w, h, rid = rect
        subgraph = squares[rid][0]
        pos = subgraph.layout_kamada_kawai()
        updated_pos = [(pos_tuple[0]*360*w/total_size + 360*x/container_width - 180, pos_tuple[1]*120*h/total_size + 120*y/container_height - 60) for pos_tuple in pos]
        subgraph.vs['pos'] = updated_pos
        list_of_subgraphs.append(subgraph)
    return list_of_subgraphs


def digraph_to_geojson(subgraphs: list[ig.Graph]) -> dict:
    features = []

    for i,subgraph in enumerate(subgraphs):
        # 1. Add node features (Points)
        for vertex in subgraph.vs:
            lat = float(vertex['pos'][1])
            lon = float(vertex['pos'][0])
            features.append({
                "type": "Feature",
                "geometry": {
                    "type": "Point",
                    "coordinates": [lon, lat],
                },
                "properties": {
                    "id": str(vertex['book_id']),
                    "size": int(vertex['count']) if 'count' in vertex.attributes() else 1,
                    "groupId": i
                }
            })



        edge_features = []
        # 2. Add edge features (LineStrings)
        for edge in subgraph.es:
            u = edge.source
            v = edge.target
            u_data = subgraph.vs[u]
            v_data = subgraph.vs[v]
            u_lat = float(u_data['pos'][1])
            u_lon = float(u_data['pos'][0])
            v_lat = float(v_data['pos'][1])
            v_lon = float(v_data['pos'][0])
            edge_features.append({
                "type": "Feature",
                "geometry": {
                    "type": "LineString",
                    "coordinates": [
                        [u_lon, u_lat],
                        [v_lon, v_lat]
                    ]
                },
                "properties": {
                    "source": str(u_data['book_id']),
                    "target": str(v_data['book_id']),
                }
            })
        edges_geojson = {
            "type": "FeatureCollection",
            "features": edge_features
        }
        with open(f"data/edges/edges_{i}.geojson", "w") as f:
            json.dump(edges_geojson, f)
    # 3. Wrap everything in a FeatureCollection
    points_geojson = {
        "type": "FeatureCollection",
        "features": features
    }

    return points_geojson

def explore_graph(G: ig.Graph):
    print(f"number of nodes: {G.vcount()}")
    print(f"number of edges: {G.ecount()}")
    print(f"number of self loops: {len(G.es.select(_loop=True))}")
    print(f"number of isolated nodes: {len([v for v in G.vs if v.degree() == 0])}")

if __name__ == "__main__":
    with open("data/graph.pkl", "rb") as f:
        books_df_1 = pd.read_parquet("data/books/1.parquet", columns=['book_id', 'title'])
        books_df_2 = pd.read_parquet("data/books/2.parquet", columns=['book_id', 'title'])
    
        G = pickle.load(f)
        G.vs['id'] = [books_df_1.loc[books_df_1['book_id'] == v['book_id'], 'title'].values[0] for v in G.vs]

        with open("data/weights.pkl", "rb") as f:
            weights = pickle.load(f)
        partition = leidenalg.find_partition(G,leidenalg.ModularityVertexPartition, weights=weights, max_comm_size=5000)
        print(f"clusters computed")

        subgraphs_with_positions = add_postions_to_vertices(partition)
    # save subgraphs with positions
    geojson = digraph_to_geojson(subgraphs_with_positions)
    with open(f"data/final_graph.geojson", "w") as f:
        json.dump(geojson, f)
