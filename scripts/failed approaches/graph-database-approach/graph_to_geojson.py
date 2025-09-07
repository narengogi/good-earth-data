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
    for i,rect in enumerate(packed):
        print(f"computing positions for subgraph {i} of {len(packed)}")
        _, x, y, w, h, rid = rect
        subgraph = squares[rid][0]
        pos = subgraph.layout_random()
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
                    "id": str(int(vertex['name'])),
                    "title": str(vertex['title']),
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
                    "source": str(int(u_data['name'])),
                    "target": str(int(v_data['name'])),
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
    # print("loading edges")
    # edges = pd.read_csv("data/sample.csv", usecols=['source', 'target', 'similarity_score'])
    # print("creating graph")
    # G = ig.Graph.TupleList(edges.values, directed=True, edge_attrs=['similarity_score'])
    # # explore_graph(G)

    # print("loading books")
    # books_df = pd.read_csv("data/books/merged_final.csv", usecols=['book_id', 'title'])

    # print("adding titles to vertices")
    # G.vs['title'] = [books_df.loc[books_df['book_id'] == v['name'], 'title'].values[0] for v in G.vs]
    # G.write_pickle("data/graph.pickle")

    G = pickle.load(open("data/graph.pickle", "rb"))

    print("computing clusters")
    partition = leidenalg.find_partition(G,leidenalg.ModularityVertexPartition, weights=G.es['similarity_score'], max_comm_size=5000)

    print("adding positions to vertices")
    subgraphs_with_positions = add_postions_to_vertices(partition)
    # save subgraphs with positions
    geojson = digraph_to_geojson(subgraphs_with_positions)
    with open(f"data/final_graph.geojson", "w") as f:
        json.dump(geojson, f)

    print(f"clusters computed")