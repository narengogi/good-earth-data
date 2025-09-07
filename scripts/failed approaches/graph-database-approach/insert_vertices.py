from neo4j_utils import get_driver

def insert_vertices(driver, vertices):
    batch_size = 10000
    with driver.session() as session:
        session.run("CREATE (n:Vertex) SET n = $properties", properties=vertices)

if __name__ == "__main__":
    driver = get_driver()
    with open("data/graph.pkl", "rb") as f:
        graph = pickle.load(f)
    insert_vertices(driver, graph.vs)