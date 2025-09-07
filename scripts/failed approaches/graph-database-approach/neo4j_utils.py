from neo4j import GraphDatabase

def get_driver():
    """
    Get a connection to the database.
    """
    return GraphDatabase.driver("bolt://localhost:7687", auth=("neo4j", "goodreads123"))