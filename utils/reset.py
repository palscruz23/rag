from utils.populate_database import clear_database
from neo4j import GraphDatabase
import os
from dotenv import load_dotenv

load_dotenv()

# database = os.getenv("NEO4J_DATABASE")
#  # Connect to Neo4j
# driver = GraphDatabase.driver(uri=os.environ["NEO4J_URI"], auth=(os.environ["NEO4J_USERNAME"], os.environ["NEO4J_PASSWORD"]), database=database)

# def clear_kg():
#     with driver.session() as session:
#         session.run("MATCH (n) DETACH DELETE n")
#         print("Knowledge graph cleared!")



if __name__ == "__main__":
    clear_database()
    print("Database cleared!")
    # clear_kg()