# from utils.populate_database import main
# from utils.extract_kg import main
import utils.extract_kg
import utils.populate_database
import utils.reset
from neo4j import GraphDatabase
import os
from dotenv import load_dotenv

load_dotenv()

database = os.getenv("NEO4J_DATABASE")
 # Connect to Neo4j
driver = GraphDatabase.driver(uri=os.environ["NEO4J_URI"], auth=(os.environ["NEO4J_USERNAME"], os.environ["NEO4J_PASSWORD"]), database=database)

if __name__ == "__main__":
    utils.populate_database.main()
    print("Vector Database generated!\n")    
    utils.reset.clear_kg()
    utils.extract_kg.main(
        
    )
    print("Knowledge graph generated!\n")