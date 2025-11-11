import os
from dotenv import load_dotenv
from langchain_community.graphs import Neo4jGraph
from langchain_community.document_loaders import WikipediaLoader
from langchain.text_splitter import TokenTextSplitter
from langchain_experimental.graph_transformers import LLMGraphTransformer
import pandas as pd
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import Neo4jVector

# Load environment variables
load_dotenv()

# Neo4j connection
NEO4J_URI = os.getenv("NEO4J_URI")
NEO4J_USERNAME = os.getenv("NEO4J_USERNAME")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD")

# Initialize Neo4j graph
graph = Neo4jGraph(
    url=NEO4J_URI,
    username=NEO4J_USERNAME,
    password=NEO4J_PASSWORD
)

# Initialize clients
llm = OpenAI(
    base_url="https://api.openrouter.ai/v1",  
    api_key=os.getenv("OPENROUTER_API_KEY")   
)

# Initialize embeddings using a HuggingFace model
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

# Create a vector store instance from the existing graph
vector_index = Neo4jVector.from_existing_graph(
    embeddings,
    search_type="hybrid",
    node_label="Document",
    text_node_properties=["text"],
    embedding_node_property="embedding"
)

def build_knowledge_graph_from_entities(entities_df):
    """
    Build knowledge graph from the top 20 entities
    """
    # Initialize LLM for graph transformation
    llm_transformer = LLMGraphTransformer(llm=llm)  # Use the initialized OpenAI client

    # Process top 20 entities
    for _, row in entities_df.head(20).iterrows():
        entity_name = row['Word']
        print(f"Processing entity: {entity_name}")
        
        # Step 1: Load Wikipedia content using a query
        raw_documents = WikipediaLoader(query=entity_name).load()
        
        if not raw_documents:
            continue
        
        # Step 2: Split documents into token-sized chunks
        text_splitter = TokenTextSplitter(chunk_size=256, chunk_overlap=50)
        documents = text_splitter.split_documents(raw_documents)
        
        # Step 3: Convert document chunks into graph-structured format
        graph_documents = llm_transformer.convert_to_graph_documents(documents)
        
        # Step 4: Add graph documents to the knowledge graph
        graph.add_graph_documents(graph_documents, baseEntityLabel=True, include_source=True)

if __name__ == "__main__":
    # Load entities from the CSV file
    entities_file = r"f:\EUCLIDO\Tasks\_____self\semeval-humayun\entities_for_wiki.csv"
    
    if not os.path.exists(entities_file):
        print(f"File not found: {entities_file}")
    else:
        print(f"Loading entities from: {entities_file}")
        entities_df = pd.read_csv(entities_file)
        
        # Build knowledge graph
        build_knowledge_graph_from_entities(entities_df)