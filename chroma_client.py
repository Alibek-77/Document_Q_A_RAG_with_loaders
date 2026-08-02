from chromadb import PersistentClient
from chromadb.utils import embedding_functions
from dotenv import load_dotenv
load_dotenv()
client=PersistentClient(path="./chroma_db")
embedding_fc=embedding_functions.GoogleGeminiEmbeddingFunction()
collection=client.get_or_create_collection(name="documents",embedding_function=embedding_fc,metadata={"hnsw:space": "cosine"})
