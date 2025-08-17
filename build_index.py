import os
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader, Settings
from llama_index.core.storage.storage_context import StorageContext
from llama_index.core.readers.base import Document
from llama_index.llms.openai import OpenAI
from llama_index.core.node_parser import SimpleNodeParser

# Asegúrate de que tu clave de API esté configurada como variable de entorno
os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")

Settings.llm = OpenAI(model="gpt-3.5-turbo")
Settings.chunk_size = 512

# Carga tus documentos
reader = SimpleDirectoryReader(input_dir="data")
documents = reader.load_data()

# Crea y persiste el índice
print("Creando el índice y guardándolo...")
index = VectorStoreIndex.from_documents(documents)
index.storage_context.persist(persist_dir="./storage")
print("Índice guardado en la carpeta 'storage'.")