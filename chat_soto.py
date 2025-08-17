# Importaciones corregidas
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader, Settings
from llama_index.llms.openai import OpenAI

# Configura tu LLM usando Settings
Settings.llm = OpenAI(model="gpt-3.5-turbo")

def crear_indice():
    # Usa SimpleDirectoryReader para leer todos los archivos del directorio 'data'
    # Esta es la forma más común y flexible de cargar datos.
    reader = SimpleDirectoryReader(input_dir="data")
    documentos = reader.load_data()
    
    # Crea el índice
    indice = VectorStoreIndex.from_documents(documentos)
    return indice

def responder(indice, pregunta):
    query_engine = indice.as_query_engine()
    respuesta = query_engine.query(pregunta)
    return respuesta