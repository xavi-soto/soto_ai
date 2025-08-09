from llama_index.core import VectorStoreIndex, StorageContext, load_index_from_storage, Settings
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.core.llms.mock import MockLLM  # Import correcto para versiones recientes

# Configuración local (usa embeddings y un LLM falso para pruebas)
Settings.embed_model = HuggingFaceEmbedding(model_name="sentence-transformers/all-MiniLM-L6-v2")
Settings.llm = MockLLM()  # Esto evita usar OpenAI y da respuestas de prueba

# Cargar el índice desde la carpeta persistida
storage_context = StorageContext.from_defaults(persist_dir="./soto_index")
index = load_index_from_storage(storage_context)

# Motor de consultas
query_engine = index.as_query_engine()

# Bucle de interacción
while True:
    pregunta = input("Pregunta para SOTO (o escribe 'salir'): ")
    if pregunta.lower() == "salir":
        break
    respuesta = query_engine.query(pregunta)
    print(f"\n💬 SOTO: {respuesta}\n")
