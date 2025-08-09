from llama_index.core import VectorStoreIndex, StorageContext, load_index_from_storage, Settings
from llama_index.embeddings.huggingface import HuggingFaceEmbedding

# 1️⃣ Configuración para trabajar 100% local
Settings.embed_model = HuggingFaceEmbedding(model_name="sentence-transformers/all-MiniLM-L6-v2")
Settings.llm = None  # 🔹 Desactivamos cualquier modelo que pida API externa

# 2️⃣ Cargar índice existente
storage_context = StorageContext.from_defaults(persist_dir="./soto_index")
index = load_index_from_storage(storage_context)

# 3️⃣ Motor de consultas
query_engine = index.as_query_engine()

# 4️⃣ Bucle de preguntas
while True:
    pregunta = input("Pregunta para SOTO (o escribe 'salir' para terminar): ")
    if pregunta.lower() == "salir":
        break
    
    respuesta = query_engine.query(pregunta)
    print(f"\n💬 SOTO: {respuesta}\n")
