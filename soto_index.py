from llama_index.core import VectorStoreIndex, SimpleDirectoryReader, Settings
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
import os

# Configurar embeddings locales
Settings.embed_model = HuggingFaceEmbedding(model_name="sentence-transformers/all-MiniLM-L6-v2")

DATA_DIR = "./data"

if not os.path.exists(DATA_DIR):
    print("❌ La carpeta 'data' no existe. Crea una y coloca tus archivos ahí.")
    exit()

print("📄 Cargando documentos desde:", DATA_DIR)
documents = SimpleDirectoryReader(DATA_DIR).load_data()

if not documents:
    print("❌ No se encontraron documentos en la carpeta /data.")
    exit()

print(f"✅ {len(documents)} documentos cargados.")
print("🧠 Generando índice...")
index = VectorStoreIndex.from_documents(documents)

print("💾 Guardando índice en /soto_index ...")
index.storage_context.persist(persist_dir="./soto_index")
print("✅ ¡Índice generado y guardado!")
