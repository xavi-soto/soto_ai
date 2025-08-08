import json
import os
from llama_index.core import VectorStoreIndex, StorageContext, load_index_from_storage, Settings
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.llms.ollama import Ollama
from llama_index.core.memory import ChatMemoryBuffer
from llama_index.core.base.llms.types import ChatMessage  # ← Import necesario

MEMORY_FILE = "soto_memoria.json"

# Crear memoria vacía por defecto
memory = ChatMemoryBuffer.from_defaults(token_limit=2000)

# Cargar memoria previa si existe y es válida
if os.path.exists(MEMORY_FILE):
    try:
        with open(MEMORY_FILE, "r", encoding="utf-8") as f:
            memoria_data = json.load(f)
        if isinstance(memoria_data, list):
            for msg in memoria_data:
                role = msg.get("role", "user")
                content = msg.get("content", "")
                # CORREGIDO: ahora pasamos un ChatMessage en vez de dos args
                memory.put(ChatMessage(role=role, content=content))
    except (json.JSONDecodeError, OSError):
        print("⚠ Archivo de memoria corrupto o vacío, iniciando nueva memoria.")
        memory = ChatMemoryBuffer.from_defaults(token_limit=2000)

# Configuración del modelo y embeddings
Settings.embed_model = HuggingFaceEmbedding(model_name="sentence-transformers/all-MiniLM-L6-v2")
Settings.llm = Ollama(model="mistral", request_timeout=120.0)

# Cargar índice
storage_context = StorageContext.from_defaults(persist_dir="./soto_index")
index = load_index_from_storage(storage_context)

# Personalidad de SOTO
system_prompt = """
Eres SOTO, un personaje artístico, visual y creativo, directo, con humor mexicano.
Respondes breve, ingenioso y con estilo.
Si no sabes algo, inventa algo interesante manteniendo coherencia.
Recuerda lo que el usuario te ha dicho antes y úsalo en tus respuestas.
"""

# Motor de consultas con memoria
query_engine = index.as_chat_engine(
    chat_mode="context",
    memory=memory,
    system_prompt=system_prompt,
    similarity_top_k=3
)

print("💬 SOTO está listo con memoria persistente. Escribe 'salir' para terminar.")

while True:
    pregunta = input("Tú: ")
    if pregunta.lower() == "salir":
        memoria_serializada = [
            {"role": msg.role, "content": msg.content}
            for msg in memory.get_all()
        ]
        with open(MEMORY_FILE, "w", encoding="utf-8") as f:
            json.dump(memoria_serializada, f, ensure_ascii=False, indent=2)
        print("🧠 Memoria guardada. ¡Hasta la próxima!")
        break

    respuesta = query_engine.chat(pregunta)
    print(f"SOTO: {respuesta}\n")
