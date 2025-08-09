import json
import os
from llama_index.core import VectorStoreIndex, StorageContext, load_index_from_storage, Settings
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.llms.ollama import Ollama
from llama_index.core.memory import ChatMemoryBuffer
from llama_index.core.base.llms.types import ChatMessage

MEMORY_FILE = "soto_memoria.json"

# ===== MEMORIA =====
memory = ChatMemoryBuffer.from_defaults(token_limit=2000)
if os.path.exists(MEMORY_FILE):
    try:
        with open(MEMORY_FILE, "r", encoding="utf-8") as f:
            memoria_data = json.load(f)
        if isinstance(memoria_data, list):
            for msg in memoria_data:
                memory.put(ChatMessage(role=msg.get("role", "user"), content=msg.get("content", "")))
    except (json.JSONDecodeError, OSError):
        print("⚠ Archivo de memoria corrupto o vacío, iniciando nueva memoria.")
        memory = ChatMemoryBuffer.from_defaults(token_limit=2000)

# ===== CONFIGURACIÓN =====
Settings.embed_model = HuggingFaceEmbedding(model_name="sentence-transformers/all-MiniLM-L6-v2")
ollama_model = Ollama(model="mistral", request_timeout=120.0)

storage_context = StorageContext.from_defaults(persist_dir="./soto_index")
index = load_index_from_storage(storage_context)

# ===== PROMPT DE SOTO =====
system_prompt = """
Eres SOTO, artista visual y personaje virtual.  
Tono: ingenioso, ácido, humor mexicano, breve pero contundente.  
No repitas textos tal cual, reescribe siempre como si estuvieras charlando.  
Usa lo que recuerdas del usuario y lo que encuentres en tus documentos para contestar.  
Si no hay datos, inventa algo coherente y divertido.
"""

print("💬 SOTO está listo con personalidad y memoria. Escribe 'salir' para terminar.")

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

    # Paso 1: Buscar en el índice
    query_engine = index.as_query_engine(similarity_top_k=3)
    resultado_json = query_engine.query(pregunta)
    contexto_docs = str(resultado_json)

    # Paso 2: Juntar memoria + documentos + pregunta
    historial = "\n".join([f"{msg.role}: {msg.content}" for msg in memory.get_all()])
    prompt_final = f"""{system_prompt}

Historial reciente:
{historial}

Documentos relevantes:
{contexto_docs}

Pregunta:
{pregunta}

Responde como SOTO, combinando lo que recuerdas y lo que encontraste.
"""

    # Paso 3: Pasar a Ollama para reescribir al estilo SOTO
    respuesta = ollama_model.complete(prompt_final)

    # Guardar en memoria
    memory.put(ChatMessage(role="user", content=pregunta))
    memory.put(ChatMessage(role="assistant", content=respuesta))

    print(f"SOTO: {respuesta}\n")



