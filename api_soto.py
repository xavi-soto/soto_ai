from fastapi import FastAPI
from pydantic import BaseModel
import json
import os
from llama_index.core import VectorStoreIndex, StorageContext, load_index_from_storage, Settings
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.llms.ollama import Ollama
from llama_index.core.memory import ChatMemoryBuffer
from llama_index.core.base.llms.types import ChatMessage

app = FastAPI()

MEMORY_FILE = "soto_memoria.json"

memory = ChatMemoryBuffer.from_defaults(token_limit=2000)
if os.path.exists(MEMORY_FILE):
    try:
        with open(MEMORY_FILE, "r", encoding="utf-8") as f:
            memoria_data = json.load(f)
        if isinstance(memoria_data, list):
            for msg in memoria_data:
                memory.put(ChatMessage(role=msg.get("role", "user"), content=msg.get("content", "")))
    except:
        memory = ChatMemoryBuffer.from_defaults(token_limit=2000)

Settings.embed_model = HuggingFaceEmbedding(model_name="sentence-transformers/all-MiniLM-L6-v2")
Settings.llm = Ollama(model="mistral", request_timeout=120.0)

storage_context = StorageContext.from_defaults(persist_dir="./soto_index")
index = load_index_from_storage(storage_context)

system_prompt = """
Eres SOTO, artista visual, creativo, con humor mexicano, directo e ingenioso.
Siempre responde con personalidad aunque no tenga datos específicos.
"""

query_engine = index.as_chat_engine(
    chat_mode="context",
    memory=memory,
    system_prompt=system_prompt,
    similarity_top_k=3
)

class Message(BaseModel):
    text: str
    
@app.get("/")
async def root():
    return {"status": "SOTO API funcionando"}

@app.post("/chat/")
async def chat(message: Message):
    user_text = message.text
    response = query_engine.chat(user_text)
    memory.put(ChatMessage(role="user", content=user_text))
    memory.put(ChatMessage(role="assistant", content=response))
    memoria_serializada = [{"role": msg.role, "content": msg.content} for msg in memory.get_all()]
    with open(MEMORY_FILE, "w", encoding="utf-8") as f:
        json.dump(memoria_serializada, f, ensure_ascii=False, indent=2)
    return {"response": response}
