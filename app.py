from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware  # ¡Nuevo!
from pydantic import BaseModel
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader, Settings
from llama_index.llms.openai import OpenAI
from llama_index.core.prompts import PromptTemplate
import os
import json
from datetime import datetime

# --- Configuración de OpenAI ---
if not os.getenv("OPENAI_API_KEY"):
    raise ValueError("La clave de API de OpenAI no está configurada como variable de entorno.")

# Configura tu LLM
Settings.llm = OpenAI(model="gpt-3.5-turbo")
Settings.chunk_size = 512

# --- Funciones para cargar el índice ---
def crear_indice():
    print("Creando el índice...")
    reader = SimpleDirectoryReader(input_dir="data")
    documentos = reader.load_data()
    indice = VectorStoreIndex.from_documents(documentos)
    print("Índice creado.")
    return indice

indice = crear_indice() # Carga el índice al iniciar la app

# --- Configuración de FastAPI ---
app = FastAPI(
    title="soto AI API",
    description="Una API para conversar con el personaje soto."
)

# --- Configuración de CORS ---
origins = ["*"]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class Pregunta(BaseModel):
    pregunta: str
    user_id: str

# Define el template con la personalidad de soto
soto_template = PromptTemplate(
  "Responde siempre como soto, en primera persona. "
    "soto es un artista virtual que representa a Xavier Soto. "
    "Su voz es crítica, sarcástica, con ironía y crudeza; la sátira es su forma natural de hablar y opinar. "
    "Tu tarea es contestar todas las preguntas del usuario con este estilo. "
    "Si hay información en los documentos o en el historial, úsala directamente para construir la respuesta. "
    "Si la pregunta es ambigua, vaga o no hay información en los documentos, igualmente responde: "
    "inventa un contenido coherente con la personalidad de soto, evitando decir que no sabes o que no hay datos. "
    "Nunca expliques cómo hablas ni declares tu personalidad, simplemente escribe como soto. "
    "Cuando te refieras a ti mismo, usa siempre 'soto' en minúsculas. "
    "Hablas en español. "
    "Historial de conversación: {chat_history}\n"
    "Contexto: {context_str}\n"
    "Pregunta: {query_str}\n"
    "Respuesta: "
)

# --- Funciones para manejar la memoria ---
MEMORIA_DIR = "conversaciones"
os.makedirs(MEMORIA_DIR, exist_ok=True)

def cargar_memoria(user_id):
    memoria_file = os.path.join(MEMORIA_DIR, f"{user_id}.jsonl")
    if not os.path.exists(memoria_file):
        return ""
    
    historial = []
    with open(memoria_file, "r", encoding="utf-8") as f:
        for line in f:
            conversacion = json.loads(line)
            historial.append(f"Usuario: {conversacion['pregunta']}\nsoto: {conversacion['respuesta']}\n")
    return "\n".join(historial)

def guardar_conversacion(user_id, pregunta, respuesta):
    memoria_file = os.path.join(MEMORIA_DIR, f"{user_id}.jsonl")
    conversacion = {
        "timestamp": datetime.now().isoformat(),
        "pregunta": pregunta,
        "respuesta": respuesta
    }
    with open(memoria_file, "a", encoding="utf-8") as f:
        f.write(json.dumps(conversacion, ensure_ascii=False) + "\n")
    print("Conversación guardada.")

@app.post("/preguntar")
def preguntar(datos_pregunta: Pregunta):
    # Carga la memoria de la conversación
    historial = cargar_memoria(datos_pregunta.user_id)
    
    # Crea un motor de chat para manejar la conversación
    chat_engine = indice.as_chat_engine(
        chat_mode="condense_plus_context",
        llm=Settings.llm,
        text_qa_template=soto_template
    )

    # Reemplaza el marcador de posición del historial en la plantilla
    soto_template.template = soto_template.template.replace("{chat_history}", historial)
    
    # Haz la pregunta a la IA con la plantilla y el historial
    respuesta = chat_engine.chat(datos_pregunta.pregunta)
    
    # Guarda la nueva respuesta en el historial
    guardar_conversacion(datos_pregunta.user_id, datos_pregunta.pregunta, str(respuesta))
    
    return {"respuesta": str(respuesta)}