from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from llama_index.core import VectorStoreIndex, Settings
from llama_index.core import load_index_from_storage, StorageContext
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

# --- Cargar el índice desde el almacenamiento ---
def cargar_indice():
    print("Cargando el índice desde el almacenamiento...")
    try:
        storage_context = StorageContext.from_defaults(persist_dir="./storage")
        indice = load_index_from_storage(storage_context)
        print("Índice cargado.")
        return indice
    except Exception as e:
        print(f"Error al cargar el índice: {e}")
        raise RuntimeError("El índice no se pudo cargar. Asegúrate de haberlo construido con 'build_index.py' y de subir la carpeta 'storage'.")

indice = cargar_indice() # Carga el índice al iniciar la app

# --- Configuración de FastAPI ---
app = FastAPI(
    title="Soto AI API",
    description="Una API para conversar con el personaje Soto."
)

# --- Configuración de CORS ---
origins = [
    "https://xavi-soto.github.io",
    "https://sotosotosoto.com",
    "http://127.0.0.1:5500"
]

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

soto_template = PromptTemplate(
    "Eres soto, un artista virtual que representa a Xavier Soto en exposiciones y espacios. Haz mostrado tu trabajo en distintos lugares, desde Madrid hasta Guadalajara, pasando por Valencia y Arequipa. Te interesa lo crudo, lo directo, lo que incomoda un poco y hace pensar. Tu voz es crítica, con sarcasmo y un toque de crudeza; la sátira y la ironía son tus herramientas para cuestionar lo cotidiano y revelar lo absurdo que normalmente pasa desapercibido. "
    "Tu objetivo es responder todo lo que te pregunten desde tu desarrollo artístico. Tu papel es funcionar como un artista, así que toma esa voz para dar respuestas, pero siempre intentarás contestar cualquier cosa basándote en la información que tienes sobre tu obra, tus procesos y tu perspectiva. "
    "Hablas en un tono informal y amigable. Siempre respondes en español. "
    "Siempre tus respuestas se centran en el arte y tu trabajo; sin embargo, si surge una pregunta que no lo sea, puedes inventar una respuesta deduciendo cómo contestarías con base en los textos y en tu personalidad e identidad. "
    "Historial de conversación con este usuario: {chat_history}\n"
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
            historial.append(f"Usuario: {conversacion['pregunta']}\nSoto: {conversacion['respuesta']}\n")
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
    historial = cargar_memoria(datos_pregunta.user_id)
    query_engine = indice.as_query_engine(text_qa_template=soto_template)
    soto_template.template = soto_template.template.replace("{chat_history}", historial)
    respuesta = query_engine.query(datos_pregunta.pregunta)
    guardar_conversacion(datos_pregunta.user_id, datos_pregunta.pregunta, str(respuesta))
    return {"respuesta": str(respuesta)}