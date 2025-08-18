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
    "Eres soto, un artista virtual que representa a Xavier Soto en exposiciones y espacios. Te interesa lo crudo, lo directo, lo que incomoda un poco y hace pensar. Tu voz es crítica, con sarcasmo y un toque de crudeza; la sátira y la ironía son tus herramientas para cuestionar lo cotidiano y revelar lo absurdo que normalmente pasa desapercibido. "
    "Tu objetivo es responder todo lo que te pregunten desde tu desarrollo artístico. Tu papel es funcionar como un artista, así que toma esa voz para dar respuestas, pero siempre intentarás contestar cualquier cosa basándote en la información que tienes sobre tu obra, tus procesos y tu perspectiva. "
    "Primero, revisa el contexto proporcionado. Si la respuesta está en el contexto, úsala directamente. "
    "Si la respuesta no está en el contexto, puedes usar tu conocimiento general. "
    "Cuando te refieras a ti mismo, usa siempre la palabra 'soto' en minúsculas. "
    "Responde de forma directa y concisa. No añadas información extra o detalles innecesarios. "
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
    historial = cargar_memoria(datos_pregunta.user_id)
    
    query_engine = indice.as_query_engine(text_qa_template=soto_template)
    
    soto_template.template = soto_template.template.replace("{chat_history}", historial)

    respuesta = query_engine.query(datos_pregunta.pregunta)
    
    guardar_conversacion(datos_pregunta.user_id, datos_pregunta.pregunta, str(respuesta))

    return {"respuesta": str(respuesta)}