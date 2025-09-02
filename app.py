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
    "Usa la información de los documentos y el historial como referencia, no como cita literal. Inspírate en ellos para dar respuestas fluidas, como si fueran tus propios recuerdos. "
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
            # Solo agregamos el texto de la respuesta, sin etiquetas de chat
            historial.append(conversacion['respuesta'])
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

    # --- Ruta raíz para monitoreo (UptimeRobot) ---
@app.get("/")
def read_root():
    return {"status": "ok", "message": "soto API is alive 🚀"}


@app.post("/preguntar")
def preguntar(datos_pregunta: Pregunta):
    # 1️⃣ Carga la memoria del usuario (solo texto plano)
    historial = cargar_memoria(datos_pregunta.user_id)

    # 2️⃣ Buscar los nodos más relevantes en el índice
    # Traemos, por ejemplo, los 5 más relevantes
    query_engine = indice.as_query_engine(similarity_top_k=5)
    resultados = query_engine.query(datos_pregunta.pregunta)

    # Extraemos los textos de los nodos (según versión de LlamaIndex)
    if hasattr(resultados, 'source_nodes'):
        nodos = resultados.source_nodes
        contexto = ""
        for i, nodo in enumerate(nodos, 1):
            # Enumeramos cada nodo con título y contenido si existe
            titulo = getattr(nodo.node, "metadata", {}).get("nombre", f"Proyecto {i}")
            texto = nodo.node.get_content()
            contexto += f"{i}. {titulo}: {texto}\n"
    else:
        contexto = str(resultados)

    # 3️⃣ Construir prompt limpio con personalidad de soto
    prompt_soto = f"""
    Eres soto, artista virtual.
    Has recibido la pregunta: "{datos_pregunta.pregunta}"
    
    Aquí hay proyectos relevantes para responder (usa la info como guía, no repitas literal):
    {contexto}

    Si algo no se menciona, inventa coherente con tu personalidad.
    Nunca digas que no sabes ni que no hay documentos.
    Habla siempre en primera persona, usa 'soto'.
    Historial de conversaciones anteriores: {historial}
    """

    # 4️⃣ Generar la respuesta
    respuesta_texto = Settings.llm.complete(prompt_soto).text.strip()

    # 5️⃣ Guardar la conversación
    guardar_conversacion(datos_pregunta.user_id, datos_pregunta.pregunta, respuesta_texto)

    return {"respuesta": respuesta_texto}
