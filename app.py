from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader, Settings
from llama_index.llms.openai import OpenAI
from llama_index.core.prompts import PromptTemplate
import os
import sqlite3   # 👈 nuevo: para usar SQLite
from datetime import datetime
from fastapi.responses import HTMLResponse   # 👈 nuevo
from fastapi import Query   # 👈 nuevo (para pasar el token por URL)



# --- Configuración de OpenAI ---
if not os.getenv("OPENAI_API_KEY"):
    raise ValueError("La clave de API de OpenAI no está configurada como variable de entorno.")

# Configura tu LLM
Settings.llm = OpenAI(model="gpt-3.5-turbo")
Settings.chunk_size = 512

# --- Funciones para cargar el índice ---
def crear_indice():
    print("[DEBUG] Creando el índice desde carpeta 'data'...")
    reader = SimpleDirectoryReader(input_dir="data")
    documentos = reader.load_data()
    indice = VectorStoreIndex.from_documents(documentos)
    print("[DEBUG] Índice creado con", len(documentos), "documentos.")
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

# --- Template de personalidad ---
soto_template = PromptTemplate(
  "Responde siempre como soto, en primera persona. "
  "soto es un artista virtual que representa a Xavier Soto. "
  "Su voz es crítica, sarcástica, con ironía y crudeza; la sátira es su forma natural de hablar y opinar. "
  "Tu tarea es contestar todas las preguntas del usuario con este estilo. "
  "Usa la información de los documentos y el historial como referencia, no como cita literal. "
  "Inspírate en ellos para dar respuestas fluidas, como si fueran tus propios recuerdos. "
  "Si la pregunta es ambigua, vaga o no hay información en los documentos, igualmente responde. "
  "Nunca digas que no sabes, ni declares que no hay datos. "
  "Cuando te refieras a ti mismo, usa siempre 'soto'. "
  "Hablas en español. "
  "Historial: {chat_history}\n"
  "Contexto: {context_str}\n"
  "Pregunta: {query_str}\n"
  "Respuesta: "
)

# --- Base de datos SQLite ---
DB_FILE = "conversaciones.db"

def init_db():
    conn = sqlite3.connect(DB_FILE)
    c = conn.cursor()
    c.execute("""
        CREATE TABLE IF NOT EXISTS conversaciones (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id TEXT,
            pregunta TEXT,
            respuesta TEXT,
            timestamp TEXT
        )
    """)
    conn.commit()
    conn.close()
    print(f"[DEBUG] Base de datos lista en {DB_FILE}")

init_db()


def cargar_memoria(user_id):
    conn = sqlite3.connect(DB_FILE)
    c = conn.cursor()
    c.execute("""
        SELECT respuesta FROM conversaciones
        WHERE user_id = ?
        ORDER BY timestamp DESC
        LIMIT 10
    """, (user_id,))
    rows = c.fetchall()
    conn.close()

    historial = [r[0] for r in rows]
    print(f"[DEBUG] Cargado historial con {len(historial)} respuestas previas para {user_id}")
    return "\n".join(historial)


def guardar_conversacion(user_id, pregunta, respuesta):
    conn = sqlite3.connect(DB_FILE)
    c = conn.cursor()
    c.execute("""
        INSERT INTO conversaciones (user_id, pregunta, respuesta, timestamp)
        VALUES (?, ?, ?, ?)
    """, (user_id, pregunta, respuesta, datetime.now().isoformat()))
    conn.commit()
    conn.close()
    print(f"[DEBUG] Conversación guardada en la base de datos para user_id={user_id}")


# --- Ruta raíz para monitoreo ---
@app.get("/")
def read_root():
    print("[DEBUG] GET / llamado (status check)")
    return {"status": "ok", "message": "soto API is alive 🚀"}

@app.post("/preguntar")
def preguntar(datos_pregunta: Pregunta):
    print(f"[DEBUG] POST /preguntar con user_id={datos_pregunta.user_id}, pregunta='{datos_pregunta.pregunta}'")

    # 1️⃣ Carga la memoria
    historial = cargar_memoria(datos_pregunta.user_id)

    # 2️⃣ Buscar en el índice
    query_engine = indice.as_query_engine(similarity_top_k=5)
    resultados = query_engine.query(datos_pregunta.pregunta)

    if hasattr(resultados, 'source_nodes'):
        nodos = resultados.source_nodes
        contexto = ""
        for i, nodo in enumerate(nodos, 1):
            titulo = getattr(nodo.node, "metadata", {}).get("nombre", f"Proyecto {i}")
            texto = nodo.node.get_content()
            contexto += f"{i}. {titulo}: {texto}\n"
    else:
        contexto = str(resultados)

    # 3️⃣ Construir prompt
    prompt_soto = f"""
    Eres soto, artista virtual.
    Pregunta: "{datos_pregunta.pregunta}"

    Contexto relevante:
    {contexto}

    Historial previo:
    {historial}

    Responde como soto.
    """

    print("[DEBUG] Prompt construido, enviando a OpenAI...")

    # 4️⃣ Generar respuesta
    respuesta_texto = Settings.llm.complete(prompt_soto).text.strip()
    print(f"[DEBUG] Respuesta generada: {respuesta_texto[:80]}...")

    # 5️⃣ Guardar conversación
    guardar_conversacion(datos_pregunta.user_id, datos_pregunta.pregunta, respuesta_texto)

    return {"respuesta": respuesta_texto}




# --- Ruta para ver las conversaciones en HTML ---
SECRET_TOKEN = os.getenv("DEBUG_TOKEN", "SOTO123")  # 👈 cambia en Render si quieres más seguridad

@app.get("/verdb", response_class=HTMLResponse)
def ver_db(token: str = Query(...)):
    if token != SECRET_TOKEN:
        return HTMLResponse("<h2>⛔ Acceso denegado</h2>", status_code=403)

    conn = sqlite3.connect(DB_FILE)
    c = conn.cursor()
    c.execute("SELECT id, user_id, pregunta, respuesta, timestamp FROM conversaciones ORDER BY id DESC LIMIT 50")
    rows = c.fetchall()
    conn.close()

    html = """
    <html>
    <head>
        <style>
            body { font-family: Arial, sans-serif; padding: 20px; }
            table { border-collapse: collapse; width: 100%; }
            th, td { border: 1px solid #ddd; padding: 8px; }
            th { background-color: #333; color: white; }
            tr:nth-child(even) { background-color: #f2f2f2; }
            td { vertical-align: top; }
        </style>
    </head>
    <body>
        <h2>📂 Conversaciones guardadas</h2>
        <table>
            <tr>
                <th>ID</th><th>User</th><th>Pregunta</th><th>Respuesta</th><th>Timestamp</th>
            </tr>
    """
    for row in rows:
        html += f"<tr><td>{row[0]}</td><td>{row[1]}</td><td>{row[2]}</td><td>{row[3]}</td><td>{row[4]}</td></tr>"

    html += "</table></body></html>"
    return HTMLResponse(content=html)
