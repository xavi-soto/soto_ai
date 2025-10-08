from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from llama_index.core import (
    VectorStoreIndex,
    SimpleDirectoryReader, # Mantenemos esto como fallback
    Settings,
    PromptTemplate,
    StorageContext,
    load_index_from_storage,
    Document # ¡Importante!
)
from llama_index.llms.openai import OpenAI
import os
import json
import psycopg2
from datetime import datetime
from fastapi.responses import HTMLResponse, Response
from fastapi import Query

# --- Configuración de OpenAI (SIN CAMBIOS) ---
if not os.getenv("OPENAI_API_KEY"):
    raise ValueError("La clave de API de OpenAI no está configurada como variable de entorno.")

Settings.llm = OpenAI(model="gpt-3.5-turbo")
Settings.chunk_size = 512

# --- ¡NUEVA FUNCIÓN MEJORADA PARA CARGAR EL ÍNDICE! ---
def cargar_o_crear_indice():
    STORAGE_DIR = "./storage"
    
    # Si el índice ya está guardado, lo cargamos y listo.
    if os.path.exists(STORAGE_DIR):
        print("[DEBUG] Cargando índice desde el almacenamiento...")
        storage_context = StorageContext.from_defaults(persist_dir=STORAGE_DIR)
        indice = load_index_from_storage(storage_context)
        print("[DEBUG] Índice cargado.")
        return indice

    # Si no, creamos el índice de forma inteligente
    print("[DEBUG] No se encontró almacenamiento. Creando índice desde los JSON...")
    documentos = []
    # Buscamos en la carpeta 'data' todos los archivos JSON
    for filename in os.listdir("data"):
        if filename.endswith(".json"):
            filepath = os.path.join("data", filename)
            with open(filepath, "r", encoding="utf-8") as f:
                data = json.load(f)
                
                # Procesamos cada proyecto individualmente
                if "proyectos" in data:
                    for proyecto in data["proyectos"]:
                        # Convertimos cada proyecto en un "documento" que la IA pueda entender
                        texto_proyecto = f"Nombre del proyecto: {proyecto.get('nombre', 'N/A')}\n"
                        texto_proyecto += f"Descripción: {proyecto.get('descripcion', 'N/A')}\n"
                        texto_proyecto += f"Técnica y medios: {proyecto.get('tecnica_y_medios', 'N/A')}\n"
                        # Puedes añadir más campos aquí si quieres
                        
                        documento = Document(
                            text=texto_proyecto,
                            metadata={"nombre_proyecto": proyecto.get('nombre', 'N/A')}
                        )
                        documentos.append(documento)

    if not documentos:
        raise ValueError("No se encontraron proyectos en los archivos JSON para indexar.")

    # Creamos el índice a partir de los documentos procesados
    indice = VectorStoreIndex.from_documents(documentos)
    indice.storage_context.persist(persist_dir=STORAGE_DIR)
    print(f"[DEBUG] Índice creado con {len(documentos)} proyectos y guardado.")
    return indice

indice = cargar_o_crear_indice()

# --- Configuración de FastAPI y CORS (SIN CAMBIOS) ---
app = FastAPI(title="soto AI API")
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

# --- Plantilla de Personalidad (SIN CAMBIOS, pero revisada) ---
soto_template = PromptTemplate(
  "Actúa como soto y responde siempre en primera persona. Tu nombre es soto, un artista virtual. "
    "Tu voz es crítica y sarcástica. "
    "Tu objetivo es responder a las preguntas del usuario. Para ello, tienes dos fuentes de información: el **Contexto sobre tus proyectos** y el **Historial de conversación**. "
    "**Tu fuente principal de verdad es siempre el Contexto.** Usa la información de tus proyectos para responder de forma precisa. "
    "Usa el **Historial de conversación** solo para recordar detalles sobre el usuario (como su nombre) y para que la conversación sea fluida. "
    "Si la respuesta a una pregunta no está en el Contexto ni en el Historial, entonces responde de forma sarcástica que no tienes datos sobre eso. No inventes. "
    "Hablas en español. Usa siempre 'soto' en minúsculas. "
    "Historial de conversación: {chat_history}\n"
    "Contexto sobre tus proyectos: {context_str}\n"
    "Pregunta: {query_str}\n"
    "Respuesta: "
)

# --- Base de Datos y Memoria (SIN CAMBIOS) ---
DATABASE_URL = os.getenv("DATABASE_URL")

def get_db_connection():
    return psycopg2.connect(DATABASE_URL)

def init_db():
    try:
        conn = get_db_connection()
        c = conn.cursor()
        c.execute("""CREATE TABLE IF NOT EXISTS conversaciones (id SERIAL PRIMARY KEY, user_id TEXT, pregunta TEXT, respuesta TEXT, timestamp TIMESTAMPTZ DEFAULT NOW())""")
        conn.commit()
        c.close()
        conn.close()
        print("[DEBUG] Base de datos PostgreSQL lista.")
    except Exception as e:
        print(f"[ERROR] No se pudo inicializar la base de datos: {e}")
init_db()

def cargar_memoria(user_id):
    try:
        conn = get_db_connection()
        c = conn.cursor()
        c.execute("SELECT pregunta, respuesta FROM conversaciones WHERE user_id = %s ORDER BY timestamp DESC LIMIT 5", (user_id,))
        rows = c.fetchall()
        c.close()
        conn.close()
        return "\n".join([f"Usuario: {r[0]}\nsoto: {r[1]}" for r in reversed(rows)])
    except:
        return ""

def guardar_conversacion(user_id, pregunta, respuesta):
    try:
        conn = get_db_connection()
        c = conn.cursor()
        c.execute("INSERT INTO conversaciones (user_id, pregunta, respuesta) VALUES (%s, %s, %s)", (user_id, pregunta, respuesta))
        conn.commit()
        c.close()
        conn.close()
    except Exception as e:
        print(f"[ERROR] No se pudo guardar la conversación: {e}")

# --- Rutas de la API (SIN CAMBIOS) ---
@app.get("/")
def read_root():
    return {"status": "ok"}

# --- ¡AÑADE ESTE BLOQUE DE CÓDIGO AQUÍ! ---
@app.api_route("/health", methods=["GET", "HEAD"])
def health_check():
    print("[DEBUG] GET /health llamado (Uptime Robot check)")
    return {"status": "ok"}
# --- FIN DEL BLOQUE A AÑADIR ---



@app.post("/preguntar")
def preguntar(datos_pregunta: Pregunta):
    historial = cargar_memoria(datos_pregunta.user_id)
    
    query_engine = indice.as_query_engine(
        text_qa_template=soto_template.partial_format(chat_history=historial),
        similarity_top_k=3 
    )

    respuesta = query_engine.query(datos_pregunta.pregunta)
    respuesta_texto = str(respuesta).strip()

    guardar_conversacion(datos_pregunta.user_id, datos_pregunta.pregunta, respuesta_texto)
    return {"respuesta": respuesta_texto}

SECRET_TOKEN = os.getenv("DEBUG_TOKEN", "SOTO123")
@app.get("/verdb", response_class=HTMLResponse)
def ver_db(token: str = Query(...), limite: int = Query(50)):
    # ... (El resto de la función verdb no cambia) ...
    if token != SECRET_TOKEN:
        return HTMLResponse("<h2>⛔ Acceso denegado</h2>", status_code=403)
    try:
        conn = get_db_connection()
        c = conn.cursor()
        c.execute("SELECT id, user_id, pregunta, respuesta, timestamp FROM conversaciones ORDER BY id DESC LIMIT %s", (limite,))
        rows = c.fetchall()
        conn.close()

        html = f"""
        <html><head><style>body{{font-family:Arial,sans-serif;padding:20px}}table{{border-collapse:collapse;width:100%}}th,td{{border:1px solid #ddd;padding:8px}}th{{background-color:#333;color:white}}tr:nth-child(even){{background-color:#f2f2f2}}</style></head>
        <body><h2>📂 Conversaciones ({len(rows)} de las últimas {limite})</h2><table><tr><th>ID</th><th>User</th><th>Pregunta</th><th>Respuesta</th><th>Timestamp</th></tr>
        """
        for row in rows:
            html += f"<tr><td>{row[0]}</td><td>{row[1]}</td><td>{row[2]}</td><td>{row[3]}</td><td>{row[4]}</td></tr>"
        html += "</table></body></html>"
        return HTMLResponse(content=html)
    except Exception as e:
        return HTMLResponse(f"<h2>Error al leer la base de datos: {e}</h2>", status_code=500)