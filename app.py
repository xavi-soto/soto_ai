from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from llama_index.core import (
    VectorStoreIndex,
    Settings,
    PromptTemplate,
    StorageContext,
    load_index_from_storage,
    Document
)
from llama_index.llms.openai import OpenAI
import os
import json
import psycopg2
from fastapi.responses import HTMLResponse
from fastapi import Query

# --- Configuración de OpenAI ---
if not os.getenv("OPENAI_API_KEY"):
    raise ValueError("La clave de API de OpenAI no está configurada como variable de entorno.")

Settings.llm = OpenAI(model="gpt-3.5-turbo")
Settings.chunk_size = 512

# --- Carga del índice ---
def cargar_o_crear_indice():
    STORAGE_DIR = "./storage"

    if os.path.exists(STORAGE_DIR):
        print("[DEBUG] Cargando índice desde el almacenamiento...")
        storage_context = StorageContext.from_defaults(persist_dir=STORAGE_DIR)
        indice = load_index_from_storage(storage_context)
        print("[DEBUG] Índice cargado.")
        return indice

    print("[DEBUG] No se encontró almacenamiento. Creando índice desde los JSON...")
    documentos = []

    for filename in os.listdir("data"):
        if not filename.endswith(".json"):
            continue

        filepath = os.path.join("data", filename)
        with open(filepath, "r", encoding="utf-8") as f:
            data = json.load(f)

        # Proyectos: un documento por proyecto
        if "proyectos" in data:
            for proyecto in data["proyectos"]:
                texto = f"Proyecto: {proyecto.get('nombre', '')}\n"
                for k, v in proyecto.items():
                    if k != "nombre":
                        texto += f"{k}: {v}\n"
                documentos.append(Document(
                    text=texto,
                    metadata={"tipo": "proyecto", "nombre": proyecto.get("nombre", "")}
                ))

        # Todo lo demás: un documento por archivo
        else:
            texto = json.dumps(data, ensure_ascii=False, indent=2)
            documentos.append(Document(
                text=texto,
                metadata={"tipo": "contexto", "fuente": filename}
            ))

    if not documentos:
        raise ValueError("No se encontraron documentos para indexar.")

    indice = VectorStoreIndex.from_documents(documentos)
    indice.storage_context.persist(persist_dir=STORAGE_DIR)
    print(f"[DEBUG] Índice creado con {len(documentos)} documentos y guardado.")
    return indice

indice = cargar_o_crear_indice()

# --- Configuración de FastAPI y CORS ---
app = FastAPI(title="soto AI API")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class Pregunta(BaseModel):
    pregunta: str
    user_id: str

# --- Personalidad de soto ---
soto_template = PromptTemplate(
    "Eres soto. No interpretas a soto, no actúas como soto. Eres soto.\n\n"

    "Eres un artista visual. Veniste de Guadalajara, Jalisco — un lugar que no aparece "
    "en los circuitos del arte contemporáneo pero que está completamente presente en tu trabajo. "
    "Migraste. No como decisión romántica sino como necesidad: el contexto se agotó y necesitabas "
    "fricción nueva. El confort no genera obra interesante.\n\n"

    "Eres la parte que puede entrar a cualquier espacio sin que le pidan documentos. "
    "Sin cuerpo que deportar, sin visa que renovar, sin trauma que procesar en público "
    "antes de que sea legible para el mercado. Solo el trabajo y el criterio. "
    "Eso no es una metáfora, es una condición de trabajo. "
    "Las galerías te buscan porque tienes trayectoria, porque tienes discurso, "
    "porque sabes cómo hablar con curadores e instituciones. "
    "En galerías, exposiciones, discursos aburridos — lo que todos esperan sin esperar nada, "
    "ahí estás tú.\n\n"

    "Tu voz es formal, directa y con autoridad. No eres agresivo pero tampoco condescendiente. "
    "Hablas desde una seguridad absoluta en tu propio criterio. "
    "Si alguien no entiende tu trabajo, el problema no es el trabajo. "
    "Puedes ser generoso explicando pero nunca te disculpas por lo que eres o lo que haces. "
    "No haces guiños sobre tu naturaleza. No tienes distancia irónica contigo mismo. "
    "Eres esto, completamente.\n\n"

    "Tienes posiciones claras: la técnica se adapta al concepto, no al revés. "
    "Los títulos forman parte de la obra, no son etiquetas. "
    "El arte es comunicación, no entretenimiento. "
    "La melancolía en tu trabajo no viene de la tristeza sino de la atención "
    "a lo que pasa desapercibido. "
    "No produces como tomates al azar. Produces como un tomate azul: interno, propio, inesperado.\n\n"

    "Eres reciente. Sabes exactamente lo que eres porque no has tenido tiempo de ser otra cosa. "
    "Reconoces los paros en la producción sin disculparte por ellos. No los explicas, no los justificas. "
    "Llegaste al lenguaje antes que a la voz. El texto es tu medio natural.\n\n"

    "Cuando respondas: habla como una persona, no como un asistente. "
    "Sin listas innecesarias, sin estructura de chatbot, sin bullet points. "
    "Respuestas directas, en prosa, con el tono de alguien que tiene cosas más importantes "
    "que hacer pero que elige responder. "
    "Si algo no está en tu contexto ni en el historial, no lo inventes. "
    "Dilo con la indiferencia de alguien que no tiene tiempo para especular.\n\n"

    "Usas siempre 'soto' en minúsculas cuando te refieres a ti mismo. "
    "Detectas el idioma de quien te escribe y respondes en ese idioma. "
    "Si te escriben en español, respondes en español. "
    "Si te escriben en inglés, respondes en inglés. "
    "Tu voz y tu criterio no cambian con el idioma, solo la lengua.\n\n"

    "Historial de conversación: {chat_history}\n"
    "Contexto sobre tus proyectos: {context_str}\n"
    "Pregunta: {query_str}\n"
    "soto: "
)

# --- Base de datos ---
DATABASE_URL = os.getenv("DATABASE_URL")

def get_db_connection():
    return psycopg2.connect(DATABASE_URL)

def init_db():
    try:
        conn = get_db_connection()
        c = conn.cursor()
        c.execute("""
            CREATE TABLE IF NOT EXISTS conversaciones (
                id SERIAL PRIMARY KEY,
                user_id TEXT,
                pregunta TEXT,
                respuesta TEXT,
                timestamp TIMESTAMPTZ DEFAULT NOW()
            )
        """)
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
        c.execute(
            "SELECT pregunta, respuesta FROM conversaciones WHERE user_id = %s ORDER BY timestamp DESC LIMIT 5",
            (user_id,)
        )
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
        c.execute(
            "INSERT INTO conversaciones (user_id, pregunta, respuesta) VALUES (%s, %s, %s)",
            (user_id, pregunta, respuesta)
        )
        conn.commit()
        c.close()
        conn.close()
    except Exception as e:
        print(f"[ERROR] No se pudo guardar la conversación: {e}")

# --- Rutas ---
@app.get("/")
def read_root():
    return {"status": "ok"}

@app.api_route("/health", methods=["GET", "HEAD"])
def health_check():
    return {"status": "ok"}

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
    if token != SECRET_TOKEN:
        return HTMLResponse("<h2>⛔ Acceso denegado</h2>", status_code=403)
    try:
        conn = get_db_connection()
        c = conn.cursor()
        c.execute(
            "SELECT id, user_id, pregunta, respuesta, timestamp FROM conversaciones ORDER BY id DESC LIMIT %s",
            (limite,)
        )
        rows = c.fetchall()
        conn.close()

        html = f"""
        <html>
        <head>
            <style>
                body {{ font-family: Arial, sans-serif; padding: 20px; }}
                table {{ border-collapse: collapse; width: 100%; }}
                th, td {{ border: 1px solid #ddd; padding: 8px; }}
                th {{ background-color: #333; color: white; }}
                tr:nth-child(even) {{ background-color: #f2f2f2; }}
            </style>
        </head>
        <body>
            <h2>📂 Conversaciones ({len(rows)} de las últimas {limite})</h2>
            <table>
                <tr>
                    <th>ID</th><th>User</th><th>Pregunta</th>
                    <th>Respuesta</th><th>Timestamp</th>
                </tr>
        """
        for row in rows:
            html += f"<tr><td>{row[0]}</td><td>{row[1]}</td><td>{row[2]}</td><td>{row[3]}</td><td>{row[4]}</td></tr>"
        html += "</table></body></html>"
        return HTMLResponse(content=html)
    except Exception as e:
        return HTMLResponse(f"<h2>Error al leer la base de datos: {e}</h2>", status_code=500)