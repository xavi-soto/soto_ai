# SOTO IA API

API de chat con personalidad artística, visual y creativa, usando RAG + Ollama + FastAPI.

## 🚀 Características
- Responde con estilo único y humor mexicano.
- Memoria persistente en `soto_memoria.json`.
- Soporte para RAG con índice local `soto_index/`.

## 📦 Instalación local
```bash
git clone https://github.com/usuario/soto_ia.git
cd soto_ia
python -m venv soto-env
.\soto-env\Scripts\activate
pip install -r requirements.txt
uvicorn api_soto:app --reload
