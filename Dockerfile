# Usa imagen oficial de PyTorch con transformers ya instalados
FROM pytorch/pytorch:2.0.1-cuda11.8-cudnn8-runtime

# Establece el directorio de trabajo
WORKDIR /app

# Copia archivos de requerimientos y código
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Expone el puerto que usará la app
EXPOSE 8000

# Comando para correr tu app FastAPI con uvicorn
CMD ["uvicorn", "api_soto:app", "--host", "0.0.0.0", "--port", "8000"]
