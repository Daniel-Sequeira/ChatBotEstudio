import time
import os
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_chroma import Chroma

def ejecutar_pipeline_ingesta(ruta_pdf, ruta_base_datos="./chroma_db"):
    load_dotenv()
    # Cargar el documento PDF
    print(f"[INFO] Cargando el archivo académico: {ruta_pdf}")
    if not os.path.exists(ruta_pdf):
        raise FileNotFoundError(f"No se encontró el archivo '{ruta_pdf}'. Asegúrate de colocarlo en la carpeta.")
        
    loader = PyPDFLoader(ruta_pdf)
    paginas = loader.load()
    print(f"[ÉXITO] Páginas leídas: {len(paginas)}")

    # Fragmentación Semántica (Chunking).
    print("[INFO] Fragmentando el texto en bloques semánticos...")
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = text_splitter.split_documents(paginas)
    print(f"[ÉXITO] Cantidad de fragmentos (chunks) generados: {len(chunks)}")

   #Modelado de Embeddings e Indexación en ChromaDB
    print("[INFO] Conectando con Google GenAI para generar Embeddings matemáticos...")
    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        raise ValueError("No se detectó GOOGLE_API_KEY. Revisa tu archivo .env")
        
    embeddings_model = GoogleGenerativeAIEmbeddings(
        model="models/gemini-embedding-001",
        google_api_key=api_key
    )
    print(f"[INFO] Guardando vectores en la base de datos local: {ruta_base_datos}")
    # Inicializamos ChromaDB completamente VACÍA para evitar que LangChain agote la cuota de API gratuita (100 peticiones por minuto).
    vector_store = Chroma(
        persist_directory=ruta_base_datos,
        embedding_function=embeddings_model
    )
    # Definimos el tamaño del bloque seguro para la API gratuita de Google GenAI
    tamano_lote = 15
    print(f"[INFO] Indexando {len(chunks)} fragmentos en bloques controlados de {tamano_lote}...")
    
    # El bucle empieza de manera estricta en 0 para procesar todo el documento por iteraciones.
    for i in range(0, len(chunks), tamano_lote):
        lote_actual = chunks[i:i + tamano_lote]
        
        # Agregamos de manera segura el bloque actual de 15 fragmentos a la base de datos
        vector_store.add_documents(lote_actual)
        print(f" -> [OK] Indexados los fragmentos del {i} al {min(i + tamano_lote, len(chunks))}")

    # Control de Tasa (Rate Limiting) para evitar el error 429 (Resource Exhausted)
        # Si aún no hemos terminado de subir todo el libro, pausamos para refrescar la cuota por minuto
        if i + tamano_lote < len(chunks):
            print(" ⏳ Pausando 15 segundos para recargar la cuota de la API de Google...")
            time.sleep(15)
            
    print("[ÉXITO] Pipeline completado. Datos indexados y persistidos.")
    return vector_store
   
