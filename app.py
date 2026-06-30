import os
import streamlit as st
from dotenv import load_dotenv
from langchain_core.messages import HumanMessage, AIMessage
from bot_console import iniciar_chatbot

# Cargar las variables de entorno
load_dotenv()

# ------------- Configuración de la página web --------------------------
st.set_page_config(
    page_title="BotEstudio",
    page_icon="🎓",
    layout="centered"
)

# -------------- Inicialización de componentes -----------------------------
# Usamos cache para que la base de datos y la cadena RAG solo se carguen una vez
@st.cache_resource
def cargar_infraestructura_rag():
    ruta_db = "./chroma_db"
    if not os.path.exists(ruta_db):
        return None
    
    # Llamamos directamente a la función de tu script bot_console.py
    return iniciar_chatbot(ruta_base_datos=ruta_db)

# Instanciamos la cadena RAG orquestada de LangChain
rag_chain = cargar_infraestructura_rag()

# -------------------- Diseño de la UI -----------------------
st.title("Chat Bot Estudio")
st.subheader("Asistente de Estudio Académico")
st.write("Bienvenido a tu Asistente de Estudio.")
st.divider()

# ---------------------- Verificación de Base de Datos -----------------
if rag_chain is None:
    st.error("No se encontró la base de datos en './chroma_db'. Por favor, carga tus documentos antes de iniciar la interfaz.")
    st.stop()

# --------------------- Memoria para el Chat (Streamlit) ----------------------
if "historial_ui" not in st.session_state:
    st.session_state.historial_ui = [
        {"role": "assistant", "content": "¡Hola! ¿Qué información o datos vamos a consultar el día de hoy?"}
    ]

if "historial_langchain" not in st.session_state:
    st.session_state.historial_langchain = []

# --------------------- Mantener la memoria de mensajes en pantalla ---------------------
for mensaje in st.session_state.historial_ui:
    with st.chat_message(mensaje["role"]):
        st.markdown(mensaje["content"])

# --------------------- Interacción en Tiempo Real con el Usuario ---------------------
if pregunta_usuario := st.chat_input("Escribe tu pregunta aquí..."):
    
    with st.chat_message("user"):
        st.markdown(pregunta_usuario)
    st.session_state.historial_ui.append({"role": "user", "content": pregunta_usuario})
    
    with st.chat_message("assistant"):
        with st.spinner("Analizando documentos académicos..."):
            try:
                resultado = rag_chain.invoke({
                    "input": pregunta_usuario,
                    "chat_history": st.session_state.historial_langchain
                })
                
                texto_respuesta = resultado['answer']
                
                # Desplegar la respuesta generada en Streamlit
                st.markdown(texto_respuesta)
                
                # Actualiza la memoria de Streamlit (UI)
                st.session_state.historial_ui.append({"role": "assistant", "content": texto_respuesta})
                
                # Actualizar la memoria nativa de LangChain
                st.session_state.historial_langchain.extend([
                    HumanMessage(content=pregunta_usuario),
                    AIMessage(content=texto_respuesta)
                ])
                
            except Exception as e:
                st.error(f"Ocurrió un error inesperado al procesar la respuesta: {e}")

# ****************---- Ejecucion de la UI python -m streamlit run app.py ----***********************