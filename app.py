import os
import shutil
import streamlit as st
from dotenv import load_dotenv
from langchain_core.messages import HumanMessage, AIMessage
from bot_console import iniciar_chatbot
from ingest import ejecutar_pipeline_ingesta

def limpiar_datos_inicio():

    carpetas = [
        "./chroma_db",
        "./documentos"
    ]

    for carpeta in carpetas:
        if os.path.exists(carpeta):
            shutil.rmtree(carpeta)

# Cargar las variables de entorno
load_dotenv()

if "iniciado" not in st.session_state:

    limpiar_datos_inicio()

    st.session_state.iniciado = True

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
    
    # Llamamos directamente a la función del script bot_console.py
    return iniciar_chatbot(ruta_base_datos=ruta_db)

# Instanciamos la cadena RAG orquestada de LangChain
rag_chain = cargar_infraestructura_rag()
# -------------------- Sidebar --------------------

with st.sidebar:

    st.title("📚 Documentos")

    archivo_subido = st.file_uploader(
    "📎 Adjuntar documento",
    type=["pdf"]
)

    if archivo_subido:

        st.success(f"📄 {archivo_subido.name}")

        carpeta_documentos = "./documentos"

        os.makedirs(carpeta_documentos, exist_ok=True)

        ruta_pdf = os.path.join(
            carpeta_documentos,
            archivo_subido.name
        )

        with open(ruta_pdf, "wb") as archivo:
            archivo.write(archivo_subido.getbuffer())

        if st.button("📚 Procesar PDF", use_container_width=True):

            with st.spinner("Procesando documento..."):

                try:

                    ejecutar_pipeline_ingesta(
                        ruta_pdf,
                        "./chroma_db"
                    )

                    st.success(
                        "Documento listo."
                    )
                    # Reiniciar el historial
                    st.session_state.historial_ui = [
                        {
                            "role": "assistant",
                            "content": "👋 Nuevo documento cargado correctamente.\n\n"
                                    "Ahora puedes realizar preguntas sobre este PDF."
                        }
                    ]

                    st.session_state.historial_langchain = []
                    st.cache_resource.clear()

                    st.rerun()

                except Exception as e:

                    st.error(e)
    st.divider()

    st.subheader("Estado")

    if rag_chain is None:

        st.warning(
        "📄 Aún no hay documentos cargados. "
        "Sube un archivo PDF para comenzar a conversar."
    )

    else:

        st.success("Documento listo para consultar.")

        if archivo_subido:
            st.info(f"📄 {archivo_subido.name}")

# -------------------- Diseño de la UI -----------------------
st.title("Chat Bot Estudio")
st.subheader("Asistente de Estudio Académico")
st.write("Bienvenido a tu Asistente de Estudio.")
st.divider()


# ---------------------- Verificación de Base de Datos -----------------
if rag_chain is None:
    st.warning(
    "📄 Aún no hay documentos cargados. "
    "Sube un archivo PDF para comenzar a conversar."
)

# --------------------- Memoria para el Chat (Streamlit) ----------------------
if "historial_ui" not in st.session_state:
    st.session_state.historial_ui = [
        {
            "role": "assistant",
            "content": "👋 Hola, soy BotEstudio AI.\n\n"
                       "Para comenzar, sube un archivo PDF con el botón de carga "
                       "y luego podrás realizar preguntas sobre su contenido."
        }
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

    st.session_state.historial_ui.append(
        {"role": "user", "content": pregunta_usuario}
    )


    with st.chat_message("assistant"):

        if rag_chain is None:

            texto_respuesta = (
                "📄 Aún no hay documentos cargados.\n\n"
                "Por favor, sube un archivo PDF antes de realizar consultas."
            )

            st.markdown(texto_respuesta)

            st.session_state.historial_ui.append(
                {
                    "role": "assistant",
                    "content": texto_respuesta
                }
            )

        else:

            with st.spinner("Analizando documentos académicos..."):

                try:
                    resultado = rag_chain.invoke({
                        "input": pregunta_usuario.strip(),
                        "chat_history":[] #enviar contexto vacio por limites de modelo gratuito, la pregunta se toma como contexto
                    })

                    texto_respuesta = resultado['answer']

                    st.markdown(texto_respuesta)

                    st.session_state.historial_ui.append(
                        {
                            "role": "assistant",
                            "content": texto_respuesta
                        }
                    )

                    st.session_state.historial_langchain.extend([
                        HumanMessage(content=pregunta_usuario),
                        AIMessage(content=texto_respuesta)
                    ])

                except Exception as e:
                    st.error(
                        f"Ocurrió un error inesperado al procesar la respuesta: {e}"
                    )
            


# ****************---- Ejecucion de la UI python -m streamlit run app.py ----***********************