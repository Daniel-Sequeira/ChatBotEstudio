import os
from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_classic.chains import create_retrieval_chain, create_history_aware_retriever
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage, AIMessage

def iniciar_chatbot(ruta_base_datos="./chroma_db"):
    print("[INFO] Iniciando el Asistente de Estudio...")

    #Carga de base de datos vectorial
    embeddings_model = GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-001")
    vector_store = Chroma(
        persist_directory=ruta_base_datos,
        embedding_function=embeddings_model
    )

    #Retriever (Recuperador) fragmentos relevantes
    retriever = vector_store.as_retriever(search_kwargs={"k": 3})

    #Configurar el LLM (gemini-2.5-flash)
    llm = ChatGoogleGenerativeAI(model="gemini-3.5-flash-lite", temperature=0.3)

    #Reformulacion de la pregunta basándose en el historial
    prompt_reformulacion = ChatPromptTemplate.from_messages([
            ("system", "Dada la historia de la conversación y la última pregunta del usuario, "
                    "formula una pregunta independiente que pueda entenderse sin el historial. "
                    "IMPORTANTE: Devuelve ÚNICAMENTE el texto de la pregunta reformulada. "
                    "No uses comillas, no agregues introducciones, no uses markdown y NO la respondas."),
            MessagesPlaceholder("chat_history"),
            ("human", "{input}"),
        ])
    history_aware_retriever = create_history_aware_retriever(llm, retriever, prompt_reformulacion)

    #Prompt principal para la respuesta fianl del asistente.
    prompt_respuesta = ChatPromptTemplate.from_messages([
        ("system", "Eres un asistente de estudio académico. Usa los siguientes fragmentos "
                   "de contexto recuperados para responder a la pregunta del usuario. "
                   "Si la respuesta no está en el contexto, indica amablemente que no "
                   "tienes esa información en los documentos de estudio. Sé claro y estructurado.\n\n"
                   "{context}"),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ])

    # Cadena que une los documentos recuperados con el LLM
    question_answer_chain = create_stuff_documents_chain(llm, prompt_respuesta)

    # Orquestación del flujo RAG Final
    rag_chain = create_retrieval_chain(history_aware_retriever, question_answer_chain)

    return rag_chain

if __name__ == "__main__":
    load_dotenv()
    
    # Inicializar la cadena RAG
    chatbot = iniciar_chatbot()
    
    # Historial de conversación en memoria (Lista de mensajes)
    historial_chat = []

    print("\n" + "="*50)
    print("CHATBOT ASISTENTE DE ESTUDIO ACTIVO")
    print("Escribe 'salir' o 'exit' para terminar la consola.")
    print("="*50 + "\n")
    
    while True:
        consulta_usuario = input("Tú: ")
        
        if consulta_usuario.lower() in ['salir', 'exit']:
            print("Bot: ¡Mucho éxito en tus estudios! Hasta luego.")
            break
            
        if not consulta_usuario.strip():
            continue
            
        print("Bot: Pensando...\r", end="")
        
        try:
            # Ejecutar la cadena pasando la pregunta y el historial actual
            respuesta = chatbot.invoke({
                "input": consulta_usuario,
                "chat_history": historial_chat
            })
            
            texto_respuesta = respuesta['answer']
            print(f"Bot: {texto_respuesta}\n")
            
            # Actualizar la memoria de la conversación
            historial_chat.extend([
                HumanMessage(content=consulta_usuario),
                AIMessage(content=texto_respuesta)
            ])
        
        except Exception as e:
            print(f"\n[ERROR] Hubo un problema al procesar la respuesta: {e}\n")