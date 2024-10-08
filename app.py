import streamlit as st
from dotenv import load_dotenv
import fitz  # PyMuPDF
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from langchain.chat_models import ChatOpenAI 
from htmlTemplates import css, bot_template, user_template
import os
import base64

def get_pdf_text(pdf_list):
    text = ""
    for pdf_path in pdf_list:
        pdf_document = fitz.open(pdf_path)
        for page_num in range(len(pdf_document)):
            page = pdf_document.load_page(page_num)
            text += page.get_text()
    return text

def get_text_chunks(text):
    text_splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len,
    )
    chunks = text_splitter.split_text(text)
    return chunks

def get_vector_store(text_chunks):
    if not text_chunks:
        st.warning("Please upload the textual PDF file - this is PDF files of image")
        return None
    embeddings = OpenAIEmbeddings(openai_api_key=st.secrets["OPEN_AI_APIKEY"])
    vectorstore = FAISS.from_texts(texts=text_chunks, embedding=embeddings)
    return vectorstore

def get_conversation_chain(vector_store):
    llm = ChatOpenAI(openai_api_key=st.secrets["OPEN_AI_APIKEY"])
    memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
    conversation_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=vector_store.as_retriever(),
        memory=memory
    )
    return conversation_chain

def handle_userInput(user_question):
    response = st.session_state.conversation({'question': user_question})
    st.session_state.chat_history = response['chat_history']

    for i, msg in enumerate(st.session_state.chat_history):
        if i % 2 == 0:
            st.write(user_template.replace("{{MSG}}", msg.content), unsafe_allow_html=True)
        else:
            st.write(bot_template.replace("{{MSG}}", msg.content), unsafe_allow_html=True)
    st.session_state.chat_history = []  # Reset chat history after each response

def main():
    load_dotenv()
    st.set_page_config(page_title="Manta Hospital Center AI", page_icon=":hospital:")
    st.write(css, unsafe_allow_html=True)

    # Function to encode image as base64 to set as background
    def get_base64_of_bin_file(bin_file):
        with open(bin_file, 'rb') as f:
            data = f.read()
        return base64.b64encode(data).decode()

    # Encode the background image
    img_base64 = get_base64_of_bin_file('MHC_chat_background.jpg')

    # Set the background image using the encoded base64 string
    st.markdown(
    f"""
    <style>
    .stApp {{
        background: url('data:image/jpeg;base64,{img_base64}') no-repeat center center fixed;
        background-size: cover;
    }}
    </style>
    """,
    unsafe_allow_html=True
)

    if "conversation" not in st.session_state:
        st.session_state.conversation = None
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    if "pdf_text" not in st.session_state:
        st.session_state.pdf_text = ""

    # Pre-train the model with the PDF
    sample_pdf_path = os.path.join(os.getcwd(), "Base_conocimiento_MHC.pdf")
    st.session_state.pdf_files = [sample_pdf_path]

    raw_text = get_pdf_text(st.session_state.pdf_files)
    st.session_state.pdf_text = raw_text
    text_chunks = get_text_chunks(raw_text)
    vector_store = get_vector_store(text_chunks)
    st.session_state.conversation = get_conversation_chain(vector_store)
    col1, col2 = st.columns ([1,2])
    with col1:
        st.image('mhc_logo.png', width=200)
    with col2:
        st.header("Manta Hospital Center - IA")
    st.write("<h5><br>¡Bienvenido al Manta Hospital Center! 🌟 Estamos aquí para ayudarte con cualquier pregunta o inquietud que tengas, ¡en cualquier idioma! Nos enorgullece ser un centro médico multicultural y accesible para todos.</h5>", unsafe_allow_html=True)
    user_question = st.text_input(label="", placeholder="Hola soy Macy, tu asistente del MHC, en que te puedo ayudar?")
    if user_question:
        handle_userInput(user_question)

if __name__ == "__main__":
    main()
