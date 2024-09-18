import os
import pandas as pd
import streamlit as st
from langchain.vectorstores import FAISS  # Updated import for FAISS
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.messages import AIMessage, HumanMessage
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.schema import Document  # Correct import for the Document class

# Load API keys from Streamlit secrets
groq_api_key = st.secrets["api_keys"]["groq_api_key"]

# Set up the Groq API key
os.environ["GROQ_API_KEY"] = groq_api_key

# Import Groq's LLaMA model
from langchain_groq import ChatGroq

# Initialize the LLaMA model (using the Groq API)
llm = ChatGroq(model="llama3-8b-8192")

# Set the page configuration and background image
st.set_page_config(
    page_title="BotByteAI",
    layout="centered",
    page_icon="🤖",
)

# Set a custom background using HTML/CSS
page_bg_img = '''
<style>
[data-testid="stAppViewContainer"] {
    background: url(".\resources\bg2.jpg");
    background-size: cover;
}
[data-testid="stSidebar"] {
    background: url(".\resources\sidebr.jpg");
}
#MainMenu {visibility: hidden;}
footer {visibility: hidden;}
header {visibility: hidden;}
.embeddedAppMetaInfoBar_container__DxxL1 {visibility: hidden;}
</style>
'''
st.markdown(page_bg_img, unsafe_allow_html=True)

# Sidebar with information
with st.sidebar:
    st.image("https://yourlogo.com/logo.png", use_column_width=True)  # Add your logo
    st.title("BotByteAI")
    st.write("""
    **Ask questions about the Indian Constitution** using this AI-powered assistant.

    This application uses advanced LLaMA models and Groq API to retrieve and answer queries.

    **Designed by**:  
    - Mohd Amaan  
    - Mohd Ayaan  

    All rights reserved © 2024
    """)
    st.write("---")


# Cache the loading of data
@st.cache_data
def load_constitution_data():
    return pd.read_csv('constitution.csv')


@st.cache_data
def load_index_data():
    return pd.read_csv('index.csv', encoding='Windows-1252')


# Preprocess the constitution data using Document objects
@st.cache_data
def preprocess_constitution_data(df):
    documents = []
    for _, row in df.iterrows():
        # Create Document object with 'page_content' and optional 'metadata'
        doc = Document(
            page_content=row['Articles'],  # Use 'Articles' column for the document text
            metadata={}  # Add metadata if needed
        )
        documents.append(doc)
    return documents


# Cache the embedding model and vector store
@st.cache_resource
def load_embeddings():
    from langchain_huggingface import HuggingFaceEmbeddings  # Updated import for HuggingFaceEmbeddings
    # Explicitly pass the model_name for HuggingFace embeddings
    return HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")


# Use _documents and _embeddings to prevent hashing of unhashable types
@st.cache_resource
def create_faiss_vectorstore(_documents, _embeddings):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    splits = text_splitter.split_documents(_documents)
    return FAISS.from_documents(splits, _embeddings)  # Use FAISS to create the vector store


# Load the data once and cache it
constitution_df = load_constitution_data()
index_df = load_index_data()
documents = preprocess_constitution_data(constitution_df)

# Load embeddings and create the vector store (cached)
embeddings = load_embeddings()
vectorstore = create_faiss_vectorstore(documents, embeddings)  # Use FAISS vector store
retriever = vectorstore.as_retriever()

# Set up the retriever and question-answer chain
system_prompt = (
    "You are an assistant that analyses news headings and articles and respond with the laws, and rules in place in the Indian Constitution. "
    "Use the following pieces of retrieved context to answer the question. "
    "If you don't know the answer, say so. Answer concisely."
    "\n\n"
    "{context}"  # This will accept the retrieved context
)

qa_prompt = ChatPromptTemplate.from_messages([("system", system_prompt), ("human", "{input}")])


# Cache the LLaMA model if needed
@st.cache_resource
def get_llm():
    return ChatGroq(model="llama3-8b-8192")


llm = get_llm()

question_answer_chain = create_stuff_documents_chain(llm, qa_prompt)


# Create a history-aware retriever
def create_history_aware_retriever(llm, retriever, chat_history_prompt):
    return RunnableWithMessageHistory(
        retriever,
        retriever_tool=chat_history_prompt,
        input_messages_key="input",
        history_messages_key="chat_history",
        output_messages_key="answer"
    )


# Setup the retrieval chain
rag_chain = create_retrieval_chain(retriever, question_answer_chain)

# Memory storage for chat history
store = {}


def get_session_history(session_id: str) -> BaseChatMessageHistory:
    if session_id not in store:
        store[session_id] = ChatMessageHistory()
    return store[session_id]


# Define conversational RAG chain
conversational_rag_chain = RunnableWithMessageHistory(
    rag_chain,
    get_session_history,
    input_messages_key="input",
    history_messages_key="chat_history",
    output_messages_key="answer",
)

# Streamlit interface
st.title("BotByteAI")
st.subheader("Powered by Groq")
st.write("Ask me questions about the Indian Constitution.")

# Session management
session_id = st.session_state.get("session_id", "default_session")

# Input field for the user's query
user_question = st.text_input("Your question:")

# Initialize chat history in the session
if 'chat_history' not in st.session_state:
    st.session_state['chat_history'] = []

# Submit button
if st.button("Submit"):
    if user_question:
        # Get the RAG model response
        response = conversational_rag_chain.invoke(
            {"input": user_question},
            config={"configurable": {"session_id": session_id}}
        )
        # Store the query and response in chat history
        st.session_state['chat_history'].extend(
            [HumanMessage(content=user_question), AIMessage(content=response["answer"])])
        # Display the response
        st.write("Bot:", response["answer"])

# Display chat history
# for message in st.session_state['chat_history']:
#     if isinstance(message, HumanMessage):
#         st.write(f"User: {message.content}")
#     else:
#         st.write(f"Bot: {message.content}")
