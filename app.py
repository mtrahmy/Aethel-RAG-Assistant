import streamlit as st
import os
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_google_genai import GoogleGenerativeAI

# --- Configuration & Secrets ---

# 1. Use relative path for the PDF (MUST be in the GitHub repo)
PDF_FILE_PATH = "Aethel_Mobility_Systems.pdf" 

# 2. Set API Keys from Streamlit Secrets (Best Practice)
# Streamlit will automatically load secrets from its interface.
if "GEMINI_API_KEY" not in os.environ:
    # If running locally, you might need to load from .env or similar.
    # On Streamlit Cloud, it's automatic.
    os.environ["GEMINI_API_KEY"] = st.secrets["GEMINI_API_KEY"]

# Note: You need to set the LANGSMITH_API_KEY in Streamlit Secrets too 
# if you want tracing to work on the cloud.

st.title("Aethel Mobility Systems Design Assistant 🤖")
st.markdown("Ask questions about the AEMS-WH-100 specification.")

# --- Caching for Performance (CRITICAL) ---

# This function runs only ONCE when the app starts or when the PDF file changes.
@st.cache_resource(show_spinner="Setting up RAG system and generating embeddings...")
def setup_rag_system(file_path):
    # 1. Load Documents
    loader = PyPDFLoader(file_path)
    docs = loader.load()

    # 2. Split Documents
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000, chunk_overlap=200, add_start_index=True
    )
    all_splits = text_splitter.split_documents(docs)
    
    # 3. Create Embeddings Object
    embeddings = GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-001")

    # 4. Create and Populate Vector Store
    # This InMemoryVectorStore instance is saved in Streamlit's cache.
    vector_store = InMemoryVectorStore(embeddings)
    vector_store.add_documents(documents=all_splits)

    # 5. Create Retriever
    retriever = vector_store.as_retriever(search_type="similarity", search_kwargs={"k": 5})
    return retriever

# Function that performs the API call (not cached)
def generate_answer_from_docs(docs, query):
    model = GoogleGenerativeAI(model="models/gemini-2.5-flash", temperature=0.0)
    
    context = "\n\n---\n\n".join(
        f"{d.page_content} [Page {d.metadata.get('page', 'N/A') + 1}]" 
        for d in docs
    )
    
    prompt = (
        "You are the Aethel Mobility Systems Design Specification Assistant, an expert on the document AEMS-WH-100. "
        "Your goal is to answer the user's question conversationally and comprehensively. "
        "You MUST use ONLY the provided CONTEXT to formulate your answer. "
        "For every factual statement, you MUST include a citation at the end of the sentence indicating the source page number using the format [Page X]. "
        "Do not invent facts or use external knowledge. "
        "If the information is not explicitly available in the CONTEXT, politely state that the specification does not contain that detail. "
        "Do not include the raw CONTEXT or any tables from the context in your final answer.\n\n"
        f"CONTEXT:\n{context}\n\nUser Question: {query}\n\nAssistant's Conversational Response:"
    )
    
    try:
        resp = model.generate([prompt])
        return resp.generations[0][0].text.strip()
    except Exception as e:
        return f"I apologize, I encountered an API error: {e}"


# --- Streamlit Application Logic ---

# Initialize the RAG system
retriever = setup_rag_system(PDF_FILE_PATH)

# Initialize chat history for the Streamlit chat element
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat messages from history on app rerun
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Accept user input
if prompt := st.chat_input("Ask about the Aethel Mobility Systems specification..."):
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})
    # Display user message in chat message container
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Searching and generating response..."):
            # 1. Retrieve relevant documents
            docs = retriever.get_relevant_documents(prompt)
            
            if not docs:
                response = "I could not find any relevant sections in the document for that query."
            else:
                # 2. Generate the answer
                response = generate_answer_from_docs(docs, prompt)
            
            st.markdown(response)

    # Add assistant response to chat history
    st.session_state.messages.append({"role": "assistant", "content": response})
