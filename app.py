import os
import faiss
import numpy as np
from flask import Flask, request, jsonify
from PyPDF2 import PdfReader
from llama_index.core import (
    Document,
    VectorStoreIndex,
    Settings,
    StorageContext,
    PromptTemplate,
    load_index_from_storage,
)
from llama_index.vector_stores.faiss import FaissVectorStore
from llama_index.llms.ollama import Ollama
from llama_index.core.llms import ChatMessage, MessageRole
from llama_index.core.prompts import ChatPromptTemplate
from llama_index.embeddings.langchain import LangchainEmbedding
from langchain.embeddings import HuggingFaceEmbeddings

# --- Configuration ---
VECTOR_STORE_DIR = "vector_store"
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
OLLAMA_MODEL = "tinyllama:1.1b" # Make sure you have this model pulled in Ollama
OLLAMA_BASE_URL = "http://localhost:11434"
CHUNK_SIZE = 1024
CHUNK_OVERLAP = 200

# --- Flask App Initialization ---
app = Flask(__name__)

# --- Global Variables & Setup ---
# Ensure vector store directory exists
os.makedirs(VECTOR_STORE_DIR, exist_ok=True)

# Initialize the embedding model
embed_model = LangchainEmbedding(HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL))

# Initialize the LLM
llm = Ollama(model=OLLAMA_MODEL, base_url=OLLAMA_BASE_URL, request_timeout=120.0)

# # Create the service context for LlamaIndex
# service_context = ServiceContext.from_defaults(
#     llm=llm,
#     embed_model=embed_model,
#     chunk_size=CHUNK_SIZE,
#     chunk_overlap=CHUNK_OVERLAP
# )
Settings.llm=llm
Settings.embed_model=embed_model
Settings.chunk_size=CHUNK_SIZE
Settings.chunk_overlap=CHUNK_OVERLAP

# Load or create the index using LlamaIndex's persistence mechanism
try:
    # Attempt to load the index from the persistent storage
    storage_context = StorageContext.from_defaults(persist_dir=VECTOR_STORE_DIR)
    index = load_index_from_storage(storage_context)
    print("Successfully loaded existing index from vector_store/. a")
except Exception:
    print("Could not load existing index. Creating a new one.")
    # If loading fails, it means the index doesn't exist yet. Create it.
    d = 384  # Dimensions of the embedding model all-MiniLM-L6-v2
    faiss_index = faiss.IndexFlatL2(d)
    vector_store = FaissVectorStore(faiss_index=faiss_index)
    storage_context = StorageContext.from_defaults(vector_store=vector_store)
    
    # Create an empty index with the specified contexts
    index = VectorStoreIndex.from_documents(
        [], storage_context=storage_context
    )

# --- API Endpoints ---

@app.route('/index', methods=['POST'])
def index_pdf():
    """
    API endpoint to index a PDF document.
    Expects a multipart/form-data request with a 'file' field containing the PDF.
    """
    global index

    if 'file' not in request.files:
        return jsonify({"error": "No file part in the request"}), 400
    file = request.files['file']
    if file.filename == '':
        return jsonify({"error": "No selected file"}), 400

    if file and file.filename.endswith('.pdf'):
        try:
            # 1. Extract Text from PDF
            pdf_reader = PdfReader(file.stream)
            text = ""
            for page in pdf_reader.pages:
                page_text = page.extract_text()
                if page_text:
                    text += page_text
            
            if not text:
                 return jsonify({"error": "Could not extract text from the PDF."}), 400

            # 2. Create LlamaIndex Document
            # LlamaIndex will handle chunking internally based on the ServiceContext
            document = Document(text=text, metadata={"filename": file.filename})

            # 3. Insert document into the index
            index.insert(document)
            
            # 4. Persist the entire index to disk
            # This single command handles saving the docstore, index store, and the Faiss vector store.
            index.storage_context.persist(persist_dir=VECTOR_STORE_DIR)

            return jsonify({"success": f"Successfully indexed '{file.filename}'"}), 200

        except Exception as e:
            return jsonify({"error": f"An error occurred: {str(e)}"}), 500
    else:
        return jsonify({"error": "Invalid file type. Please upload a PDF."}), 400


@app.route('/query', methods=['POST'])
def query_index():
    """
    API endpoint to query the indexed documents.
    Expects a JSON request with a 'question' field.
    """
    global index
    
    data = request.get_json()
    if not data or 'question' not in data:
        return jsonify({"error": "Request must be JSON with a 'question' field"}), 400

    question = data['question']

    try:
        customized_prompt = ''' You are an intelligent Chatbot named as BOT, You are designed to provide answers for students based on the context given by the user. Analyze and understand question and context properly.
                                    - You are build for Q&A for students on there respective subject documents.
                                    - Dont modify the context, just use it as it is.
                                    - Answer should be based on the context given by user. Donot rephrase it.
                                    - If you don't know the answer, just say that you don't know.
                                    - If question is irrelavent to the content, Just reply with "Sorry, I don't have an answer for this."
                                    - If user greets with Hi, Hello, Good Morning, Good Evening etc., then respond back with a greeting.
                                    - Answer should be precise and to the point.
                                    - Handle mis-spelled words properly.
                                    - Handle abbreviations and shortcuts, and respond accordingly.'''

            # Text QA Prompt
        chat_text_qa_msgs = [
            ChatMessage(role=MessageRole.SYSTEM,
                        content=(customized_prompt),
                        ),
            ChatMessage(role=MessageRole.USER,
                        content=('''Context information is below.
                                    ---------------------
                                    {context_str}
                                    ---------------------
                                    Given the context information and not prior knowledge, answer the query. Answer should be from above given context only, not generalized answers.
                                    Query: {query_str}
                                    Answer:'''
                            ),
                        ),
        ]

        text_qa_template = ChatPromptTemplate(chat_text_qa_msgs)

 

        # Build Query Engine
        query_engine = index.as_query_engine(
            similarity_top_k=3, 
            text_qa_template=text_qa_template
        )

        # Perform Query
        response = query_engine.query(question)

        # Format Response
        answer = str(response)
        source_nodes = [
            {
                "text": node.get_content(),
                "score": float(node.get_score()),
                "filename": node.metadata.get("filename", "N/A")
            } for node in response.source_nodes
        ]

        return jsonify({
            "answer": answer,
            "source_nodes": source_nodes
        }), 200

    except Exception as e:
        return jsonify({"error": f"An error occurred during query: {str(e)}"}), 500

# --- Main Execution ---
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
