from flask import Flask, request, jsonify
import os
import time
from typing import List, Dict

# Langchain imports 
from langchain.prompts import PromptTemplate
from langchain.memory import ConversationBufferMemory
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings.ollama import OllamaEmbeddings
from langchain_community.llms import Ollama
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain.callbacks.manager import CallbackManager
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains import RetrievalQA

app = Flask(__name__)

# Create  directories
if not os.path.exists('pdfFiles'):
    os.makedirs('pdfFiles')
if not os.path.exists('vectorDB'):
    os.makedirs('vectorDB')

# Add chat history storage
chat_history: List[Dict[str, str]] = []

# Initialize components that were in session state
template = """You are a knowledgeable chatbot, here to help with questions of the user. Your tone should be professional and informative.

Context: {context}
History: {history}

User: {question}
Chatbot:"""

prompt = PromptTemplate(
    input_variables=["history", "context", "question"],
    template=template,
)

memory = ConversationBufferMemory(
    memory_key="history",
    return_messages=True,
    input_key="question",
)

# Initialize vectorstore
vectorstore = Chroma(
    persist_directory='vectorDb',
    embedding_function=OllamaEmbeddings(
        base_url='http://localhost:11434',
        model="llama3"
    )
)

# Initialize LLM
llm = Ollama(
    base_url="http://localhost:11434",
    model="llama3",
    verbose=True,
    num_predict=50,
    callback_manager=CallbackManager([StreamingStdOutCallbackHandler()]),
)

# Initialize QA chain
qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type='stuff',
    retriever=vectorstore.as_retriever(),
    verbose=True,
    chain_type_kwargs={
        "verbose": True,
        "prompt": prompt,
        "memory": memory,
    }
)

def process_pdf(file_path):
    """Process PDF and update vectorstore"""
    loader = PyPDFLoader(file_path)
    data = loader.load()

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1500,
        chunk_overlap=200,
        length_function=len
    )

    all_splits = text_splitter.split_documents(data)

    global vectorstore
    vectorstore = Chroma.from_documents(
        documents=all_splits,
        embedding=OllamaEmbeddings(model="llama3")
    )
    vectorstore.persist()

    global qa_chain
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type='stuff',
        retriever=vectorstore.as_retriever(),
        verbose=True,
        chain_type_kwargs={
            "verbose": True,
            "prompt": prompt,
            "memory": memory,
        }
    )

@app.route('/chat', methods=['POST'])
def chat():
    """
    Endpoint to handle chat requests
    Expects JSON with:
    {
        "query": "user question here"
    }
    """
    try:
        data = request.get_json()
        if not data or 'query' not in data:
            return jsonify({"error": "No query provided"}), 400

        # Get response from QA chain
        response = qa_chain(data['query'])
        
        # Store the conversation in chat history
        chat_history.append({
            "role": "user",
            "message": data['query'],
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
        })
        chat_history.append({
            "role": "assistant",
            "message": response['result'],
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
        })
        
        return jsonify({
            "response": response['result'],
            "status": "success"
        })

    except Exception as e:
        return jsonify({
            "error": str(e),
            "status": "error"
        }), 500

@app.route('/history', methods=['GET'])
def get_history():
    """
    Endpoint to retrieve chat history
    Optional query parameters:
    - limit: number of messages to return (default: all)
    - role: filter by role ('user' or 'assistant')
    """
    try:
        # Get query parameters
        limit = request.args.get('limit', type=int)
        role = request.args.get('role')
        
        # Filter history based on role if specified
        filtered_history = chat_history
        if role:
            filtered_history = [msg for msg in chat_history if msg['role'] == role]
            
        # Apply limit if specified
        if limit:
            filtered_history = filtered_history[-limit:]
            
        return jsonify({
            "history": filtered_history,
            "count": len(filtered_history),
            "status": "success"
        })
        
    except Exception as e:
        return jsonify({
            "error": str(e),
            "status": "error"
        }), 500

if __name__ == '__main__':
    app.run(debug=True, port=5000)