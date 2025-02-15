🤖 RAG Chatbot API with PDF Knowledge Base
A powerful Flask-based API that creates an intelligent chatbot capable of answering questions from PDF documents. Built with LangChain and featuring VectorDatabse storage for chat history.

✨ Features
📚 Upload and process PDF documents

💬 Question-answering using RAG (Retrieval-Augmented Generation)

🚀 RESTful API endpoints

📊 Filtered history retrieval

🔍 Advanced text processing with LangChain

🚀 Quick Start
Installation
Clone the repository git clone https://github.com/Rishiraj-singh7/rag-chatbot.git

Create a virtual environment python -m venv venv

Install dependencies bashCopypip install -r requirements.txt

Start the Flask Server

python app.py
The server will start on http://localhost:5000

📚 API Documentation
Upload PDF Process your PDF documents through the file system: CopyPlace PDF files in the 'pdfFiles' directory Chat Endpoint Send questions about your PDFs:

{ "query": "What does you wanna know from document x?" }

Copyrag-chatbot/
├── app.py # Main application file

├── requirements.txt # Python dependencies

├── pdfFiles/ # Directory for PDF storage

├── vectorDB/ # Vector database storage

└── README.md # Documentation
