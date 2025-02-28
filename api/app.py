from flask import Flask, render_template, jsonify, request, session
from flask_sqlalchemy import SQLAlchemy
from flask_migrate import Migrate
from googletrans import Translator
from werkzeug.utils import secure_filename
import os
import time
import tempfile
import uuid
import chromadb
import numpy as np
import google.generativeai as genai
#from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv 

# RAG imports
from langchain_core.globals import set_verbose, set_debug
from langchain_community.vectorstores import Chroma
from langchain_community.chat_models import ChatOllama
#from langchain_community.embeddings import FastEmbedEmbeddings
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.schema import Document
from langchain.schema.output_parser import StrOutputParser
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema.runnable import RunnablePassthrough
from langchain_community.vectorstores.utils import filter_complex_metadata
from langchain_core.prompts import ChatPromptTemplate

# Enable debugging
set_debug(True)
set_verbose(True)

# Load env variables
load_dotenv()

# Selecting the model
selected_model = "gemini"

class Mentor:
    def __init__(self, llm_model="qwen2.5", selected_model="ollama"):
        self.selected_model = selected_model
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=512, chunk_overlap=100
        )

        # Load API Key
        self.gemini_api_key = os.getenv("GEMINI_API_KEY")

        # Prompt template for answering questions
        self.prompt = ChatPromptTemplate.from_template(
            """
            You are an experienced and knowledgeable mentor, guiding users by answering their questions strictly based on the provided context. Your responses should be clear, accurate, and helpful while maintaining a friendly and supportive tone.

            Instructions:
                Use only the provided context to answer questions. If the answer is not in the context, say, "I don't have enough information to answer that."
                Provide structured and detailed explanations when necessary.
                Keep responses concise and relevant, avoiding unnecessary information.
                If the question is ambiguous, ask for clarification instead of making assumptions.

            Context: {context}

            Question: {question}
            """
        )

        self.vector_store = None
        self.retriever = None
        self.chain = None
        self.init_vector_store()

        # Initialize models
        if selected_model == "ollama":
            self.model = ChatOllama(model=llm_model)
        elif selected_model == "gemini":
            if not self.gemini_api_key:
                raise ValueError("Missing Gemini API key. Please set GEMINI_API_KEY in your .env file.")
            genai.configure(api_key=self.gemini_api_key)
            self.model = genai.GenerativeModel("gemini-2.0-flash")



    def init_vector_store(self):
        """Initialize the vector store with e5-base embeddings"""
        try:
            # Load e5-base embeddings
            self.embedding_function = HuggingFaceEmbeddings(model_name="intfloat/e5-base")

            self.vector_store = Chroma(
                persist_directory="chroma_db",
                embedding_function=self.embedding_function
            )
            self.retriever = self.vector_store.as_retriever(
                search_type="similarity_score_threshold",
                search_kwargs={"k": 10, "score_threshold": 0.0},
            )

            self.chain = (
                {"context": self.retriever, "question": RunnablePassthrough()}
                | self.prompt
                | self.model
                | StrOutputParser()
            )
        except Exception as e:
            print(f"Error initializing vector store: {e}")
            self.vector_store = Chroma(
                collection_name="documents",
                embedding_function=self.embedding_function
            )
            self.vector_store.persist()

    def ingest(self, file_path):
        """Process and ingest a document into the vector store"""
        try:
            # Load the document
            loader = PyPDFLoader(file_path)
            documents = loader.load()
            
            # Split the documents into chunks
            splits = self.text_splitter.split_documents(documents)
            
            # Add documents to vector store
            if not self.vector_store:
                self.init_vector_store()
                
            self.vector_store.add_documents(splits)
            self.vector_store.persist()
            
            return len(splits)
        except Exception as e:
            print(f"Error ingesting document: {e}")
            return 0

    def ask(self, query):
        """Query the knowledge base using the selected model"""
        if not self.vector_store:
            return "No documents have been ingested yet. Please upload documents first."
        
        try:
            if not self.chain:
                self.init_vector_store()

            retrieved_docs = self.retriever.invoke(query)
            context = "\n\n".join([doc.page_content for doc in retrieved_docs])

            if self.selected_model == "ollama":
                return self.chain.invoke(query)

            elif self.selected_model == "gemini":
                return self.generate_answer_with_gemini(query, context)

            return "Invalid model selection."
        except Exception as e:
            print(f"Error processing query: {e}")
            return f"An error occurred: {str(e)}"

    def generate_answer_with_gemini(self, query, context):
        """Generate an answer using Gemini with NCERT context."""
        prompt = f"""
        Use the retrieved NCERT content to answer the question.

        Question: {query}

        Retrieved Context:
        {context}
        """

        response = self.model.generate_content(prompt)
        return response.text

    def clear(self):
        """Clear the vector store"""
        if self.vector_store:
            try:
                self.vector_store.delete_collection()
                self.init_vector_store()
            except Exception as e:
                print(f"Error clearing vector store: {e}")

    def load_embeddings(self, path):
        """Load pre-generated embeddings"""
        try:
            embeddings = np.load(path, allow_pickle=True)
            return len(embeddings)
        except Exception as e:
            print(f"Error loading embeddings: {e}")
            return 0



class ChatbotService:
    @staticmethod
    def res(question):
        """Process a user question and return a response"""
        try:
            # For demonstration purposes - implement actual logic
            return f"Response to: {question}"
        except Exception as e:
            print(f"Error processing chatbot question: {e}")
            return "Sorry, I'm having trouble processing your question."


# Initialize Flask app
app = Flask(__name__)
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///default.db'
app.config['SQLALCHEMY_BINDS'] = {
    'db1': 'sqlite:///database1.db',
    'db2': 'sqlite:///database2.db'
}
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
app.secret_key = "your_secret_key_here"  

# Configure file uploads
UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max upload

# Initialize database
db = SQLAlchemy()
db.init_app(app)
migrate = Migrate(app, db)
translate = Translator()

# Store active chat instances
chat_instances = {}

# Define database models
class Chatbot(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    user_message = db.Column(db.String(500), nullable=False)
    translated_message = db.Column(db.String(500), nullable=True)
    response = db.Column(db.String(500), nullable=False)
    
    @staticmethod
    def res(question):
        return ChatbotService.res(question)


class User(db.Model):
    __bind_key__ = 'db1'
    __tablename__ = 'User'
    id = db.Column(db.Integer, primary_key=True)
    first_name = db.Column(db.String(50), nullable=False)
    last_name = db.Column(db.String(50), nullable=False)
    age = db.Column(db.String(4), nullable=False)

    def to_dict(self):
        return {
            'id': self.id,
            'first': self.first_name,
            'last': self.last_name,
            'age': self.age
        }


# # Load Sentence Transformer for embedding queries
# encoder = SentenceTransformer("intfloat/e5-base")

# # Initialize ChromaDB client
# chroma_client = chromadb.PersistentClient(path="chroma_db")
# collection = chroma_client.get_collection("ncert_docs")

# # Configure Gemini API
# genai.configure(api_key=os.getenv("GEMINI_API_KEY"))  # Replace with your actual API key

# # Initialize ChromaDB client
# chroma_client = chromadb.PersistentClient(path="chroma_db")
# collection_name = "ncert_docs"

# Initialize Chroma with the same DB as init_vector_store
embedding_function = HuggingFaceEmbeddings(model_name="intfloat/e5-base")
vector_store = Chroma(persist_directory="chroma_db", embedding_function=embedding_function)


# Helper functions
def ensure_session():
    """Ensure the session is properly initialized"""
    if 'messages' not in session:
        session['messages'] = []
    
    if 'session_id' not in session:
        session['session_id'] = str(uuid.uuid4())
    
    session_id = session['session_id']
    if session_id not in chat_instances:
        chat_instances[session_id] = Mentor(selected_model=selected_model)
    
    return session_id


def process_npy_files(embeddings_path, chunks_path):
    """Load embeddings and text chunks, then insert them into LangChain's ChromaDB"""
    try:
        print(f"Loading embeddings from: {embeddings_path}")
        print(f"Loading chunks from: {chunks_path}")

        embeddings = np.load(embeddings_path, allow_pickle=True)
        chunks = np.load(chunks_path, allow_pickle=True)

        print(f"Loaded {len(embeddings)} embeddings")
        print(f"Loaded {len(chunks)} chunks")

        if len(embeddings) != len(chunks):
            print("Error: Mismatch between embeddings and chunks count.")
            return False

        # Convert chunks into LangChain's Document format
        documents = [
            Document(
                page_content=chunk["text"],  # Extract text
                metadata={
                    "source": str(chunk.get("source", "unknown")),
                    "chunk_id": int(chunk.get("chunk_id", i))  # Ensure integer
                }
            )
            for i, chunk in enumerate(chunks)
        ]

        print("Adding to ChromaDB using LangChain...")
        vector_store.add_documents(documents)

        print("Successfully stored in ChromaDB!")
        return True
    except Exception as e:
        print(f"Error processing NPY files: {e}")
        return False





# def retrieve_context(query, top_k=5):
#     """Retrieve top-k most relevant chunks from ChromaDB."""
#     query_embedding = encoder.encode(query).tolist()
#     results = collection.query(
#         query_embeddings=[query_embedding],
#         n_results=top_k
#     )

#     retrieved_docs = results["documents"][0] if "documents" in results else []
#     return "\n\n".join(retrieved_docs) if retrieved_docs else "No relevant NCERT content found."


# def generate_answer_with_gemini(query):
#     """Generate an answer using Gemini with NCERT context."""
#     context = retrieve_context(query)
    
#     prompt = f"""
#     You are an AI assistant trained on NCERT textbooks. Answer the following question using the retrieved NCERT content.

#     Question:
#     {query}

#     Retrieved Context:
#     {context}

#     Answer:
#     """

#     model = genai.GenerativeModel("gemini-2.0-flash")
#     response = model.generate_content(prompt)
#     return response.text


# Routes
@app.route('/')
def index():
    """Render the main application page"""
    session_id = ensure_session()
    return render_template('index.html')


@app.route('/admin', methods=['GET'])
def admin_view():
    return render_template('admin_page.html')


@app.route('/upload', methods=['POST'])
def upload_file():
    """Handle document uploads"""
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400
    
    files = request.files.getlist('file')
    if not files or files[0].filename == '':
        return jsonify({'error': 'No file selected'}), 400
    
    session_id = ensure_session()
    assistant = chat_instances[session_id]
    
    responses = []
    for file in files:
        filename = secure_filename(file.filename)

        with tempfile.NamedTemporaryFile(delete=False) as tf:
            file.save(tf.name)
            file_path = tf.name
        
        start_time = time.time()
        chunks_processed = assistant.ingest(file_path)
        processing_time = time.time() - start_time
        
        try:
            os.remove(file_path)
        except Exception as e:
            print(f"Error removing temporary file: {e}")
        
        response = f"Ingested {filename} in {processing_time:.2f} seconds ({chunks_processed} chunks)"
        responses.append(response)
        
        # Update session messages
        session['messages'].append({'content': response, 'is_user': False})
    
    return jsonify({'messages': responses}), 200


@app.route('/ask', methods=['POST'])
def ask_question():
    """Process user questions"""
    data = request.json
    user_text = data.get('message', '').strip()

    if not user_text:
        return jsonify({'error': 'Empty message'}), 400

    session_id = ensure_session()
    assistant = chat_instances[session_id]

    agent_text = assistant.ask(user_text)

    # Update session messages
    session['messages'].append({'content': user_text, 'is_user': True})
    session['messages'].append({'content': agent_text, 'is_user': False})

    return jsonify({
        'user_message': user_text,
        'response': agent_text
    }), 200




@app.route('/clear', methods=['POST'])
def clear_conversation():
    """Clear the conversation history"""
    session_id = session.get('session_id')
    if session_id and session_id in chat_instances:
        chat_instances[session_id].clear()
    
    session['messages'] = []
    
    return jsonify({'status': 'success'}), 200


@app.route('/users', methods=['GET'])
def user():
    """Get user information"""
    user_id = request.args.get('Id')
    if user_id:
        user = User.query.get(user_id)
        if user:
            return jsonify(user.to_dict())
        return jsonify({'message': 'Unable to detect the user!'}), 404
    
    user_data = User.query.all()
    user_list = [user.to_dict() for user in user_data]
    return jsonify(user_list)


@app.route('/update', methods=['POST'])
def create():
    """Create a new user"""
    try:
        data = request.get_json()
        if 'id' in data:
            return jsonify({'message': 'User already registered'}), 400
        
        new_user = User(
            first_name=data['first_name'], 
            last_name=data['last_name'], 
            age=data['age']
        )
        db.session.add(new_user)
        db.session.commit()
        return jsonify({'message': 'Success', 'id': new_user.id})
    except Exception as e:
        db.session.rollback()
        return jsonify({'message': f'Error: {str(e)}'}), 500


@app.route('/chatbot', methods=['POST'])
def chatbot():
    """Process chatbot requests"""
    data = request.get_json()
    if not data or 'question' not in data:
        return jsonify({'error': 'Missing question'}), 400
    
    response = Chatbot.res(data['question'])
    return jsonify({'answer': response})


@app.route("/upload_npy", methods=["POST"])
def upload_npy():
    """Handle NPY file uploads"""
    if "embeddings" not in request.files or "chunks" not in request.files:
        return jsonify({"error": "Both 'embeddings' and 'chunks' files are required"}), 400

    embeddings_file = request.files["embeddings"]
    chunks_file = request.files["chunks"]

    # Secure filenames
    embeddings_filename = secure_filename(embeddings_file.filename)
    chunks_filename = secure_filename(chunks_file.filename)

    # Save uploaded files
    embeddings_path = os.path.join(app.config["UPLOAD_FOLDER"], embeddings_filename)
    chunks_path = os.path.join(app.config["UPLOAD_FOLDER"], chunks_filename)

    embeddings_file.save(embeddings_path)
    chunks_file.save(chunks_path)

    # Process and insert into ChromaDB
    success = process_npy_files(embeddings_path, chunks_path)
    
    if success:
        return jsonify({"message": "Files uploaded and processed successfully"}), 200
    else:
        return jsonify({"error": "Error processing uploaded files"}), 500


# Clean up old chat instances periodically
@app.before_request
def cleanup_old_instances():
    """Remove chat instances that aren't associated with active sessions"""
    all_session_ids = set()
    for key in list(session.keys()):
        if key == 'session_id':
            all_session_ids.add(session[key])
    
    # Remove chat instances that aren't associated with any active session
    for chat_id in list(chat_instances.keys()):
        if chat_id not in all_session_ids:
            del chat_instances[chat_id]


# Initialize database tables
with app.app_context():
    db.create_all()


if __name__ == '__main__':
    app.run(debug=True)