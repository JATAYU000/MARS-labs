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
from dotenv import load_dotenv 
import requests
import json
import shutil

# RAG imports
from langchain_core.globals import set_verbose, set_debug
from langchain_community.vectorstores import Chroma
from langchain_community.chat_models import ChatOllama
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

ng_uri = os.getenv("NGROK_URI")
ollama_model = os.getenv("OLLAMA_MODEL")
# Ngrok Ollama configuration from the first file
NGROK_OLLAMA_URL = ng_uri
HEADERS = {
    "Host": ng_uri,
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:135.0) Gecko/20100101 Firefox/135.0",
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.5",
    "Accept-Encoding": "gzip, deflate, br, zstd",
    "Connection": "keep-alive",
    "Cookie": f"abuse_{ng_uri}",
    "Upgrade-Insecure-Requests": "1",
    "Sec-Fetch-Dest": "document",
    "Sec-Fetch-Mode": "navigate",
    "Sec-Fetch-Site": "none",
    "Sec-Fetch-User": "?1",
    "Priority": "u=0, i",
    "TE": "trailers",
    "Content-Type": "application/json"
}

class RemoteOllamaWrapper:
    """A wrapper class to handle Ollama API calls through ngrok"""
    def __init__(self, model="phi3"):
        self.model = model
        self.url = NGROK_OLLAMA_URL
        self.headers = HEADERS
    
    def invoke(self, prompt):
        """Send a prompt to the remote Ollama instance and return the response"""
        try:
            ollama_data = {"model": self.model, "prompt": prompt}
            response = requests.post(
                f"{self.url}/api/generate",
                json=ollama_data,
                headers=self.headers
            )
            
            # For non-streaming, collect and combine all token responses
            full_text = ""
            for line in response.text.strip().split('\n'):
                try:
                    data = json.loads(line)
                    if 'response' in data:
                        full_text += data['response']
                except json.JSONDecodeError:
                    pass
            
            return full_text
        except Exception as e:
            print(f"Error calling remote Ollama: {e}")
            return f"Error: {str(e)}"


class Mentor:
    def __init__(self, llm_model=ollama_model, selected_model="ollama"):
        self.selected_model = selected_model
        self.llm_model = llm_model
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=512, chunk_overlap=100
        )

        # Load API Key
        self.gemini_api_key = os.getenv("GEMINI_API_KEY")

        # Prompt template for answering questions
        self.prompt = ChatPromptTemplate.from_template(
            """
            You are an expert mentor providing personalized guidance based on the following context. 

            CONTEXT: {context}

            USER QUERY: {question}

            INSTRUCTIONS:
            1. Prioritize information from the provided context, but supplement with general knowledge when necessary.
            2. Deliver concise but sufficiently explanatory, factually accurate responses with examples(if available) that directly address the query.
            3. Do not mention the text, if it is from the text or outside the text, just give the response and don't specify whether it is form the text or outside.
            4. Consider the chat history (included in the query) to personalize your guidance.
            5. Maintain a supportive, encouraging tone throughout.
            6. If the user's input is not in English, respond in the same language.
            7. If chat history shows a model switch occurred, adjust your response style accordingly.
            8. Remember this is a RAG (Retrieval-Augmented Generation) implementation using two models.

            Respond in a clear, helpful manner that builds the user's confidence while providing accurate information.
            """
        )

        self.vector_store = None
        self.temp_vector_store = None
        self.retriever = None
        self.temp_retriever = None
        self.chain = None
        
        # Initialize models BEFORE initializing vector store
        if selected_model == "ollama":
            # Use the remote Ollama wrapper instead of local ChatOllama
            self.remote_ollama = RemoteOllamaWrapper(model=llm_model)
            # We'll still use ChatOllama for the chain setup, but override its invoke method
            self.model = ChatOllama(model=llm_model)
        elif selected_model == "gemini":
            if not self.gemini_api_key:
                raise ValueError("Missing Gemini API key. Please set GEMINI_API_KEY in your .env file.")
            genai.configure(api_key=self.gemini_api_key)
            self.model = genai.GenerativeModel("gemini-2.0-flash")
        else:
            raise ValueError(f"Unsupported model type: {selected_model}")
            
        # Now initialize the vector store after the model is set
        self.init_vector_store()
        # Initialize temporary vector store for session-specific documents
        self.init_temp_vector_store()


    def init_vector_store(self):
        """Initialize the vector store with e5-base embeddings using the existing database"""
        try:
            # Load e5-base embeddings
            self.embedding_function = HuggingFaceEmbeddings(model_name="intfloat/e5-base")

            # Load the existing ChromaDB instance
            self.vector_store = Chroma(
                persist_directory="chroma_db",
                embedding_function=self.embedding_function
            )

            # Ensure the retriever uses the stored data
            self.retriever = self.vector_store.as_retriever(
                search_type="similarity_score_threshold",
                search_kwargs={"k": 5, "score_threshold": 0.0},
            )

            # Only set up the chain if using Ollama - Gemini will use a separate method
            if self.selected_model == "ollama" and self.model is not None:
                self.chain = (
                    {"context": self.retriever, "question": RunnablePassthrough()}
                    | self.prompt
                    | self.model
                    | StrOutputParser()
                )
            # For Gemini, we'll handle it differently in the ask method
            
            print("Vector store initialized successfully!")

        except Exception as e:
            print(f"Error initializing vector store: {e}")
            
    def init_temp_vector_store(self, session_id=None):
        """Initialize a temporary vector store for session-specific documents"""
        try:
            # Create a unique directory for this session if provided
            persist_dir = f"temp_chroma_{session_id}" if session_id else None
            
            # Create a temporary in-memory vector store
            self.temp_vector_store = Chroma(
                collection_name="temp_docs",
                embedding_function=self.embedding_function,
                persist_directory=persist_dir
            )
            
            # Set up the temporary retriever
            self.temp_retriever = self.temp_vector_store.as_retriever(
                search_type="similarity_score_threshold",
                search_kwargs={"k": 5, "score_threshold": 0.0},
            )
            
            print("Temporary vector store initialized successfully!")
            return True
        except Exception as e:
            print(f"Error initializing temporary vector store: {e}")
            return False


    def ingest(self, file_path):
        """Process and ingest a document into the persistent vector store"""
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
            
    def ingest_temp(self, file_path, session_id=None):
        """Process and ingest a document into the temporary vector store"""
        try:
            # Load the document
            loader = PyPDFLoader(file_path)
            documents = loader.load()
            
            # Add source metadata to identify this as a temporary document
            for doc in documents:
                doc.metadata["source"] = "temp_upload"
                doc.metadata["session_id"] = session_id if session_id else "unknown"
            
            # Split the documents into chunks
            splits = self.text_splitter.split_documents(documents)
            
            # Ensure temporary vector store is initialized
            if not self.temp_vector_store:
                self.init_temp_vector_store(session_id)
                
            # Add documents to temporary vector store
            self.temp_vector_store.add_documents(splits)
            
            # If we have a persist directory, save it
            if session_id:
                self.temp_vector_store.persist()
            
            return len(splits)
        except Exception as e:
            print(f"Error ingesting temporary document: {e}")
            return 0

    def ask(self, query):
        """Query both knowledge bases using the selected model"""
        if not self.vector_store:
            return "No documents have been ingested yet. Please upload documents first."
        
        try:
            # Retrieve from both vector stores if available
            main_docs = self.retriever.invoke(query) if self.retriever else []
            temp_docs = self.temp_retriever.invoke(query) if self.temp_retriever else []
            
            # Combine contexts from both sources with labels
            all_docs = main_docs + temp_docs
            
            # If we got no relevant documents, inform the user
            if not all_docs:
                return "I couldn't find any relevant information to answer your question. Please try rephrasing or asking a different question."
            
            # Prepare combined context
            context = "\n\n".join([doc.page_content for doc in all_docs])

            if self.selected_model == "ollama":
                # Use the remote Ollama wrapper instead of the chain
                formatted_prompt = self.prompt.format(context=context, question=query)
                return self.remote_ollama.invoke(formatted_prompt)
            elif self.selected_model == "gemini":
                return self.generate_answer_with_gemini(query, context)
            else:
                return "Invalid model selection or model not properly initialized."
        except Exception as e:
            print(f"Error processing query: {e}")
            return f"An error occurred: {str(e)}"

    def generate_answer_with_gemini(self, query, context):
        """Generate an answer using Gemini with the provided context."""
        prompt = f"""
            You are an expert mentor providing personalized guidance based on the following context. 

            CONTEXT: {context}

            USER QUERY: {query}

            INSTRUCTIONS:
            1. Prioritize information from the provided context, but supplement with general knowledge when necessary.
            2. Deliver concise but sufficiently explanatory, factually accurate responses with examples(if available) that directly address the query.
            3. Do not mention the text, if it is from the text or outside the text, just give the response and don't specify whether it is form the text or outside.
            4. Consider the chat history (included in the query) to personalize your guidance.
            5. Maintain a supportive, encouraging tone throughout.
            6. If the user's input is not in English, respond in the same language.
            7. If chat history shows a model switch occurred, adjust your response style accordingly.
            8. Remember this is a RAG (Retrieval-Augmented Generation) implementation using two models.

            Respond in a clear, helpful manner that builds the user's confidence while providing accurate information.
            """


        response = self.model.generate_content(prompt)
        return response.text

    def clear(self):
        """Clear the persistent vector store"""
        if self.vector_store:
            try:
                self.vector_store.delete_collection()
                self.init_vector_store()
            except Exception as e:
                print(f"Error clearing vector store: {e}")
                
    def clear_temp(self, session_id=None):
        """Clear the temporary vector store"""
        if self.temp_vector_store:
            try:
                self.temp_vector_store.delete_collection()
                self.init_temp_vector_store(session_id)
                
                # Clean up the directory if it exists
                if session_id:
                    temp_dir = f"temp_chroma_{session_id}"
                    if os.path.exists(temp_dir):
                        shutil.rmtree(temp_dir)
            except Exception as e:
                print(f"Error clearing temporary vector store: {e}")

    def load_embeddings(self, path):
        """Load pre-generated embeddings"""
        try:
            embeddings = np.load(path, allow_pickle=True)
            return len(embeddings)
        except Exception as e:
            print(f"Error loading embeddings: {e}")
            return 0


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
TEMP_UPLOAD_FOLDER = "temp_uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(TEMP_UPLOAD_FOLDER, exist_ok=True)
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER
app.config["TEMP_UPLOAD_FOLDER"] = TEMP_UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max upload

# Initialize database
db = SQLAlchemy()
db.init_app(app)
migrate = Migrate(app, db)



chat_instances = {}

# Define database models

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

class Progress(db.Model):
    __bind_key__ = 'db1'
    __tablename__ = 'Progress'
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('User.id'), nullable=False)
    progress_data = db.Column(db.JSON, nullable=False)

    def to_dict(self):
        return {
            'id': self.id,
            'user_id': self.user_id,
            'progress_data': self.progress_data
        }


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


# Routes
@app.route('/')
def index():
    """Render the main application page"""
    session_id = ensure_session()
    return render_template('index.html')

@app.route('/progress', methods=['POST'])
def add_progress():
    """Add progress data for a user"""
    try:
        data = request.get_json()
        user_id = data.get('user_id')
        progress_data = data.get('progress_data')

        if not user_id or not progress_data:
            return jsonify({'message': 'User ID and progress data are required'}), 400

        new_progress = Progress(user_id=user_id, progress_data=progress_data)
        db.session.add(new_progress)
        db.session.commit()
        return jsonify({'message': 'Progress added successfully', 'id': new_progress.id})
    except Exception as e:
        db.session.rollback()
        return jsonify({'message': f'Error: {str(e)}'}), 500


@app.route('/progress/<int:user_id>', methods=['GET'])
def get_progress(user_id):
    """Get progress data for a user"""
    try:
        progress = Progress.query.filter_by(user_id=user_id).all()
        if not progress:
            return jsonify({'message': 'No progress data found for the user'}), 404

        progress_list = [p.to_dict() for p in progress]
        return jsonify(progress_list)
    except Exception as e:
        return jsonify({'message': f'Error: {str(e)}'}), 500


@app.route('/progress/<int:user_id>', methods=['PUT'])
def update_progress(user_id):
    """Update progress data for a user"""
    try:
        data = request.get_json()
        progress_data = data.get('progress_data')

        if not progress_data:
            return jsonify({'message': 'Progress data is required'}), 400

        progress = Progress.query.filter_by(user_id=user_id).first()
        if not progress:
            return jsonify({'message': 'No progress data found for the user'}), 404

        progress.progress_data = progress_data
        db.session.commit()
        return jsonify({'message': 'Progress updated successfully'})
    except Exception as e:
        db.session.rollback()
        return jsonify({'message': f'Error: {str(e)}'}), 500


@app.route('/chat')
def chat():
    """Render the chat application page"""
    session_id = ensure_session()
    return render_template('chat.html')

@app.route('/admin', methods=['GET'])
def admin_view():
    return render_template('admin_page.html')

@app.route('/user/progress', methods=['GET'])
def user_progress_view():
    return render_template('progress_page.html')


@app.route('/upload', methods=['POST'])
def upload_file():
    """Handle document uploads for the permanent vector store"""
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


@app.route('/upload/temp', methods=['POST'])
def upload_temp_file():
    """Handle document uploads for temporary session-specific vector store"""
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
        
        # Save to the temp uploads folder
        temp_file_path = os.path.join(app.config["TEMP_UPLOAD_FOLDER"], f"{session_id}_{filename}")
        file.save(temp_file_path)
        
        start_time = time.time()
        chunks_processed = assistant.ingest_temp(temp_file_path, session_id)
        processing_time = time.time() - start_time
        
        response = f"Added temporary document {filename} for this session ({chunks_processed} chunks)"
        responses.append(response)
        
        # Update session messages
        session['messages'].append({'content': response, 'is_user': False})
    
    return jsonify({'messages': responses, 'temp_docs_added': True}), 200


@app.route('/ask', methods=['POST'])
def ask_question():
    """Process user questions"""
    data = request.json
    user_text = data.get('message', '').strip()
    model_selection = data.get('model')
    user_id = data.get('user_id')
    chat_history = data.get('chat_history', [])
    use_temp_docs = data.get('use_temp_docs', True)  # Default to using temporary docs if available

    print(data)

    if not user_text:
        return jsonify({'error': 'Empty message'}), 400

    session_id = ensure_session()
    
    if model_selection and chat_instances[session_id].selected_model != model_selection:
        # Save the temporary vector store before recreating the instance
        temp_vector_store = None
        if session_id in chat_instances and chat_instances[session_id].temp_vector_store:
            temp_vector_store = chat_instances[session_id].temp_vector_store
        
        chat_instances[session_id] = Mentor(selected_model=model_selection)
        
        # Restore temporary vector store if it existed
        if temp_vector_store:
            chat_instances[session_id].temp_vector_store = temp_vector_store
            chat_instances[session_id].temp_retriever = temp_vector_store.as_retriever(
                search_type="similarity_score_threshold",
                search_kwargs={"k": 5, "score_threshold": 0.0},
            )
    
    assistant = chat_instances[session_id]
    
    # Handle chat history 
    if chat_history:
        pass
    else:
        chat_history = "No chat history available."

    # Retrieve user progress data
    progress = Progress.query.filter_by(user_id=user_id).all()
    if progress:
        progress_text = "\n".join([str(p.progress_data) for p in progress])
    else:
        progress_text = "No progress data available."

    # Include progress data in the prompt
    final_prompt = f"""
    User Progress:
    {progress_text}

    Question: {user_text}

    chat history: {chat_history}
    """

    agent_text = assistant.ask(final_prompt)

    session['messages'].append({'content': user_text, 'is_user': True})
    session['messages'].append({'content': agent_text, 'is_user': False})

    return jsonify({
        'user_message': user_text,
        'response': agent_text
    }), 200


@app.route('/clear', methods=['POST'])
def clear_conversation():
    """Clear the conversation history and temporary vector store"""
    session_id = session.get('session_id')
    if session_id and session_id in chat_instances:
        # Only clear the temporary store, not the persistent one
        chat_instances[session_id].clear_temp(session_id)
        
        # Clean up temp files
        for file in os.listdir(app.config["TEMP_UPLOAD_FOLDER"]):
            if file.startswith(f"{session_id}_"):
                try:
                    os.remove(os.path.join(app.config["TEMP_UPLOAD_FOLDER"], file))
                except Exception as e:
                    print(f"Error removing temporary file: {e}")
    
    session['messages'] = []
    
    return jsonify({'status': 'success'}), 200


@app.route('/clear/temp', methods=['POST'])
def clear_temp_documents():
    """Clear only the temporary documents"""
    session_id = session.get('session_id')
    if session_id and session_id in chat_instances:
        chat_instances[session_id].clear_temp(session_id)
        
        # Clean up temp files
        for file in os.listdir(app.config["TEMP_UPLOAD_FOLDER"]):
            if file.startswith(f"{session_id}_"):
                try:
                    os.remove(os.path.join(app.config["TEMP_UPLOAD_FOLDER"], file))
                except Exception as e:
                    print(f"Error removing temporary file: {e}")
    
    return jsonify({'status': 'success', 'message': 'Temporary documents cleared'}), 200


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
    
    for chat_id in list(chat_instances.keys()):
        if chat_id not in all_session_ids:
            del chat_instances[chat_id]


with app.app_context():
    db.create_all()


if __name__ == '__main__':
    app.run(debug=True)