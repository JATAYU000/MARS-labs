from flask import Flask, render_template, jsonify, request, session
from flask_sqlalchemy import SQLAlchemy
from flask_migrate import Migrate
from googletrans import Translator
from werkzeug.utils import secure_filename
import os
import time
import tempfile
import uuid
from rag import Mentor
import chromadb
import numpy as np

app = Flask(__name__)

app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///default.db'
app.config['SQLALCHEMY_BINDS'] = {
    'db1': 'sqlite:///database1.db',
    'db2': 'sqlite:///database2.db'
}
app.secret_key = "your_secret_key_here"  
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024 

UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER

chat_instances = {}

db = SQLAlchemy()
db.init_app(app)
migrate = Migrate(app, db)
translate = Translator()

class Chatbot(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    user_message = db.Column(db.String(500), nullable=False)
    translated_message = db.Column(db.String(500), nullable=True)
    response = db.Column(db.String(500), nullable=False)

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

@app.route('/')
def index():
    if 'messages' not in session:
        session['messages'] = []
    
    if 'session_id' not in session:
        session['session_id'] = str(uuid.uuid4())
    
    if session['session_id'] not in chat_instances:
        chat_instances[session['session_id']] = Mentor()
    
    return render_template('index.html', messages=session.get('messages', []))

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400
    
    files = request.files.getlist('file')
    if not files or files[0].filename == '':
        return jsonify({'error': 'No file selected'}), 400
    

    session_id = session.get('session_id')
    if not session_id or session_id not in chat_instances:
        session['session_id'] = str(uuid.uuid4())
        chat_instances[session['session_id']] = ChatPDF()
        session_id = session['session_id']
    

    assistant = chat_instances[session_id]
    
    responses = []
    for file in files:
        filename = secure_filename(file.filename)

        with tempfile.NamedTemporaryFile(delete=False) as tf:
            file.save(tf.name)
            file_path = tf.name
        

        start_time = time.time()
        assistant.ingest(file_path)
        processing_time = time.time() - start_time
        
        os.remove(file_path)
        
        response = f"Ingested {filename} in {processing_time:.2f} seconds"
        responses.append(response)
        
        # Update session messages
        if 'messages' not in session:
            session['messages'] = []
        session['messages'].append({'content': response, 'is_user': False})
    
    return jsonify({'messages': responses}), 200

@app.route('/ask', methods=['POST'])
def ask_question():
    data = request.json
    user_text = data.get('message', '').strip()
    
    if not user_text:
        return jsonify({'error': 'Empty message'}), 400
    

    session_id = session.get('session_id')
    if not session_id or session_id not in chat_instances:
        return jsonify({'error': 'No documents ingested yet'}), 400
    
    # Get the ChatPDF instance for this session
    assistant = chat_instances[session_id]
    
    # Process the question
    agent_text = assistant.ask(user_text)
    
    # Update session messages
    if 'messages' not in session:
        session['messages'] = []
    
    session['messages'].append({'content': user_text, 'is_user': True})
    session['messages'].append({'content': agent_text, 'is_user': False})
    
    return jsonify({
        'user_message': user_text,
        'response': agent_text
    }), 200

@app.route('/clear', methods=['POST'])
def clear_conversation():
    # Get the session ID
    session_id = session.get('session_id')
    if session_id and session_id in chat_instances:
        # Clear the ChatPDF instance
        chat_instances[session_id].clear()
    
    # Clear messages in session
    session['messages'] = []
    
    return jsonify({'status': 'success'}), 200

# Optional: Clean up unused chat instances periodically
@app.before_request
def cleanup_old_instances():
    # This is a simple cleanup approach - you might want to implement a more 
    # sophisticated approach with timeouts for production use
    all_session_ids = set()
    for sid in session.keys():
        if sid.startswith('session_id'):
            all_session_ids.add(session[sid])
    
    # Remove chat instances that aren't associated with any active session
    for chat_id in list(chat_instances.keys()):
        if chat_id not in all_session_ids:
            del chat_instances[chat_id]

@app.route('/users', methods=['GET'])
def user():
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
    data = request.get_json()
    if 'id' in data:
        return jsonify({'message': 'User already registered'}), 400
    data = User(first_name=data['first_name'], last_name=data['last_name'], age=data['age'])
    db.session.add(data)
    db.session.commit()
    return jsonify({'message': 'Success'})

@app.route('/chatbot', methods=['POST'])
def chatbot():
    data = request.get_json()
    response = Chatbot.res(data['question'])
    return jsonify({'answer': response})

chroma_client = chromadb.PersistentClient(path="chroma_db")
collection = chroma_client.get_or_create_collection("ncert_docs")

def process_npy_files(embeddings_path, chunks_path):
    """Loads .npy files, inserts embeddings into ChromaDB, and deletes files after processing."""
    try:
        embeddings = np.load(embeddings_path, allow_pickle=True)
        chunked_docs = np.load(chunks_path, allow_pickle=True)

        # Insert into ChromaDB
        for i, (doc, emb) in enumerate(zip(chunked_docs, embeddings)):
            collection.add(
                ids=[f"{doc['source']}_chunk_{doc['chunk_id']}"],
                embeddings=[emb.tolist()],  # Convert numpy array to list
                metadatas=[{"source": doc["source"], "chunk_id": doc["chunk_id"]}],
                documents=[doc["text"]]
            )

    finally:
        # Delete the files after processing
        os.remove(embeddings_path)
        os.remove(chunks_path)


@app.route("/upload_npy", methods=["POST"])
def upload_npy():
    """Handles file uploads and processes them into ChromaDB."""
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
    process_npy_files(embeddings_path, chunks_path)

    return jsonify({"message": "Files uploaded and processed successfully"}), 200


with app.app_context():
    db.create_all()

if __name__ == '__main__':
    app.run(debug=True)