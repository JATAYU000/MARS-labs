from flask import Flask , render_template , jsonify , request
from flask_sqlalchemy import SQLAlchemy
from flask_migrate import Migrate
from googletrans import Translator

app = Flask(__name__)

app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///default.db'

app.config['SQLALCHEMY_BINDS'] = {
    'db1': 'sqlite:///database1.db',
    'db2': 'sqlite:///database2.db'
}

#os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

chat_instances = {}


db = SQLAlchemy()
db.init_app(app)

migrate = Migrate(app,db)

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
    last_name = db.Column(db.String(50),nullable=False)
    age = db.Column(db.String(4),nullable=False)

    def to_dict(self):
        return {
            'id' : self.id,
            'first' : self.first_name,
            'last' : self.last_name,
            'age' : self.age
        }

class User_Progress(db.Model):
    __bind_key__ = 'db2'
    __tablename__ = 'Progress'
    id = db.Column(db.Integer, primary_key=True)
    classes = db.Column(db.String(50), nullable=False)
    assignments = db.Column(db.String(50),nullable=False)
    test = db.Column(db.String(50),nullable=False)
    time_spend = db.Column(db.String(50),nullable=False)
    

@app.route('/')
def index():
    users = User.query.all()
    progress = User_Progress.query.all()
    return render_template('index.html',users=users,progress=progress)

@app.route('/chatbot', methods=['POST'])
def chatbot():
    data = request.get_json()
    user_input = data.get('message','')

    if not user_input:
        return jsonify({'error' : 'No message provided'}) ,400
    language = translate.detect(user_input).lang
    print(f'Detected lang :{language}')

    t_lang = user_input
    if language != 'en':
        t_lang = translate.translate(user_input,dest = 'en').text
        print(t_lang)

    llm_response = f"LLM response: {t_lang}"  

    history_entry = Chatbot(
        user_message=user_input,
        translated_message=t_lang if language != 'en' else None,
        response=llm_response
    )
    db.session.add(history_entry)
    db.session.commit()

    return jsonify({"response": llm_response})

@app.route('/users',methods = ['GET'])
def user():
    user_id = request.args.get('Id')
    if user_id:
        user = User.query.get(user_id)
        if user:
            return jsonify(user.to_dict())
        return jsonify({'message' : 'Unable to detect the user!'}), 404
    
    user_data = User.query.all()
    user_list = [user.to_dict() for user in user_data]
    return jsonify(user_list)

    

@app.route('/update', methods = ['POST'])
def create():
    data = request.get_json()
    if 'id' in data:
        return jsonify({'message' : 'User already registered'}) , 400
    data = User(first_name = data['first_name'],last_name = data['last_name'],age = data['age'])
    db.session.add(data)
    db.session.commit()

    return jsonify({'message' : 'Success'})

@app.route('/ask',methods = ['POST'])
def ask():
    data = request.get_json()
    response = Chatbot.res(data['question'])
    return jsonify({'answer' : response})


@app.route("/upload_pdf", methods=["POST"])
def upload_pdf_file():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400
    
    files = request.files.getlist('file')
    if not files or files[0].filename == '':
        return jsonify({'error': 'No file selected'}), 400
    
    # Get the session ID
    session_id = session.get('session_id')
    if not session_id or session_id not in chat_instances:
        session['session_id'] = str(uuid.uuid4())
        chat_instances[session['session_id']] = ChatPDF()
        session_id = session['session_id']
    
    # Get the ChatPDF instance for this session
    assistant = chat_instances[session_id]
    
    responses = []
    for file in files:
        filename = secure_filename(file.filename)
        # Create a temporary file
        with tempfile.NamedTemporaryFile(delete=False) as tf:
            file.save(tf.name)
            file_path = tf.name
        
        # Process the file
        start_time = time.time()
        assistant.ingest(file_path)
        processing_time = time.time() - start_time
        
        # Remove the temporary file
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
    
    # Get the session ID
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

with app.app_context():
    db.create_all()

if __name__ ==  '__main__':
    app.run(debug=True)