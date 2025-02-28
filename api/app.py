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
    user_id = db.Column(db.Integer, db.ForeignKey('User.id'), nullable = False)
    subjects = db.relationship('Subject', backref='progress', lazy = True)

class Subject(db.Model):
    __bind_key__ = 'db2'
    __tablename__ = 'subject'
    id = db.Column(db.Integer, primary_key = True)
    name = db.Column(db.String(50),  nullable=False)
    user_progress_id = db.Column(db.Integer, db.ForeignKey('user_progress.id'),nullable = False)
    simulation = db.relationship('Simulation', backref = 'subject', lazy = True)

class Simulation(db.Model):
    __bind_key__ = 'db2'
    __tablename__ = 'simulation'
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(100), nullable=False)
    subject_id = db.Column(db.Integer, db.ForeignKey('subject.id'), nullable=False)
    progress = db.relationship('Progress', backref='simulation', uselist=False)

class Progress(db.Model):
    __bind_key__='db2'
    __tablename__ = 'progress'
    id = db.Column(db.Integer, primary_key = True)
    simulation_id = db.Column(db.Integer, db.ForeignKey('simulation.id'), nullable = False)
    quiz = db.Column(db.Boolean, default = False)
    theory = db.Column(db.Boolean, default = False)
    simulation = db.Column(db.Boolean, default=False)
    animation = db.Column(db.Boolean, default=False)

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

    return jsonify({'message' : 'User created successfully'})

@app.route('api/ask',methods = ['POST'])
def ask():
    data = request.get_json()
    response = Chatbot.res(data['question'])
    return jsonify({'answer' : response})




with app.app_context():
    db.create_all()

if __name__ ==  '__main__':
    app.run(debug=True)