from flask import Flask , render_template , jsonify , request
from flask_sqlalchemy import SQLAlchemy
from flask_migrate import Migrate


app = Flask(__name__)

app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///default.db'

app.config['SQLALCHEMY_BINDS'] = {
    'db1': 'sqlite:///database1.db',
    'db2': 'sqlite:///database2.db'
}

db = SQLAlchemy()
db.init_app(app)

migrate = Migrate(app,db)

class Chatbot(db.Model):
    
    id = db.Column(db.Integer , primary_key = True)
    question = db.Column(db.String(200), unique=True, nullable=False)
    answer = db.Column(db.String(500),nullable = False)

    def res(user_input):
        response = Chatbot.query.filter_by(question = user_input).first()
        if response:
            return response.answer
        return 'Sorry idk'

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

@app.route('/chatbot', methods=['GET'])
def chatbot():
    user_input = request.args.get('question')
    if user_input:
        response = Chatbot.res(user_input)
        return jsonify({"answer": response})
    return jsonify({"error": "No question provided"}), 400


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

    return jsonify({'message' : 'success'})

with app.app_context():
    db.create_all()

if __name__ ==  '__main__':
    app.run(debug=True)