from flask import Flask, render_template, request, redirect, url_for, session, flash, abort, jsonify
from flask import send_from_directory, send_file
from flask_rangerequest import RangeRequest
import sqlite3
import os
import requests
import time
from flask_cors import CORS
import uuid
from apscheduler.schedulers.background import BackgroundScheduler
from mutagen.mp3 import MP3  # Import mutagen to get audio length

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes
app.secret_key = 'your_secret_key'  # Change this to a secure secret key
app.config['UPLOAD_FOLDER'] = 'uploads'  # Folder to store uploaded MP3 files

# Transcription API base URL
TRANSCRIPTION_API_BASE_URL = 'http://localhost:4200'

# Database connection
def get_db_connection():
    conn = sqlite3.connect('database.db')
    conn.row_factory = sqlite3.Row
    return conn

# Initialize the database
def init_db():
    conn = get_db_connection()
    with conn:
        conn.execute('''
            CREATE TABLE IF NOT EXISTS users (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                username TEXT NOT NULL,
                password TEXT NOT NULL
            )
        ''')
        conn.execute('''
            CREATE TABLE IF NOT EXISTS projects (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id INTEGER NOT NULL,
                name TEXT NOT NULL,
                mp3_path TEXT NOT NULL,
                transcription_id TEXT,
                transcription_status TEXT,
                transcription_result TEXT,
                created_at REAL NOT NULL,  -- Add created_at column
                FOREIGN KEY (user_id) REFERENCES users (id)
            )
        ''')
        conn.execute('''
            CREATE TABLE IF NOT EXISTS transcription_speed (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                project_id INTEGER NOT NULL,
                audio_length REAL NOT NULL,
                transcription_time REAL NOT NULL,
                FOREIGN KEY (project_id) REFERENCES projects (id)
            )
        ''')
    conn.close()

# Function to get audio length
def get_audio_length(mp3_path):
    audio = MP3(mp3_path)
    return audio.info.length

# Function to send audio file for transcription
def send_for_transcription(mp3_path, transcription_id):
    with open(mp3_path, 'rb') as f:
        files = {'audio_file': f}
        data = {'id': transcription_id}
        response = requests.post(f'{TRANSCRIPTION_API_BASE_URL}/transcribe', files=files, data=data)
        if response.status_code == 200:
            return response.json()['status']
        elif response.status_code == 404:
            return 'failed'
        else:
            return None

# Function to update transcription status
def update_transcription_status(project, conn):
    transcription_id = project['transcription_id']
    if transcription_id:
        status_response = requests.get(f'{TRANSCRIPTION_API_BASE_URL}/status/{transcription_id}')
        if status_response.status_code == 200:
            status = status_response.json()['status']
            conn.execute('UPDATE projects SET transcription_status = ? WHERE id = ?', (status, project['id']))
            conn.commit()
            if status == 'completed':
                result_response = requests.get(f'{TRANSCRIPTION_API_BASE_URL}/transcribe/{transcription_id}')
                if result_response.status_code == 200:
                    result = result_response.json()
                    conn.execute('UPDATE projects SET transcription_result = ? WHERE id = ?', (str(result), project['id']))
                    conn.commit()
                    # Calculate transcription time
                    created_at = project['created_at']
                    transcription_time = time.time() - created_at
                    audio_length = get_audio_length(project['mp3_path'])
                    conn.execute('INSERT INTO transcription_speed (project_id, audio_length, transcription_time) VALUES (?, ?, ?)', 
                                 (project['id'], audio_length, transcription_time))
                    conn.commit()
        elif status_response.status_code == 404:
            conn.execute('UPDATE projects SET transcription_status = ? WHERE id = ?', ('failed', project['id']))
            conn.commit()

# Background task to periodically check transcription status
def check_transcription_status():
    conn = get_db_connection()
    projects = conn.execute('SELECT * FROM projects WHERE transcription_status != ?', ('completed',)).fetchall()
    for project in projects:
        update_transcription_status(project, conn)
    conn.close()

# Initialize the background scheduler
scheduler = BackgroundScheduler()
scheduler.add_job(check_transcription_status, 'interval', seconds=30)  # Adjust the interval as needed
scheduler.start()

# Route for the root URL
@app.route('/')
def index():
    if 'user_id' not in session:
        return redirect(url_for('login'))
    return redirect(url_for('projects'))

# Route for the login page
@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']

        conn = get_db_connection()
        user = conn.execute('SELECT * FROM users WHERE username = ? AND password = ?', (username, password)).fetchone()
        conn.close()

        if user:
            session['user_id'] = user['id']
            return redirect(url_for('projects'))
        else:
            return render_template('login.html', error='Invalid username or password')

    return render_template('login.html')

# Route for the registration page
@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']

        conn = get_db_connection()
        conn.execute('INSERT INTO users (username, password) VALUES (?, ?)', (username, password))
        conn.commit()
        conn.close()

        return redirect(url_for('login'))

    return render_template('register.html')

# Route for the projects page
@app.route('/projects', methods=['GET', 'POST'])
def projects():
    if 'user_id' not in session:
        return redirect(url_for('login'))

    user_id = session['user_id']
    conn = get_db_connection()

    # Fetch the username
    user = conn.execute('SELECT username FROM users WHERE id = ?', (user_id,)).fetchone()
    username = user['username'] if user else 'Unknown'

    if request.method == 'POST':
        if 'name' in request.form and 'mp3' in request.files:
            name = request.form['name']
            mp3 = request.files['mp3']
            if mp3:
                filename = os.path.join(app.config['UPLOAD_FOLDER'], mp3.filename)
                mp3.save(filename)
                transcription_id = str(uuid.uuid4())  # Generate a unique ID for the transcription
                status = send_for_transcription(filename, transcription_id)
                if status == 'in_queue':
                    created_at = time.time()  # Capture creation time
                    conn.execute('INSERT INTO projects (user_id, name, mp3_path, transcription_id, transcription_status, created_at) VALUES (?, ?, ?, ?, ?, ?)', 
                                 (user_id, name, filename, transcription_id, 'in_queue', created_at))
                    conn.commit()
        elif 'delete' in request.form:
            project_id = request.form['delete']
            conn.execute('DELETE FROM projects WHERE id = ? AND user_id = ?', (project_id, user_id))
            conn.commit()

    projects = conn.execute('SELECT * FROM projects WHERE user_id = ?', (user_id,)).fetchall()
    conn.close()

    return render_template('projects.html', projects=projects, username=username)

# Route for viewing a specific project
@app.route('/project/<int:project_id>')
def view_project(project_id):
    if 'user_id' not in session:
        return redirect(url_for('login'))

    user_id = session['user_id']
    conn = get_db_connection()
    project = conn.execute('SELECT * FROM projects WHERE id = ? AND user_id = ?', (project_id, user_id)).fetchone()
    if project is None:
        conn.close()
        abort(404)  # Return a 404 error if the project does not exist or does not belong to the user

    update_transcription_status(project, conn)
    conn.close()

    return render_template('view_project.html', project=project)

# Route for serving project status
@app.route('/project/<int:project_id>/status', methods=['GET'])
def get_project_status(project_id):
    if 'user_id' not in session:
        return jsonify({'error': 'Unauthorized'}), 401

    user_id = session['user_id']
    conn = get_db_connection()
    project = conn.execute('SELECT * FROM projects WHERE id = ? AND user_id = ?', (project_id, user_id)).fetchone()
    conn.close()

    if project is None:
        return jsonify({'error': 'Project not found'}), 404

    return jsonify({'status': project['transcription_status']})

# Route for the audio editing page
@app.route('/editor/<int:project_id>')
def editor(project_id):
    if 'user_id' not in session:
        return redirect(url_for('login'))

    user_id = session['user_id']
    conn = get_db_connection()
    project = conn.execute('SELECT * FROM projects WHERE id = ? AND user_id = ?', (project_id, user_id)).fetchone()
    if project is None:
        conn.close()
        abort(404)  # Return a 404 error if the project does not exist or does not belong to the user

    update_transcription_status(project, conn)
    conn.close()

    # Convert the Row object to a dictionary
    project_dict = dict(project)

    return render_template('editor.html', project=project_dict, project_id=project_id)


@app.route('/project/<int:project_id>/transcription', methods=['GET'])
def get_transcription(project_id):
    conn = get_db_connection()
    project = conn.execute('SELECT * FROM projects WHERE id = ?', (project_id,)).fetchone()
    conn.close()
    if project and project['transcription_result']:
        return jsonify(eval(project['transcription_result']))
    return jsonify({'error': 'Transcription not found'}), 404

@app.route('/uploads/<path:filename>')
def download_file(filename):
    file_path = os.path.join(filename)
    range_header = request.headers.get('Range', None)
    
    app.logger.debug(f'Range header: {range_header}')
    
    if not range_header:
        return send_file(file_path)

    try:
        with open(file_path, 'rb') as f:
            range_request = RangeRequest(f, range_header)
            return range_request.make_response()
    except ValueError as e:
        app.logger.error(f'Range request error: {str(e)}')
        return send_file(file_path)

@app.route('/project/<int:project_id>/update_transcription', methods=['POST'])
def update_transcription(project_id):
    if 'user_id' not in session:
        return jsonify({'error': 'Unauthorized'}), 401

    user_id = session['user_id']
    data = request.json
    chunk_index = data.get('chunk_index')
    updated_text = data.get('updated_text')
    print("Chunk index: ", chunk_index, "Updated text: ", updated_text)

    if chunk_index is None or updated_text is None:
        return jsonify({'error': 'Invalid data'}), 400

    conn = get_db_connection()
    project = conn.execute('SELECT * FROM projects WHERE id = ? AND user_id = ?', (project_id, user_id)).fetchone()
    if project is None:
        conn.close()
        return jsonify({'error': 'Project not found'}), 404

    try:
        transcription_result = eval(project['transcription_result'])
    except Exception as e:
        conn.close()
        return jsonify({'error': 'Invalid transcription result format'}), 400

    if chunk_index < 0 or chunk_index >= len(transcription_result):
        conn.close()
        return jsonify({'error': 'Chunk index out of range'}), 400

    transcription_result[chunk_index]['text'] = updated_text
    conn.execute('UPDATE projects SET transcription_result = ? WHERE id = ?', (str(transcription_result), project_id))
    conn.commit()
    conn.close()
    return jsonify({'success': True})

if __name__ == '__main__':
    if not os.path.exists(app.config['UPLOAD_FOLDER']):
        os.makedirs(app.config['UPLOAD_FOLDER'])
    init_db()
    app.run(host="0.0.0.0", debug=True)