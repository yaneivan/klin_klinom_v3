from flask import Flask, render_template, request, redirect, url_for, session, flash, abort, jsonify
import sqlite3
import os
import requests
import time
from flask_cors import CORS
import uuid
from apscheduler.schedulers.background import BackgroundScheduler

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
                FOREIGN KEY (user_id) REFERENCES users (id)
            )
        ''')
    conn.close()

# Function to send audio file for transcription
def send_for_transcription(mp3_path, transcription_id):
    with open(mp3_path, 'rb') as f:
        files = {'audio_file': f}
        data = {'id': transcription_id}
        response = requests.post(f'{TRANSCRIPTION_API_BASE_URL}/transcribe', files=files, data=data)
        if response.status_code == 200:
            return response.json()['status']
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
                    conn.execute('INSERT INTO projects (user_id, name, mp3_path, transcription_id, transcription_status) VALUES (?, ?, ?, ?, ?)', 
                                 (user_id, name, filename, transcription_id, 'in_queue'))
                    conn.commit()
        elif 'delete' in request.form:
            project_id = request.form['delete']
            conn.execute('DELETE FROM projects WHERE id = ? AND user_id = ?', (project_id, user_id))
            conn.commit()

    projects = conn.execute('SELECT * FROM projects WHERE user_id = ?', (user_id,)).fetchall()
    conn.close()

    return render_template('projects.html', projects=projects)

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

    return render_template('editor.html', project=project)

if __name__ == '__main__':
    if not os.path.exists(app.config['UPLOAD_FOLDER']):
        os.makedirs(app.config['UPLOAD_FOLDER'])
    init_db()
    app.run(debug=True)