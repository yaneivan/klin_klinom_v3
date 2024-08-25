from flask import Flask, jsonify, request
from transcriber import Transcriber
import torchaudio
from io import BytesIO
import os
import time
import threading
import torch

app = Flask(__name__)

# In-memory storage for transcription results and statuses
transcription_results = {}
transcription_statuses = {}
queue = []

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Function to process transcription in a separate thread
def process_transcription(audio_id, audio_stream):
    print("Device detected: ", device)
    transcriber = Transcriber(whisper_model_name="openai/whisper-tiny", language='ru', device = device)

    # Load the audio file and convert it to a format suitable for transcription
    waveform, sample_rate = torchaudio.load(BytesIO(audio_stream))
    numpy_waveform = waveform.mean(dim=0).numpy()

    # Perform transcription with speaker detection
    result = transcriber.transcribe_with_speaker_detection({
        "waveform": waveform,
        "sample_rate": sample_rate,
        "raw": numpy_waveform,
        "sampling_rate": sample_rate
    })

    # Update transcription results and statuses
    transcription_results[audio_id] = result
    transcription_statuses[audio_id] = 'completed'
    queue.remove(audio_id)

# Route to handle transcription requests
@app.route('/transcribe', methods=['POST'])
def transcribe():
    # Extract the audio file and its ID from the request
    audio = request.files['audio_file']
    audio_id = request.form.get('id')
    audio_stream = audio.read()

    # Add the audio ID to the queue and set its initial status
    queue.append(audio_id)
    transcription_statuses[audio_id] = 'in_queue'

    # Start a new thread to process the transcription asynchronously
    threading.Thread(target=process_transcription, args=(audio_id, audio_stream)).start()

    # Return the initial status of the transcription request
    return jsonify({'status': 'in_queue', 'id': audio_id})

# Route to fetch the transcription result by ID
@app.route('/transcribe/<id>', methods=['GET'])
def get_transcription(id):
    # Retrieve the transcription result for the given ID
    result = transcription_results.get(id)
    if result:
        # Return the transcription result if found
        return jsonify(result)
    else:
        # Return an error if the transcription result is not found
        return jsonify({'error': 'Transcription not found'}), 404

# Route to fetch the transcription status by ID
@app.route('/status/<id>', methods=['GET'])
def get_status(id):
    # Retrieve the transcription status for the given ID
    status = transcription_statuses.get(id)
    if status:
        # Return the transcription status if found
        return jsonify({'status': status})
    else:
        # Return an error if the transcription status is not found
        return jsonify({'error': 'Status not found'}), 404

# Route to fetch the length of the transcription queue
@app.route('/queue_length', methods=['GET'])
def get_queue_length():
    # Return the number of items currently in the transcription queue
    return jsonify({'length': len(queue)})

if __name__ == '__main__':
    # Start the Flask application
    app.run(host="0.0.0.0", port=4200, debug=True)