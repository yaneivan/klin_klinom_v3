from flask import Flask, jsonify, request
from transcriber import Transcriber
import torchaudio
from io import BytesIO
import os
import time

app = Flask(__name__)

# In-memory storage for transcription results and statuses
transcription_results = {}
transcription_statuses = {}
queue = []

@app.route('/transcribe', methods=['POST'])
def transcribe():
    audio = request.files['audio_file']
    audio_id = request.form.get('id')
    audio_stream = BytesIO(audio.read())

    transcriber = Transcriber()

    waveform, sample_rate = torchaudio.load(audio_stream)
    numpy_waveform = waveform.mean(dim=0).numpy()

    # Simulate queuing and processing
    queue.append(audio_id)
    transcription_statuses[audio_id] = 'in_queue'

    # Simulate processing (this should be done asynchronously in a real-world scenario)
    time.sleep(5)  # Simulate processing time
    result = transcriber.transcribe_with_speaker_detection({
        "waveform": waveform,
        "sample_rate": sample_rate,
        "raw": numpy_waveform,
        "sampling_rate": sample_rate
    })

    transcription_results[audio_id] = result
    transcription_statuses[audio_id] = 'completed'
    queue.remove(audio_id)

    return jsonify({'status': 'completed', 'id': audio_id})

@app.route('/transcribe/<id>', methods=['GET'])
def get_transcription(id):
    result = transcription_results.get(id)
    if result:
        return jsonify(result)
    else:
        return jsonify({'error': 'Transcription not found'}), 404

@app.route('/status/<id>', methods=['GET'])
def get_status(id):
    status = transcription_statuses.get(id)
    if status:
        return jsonify({'status': status})
    else:
        return jsonify({'error': 'Status not found'}), 404

@app.route('/queue_length', methods=['GET'])
def get_queue_length():
    return jsonify({'length': len(queue)})

if __name__ == '__main__':
    trans = Transcriber(whisper_model_name="openai/whisper-tiny", language='ru')
    app.run(host="0.0.0.0", port=4200, debug=True)