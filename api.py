from flask import Flask, jsonify, request
from transcriber import Transcriber
import torchaudio
from io import BytesIO
import os
import time
import threading
import torch
import traceback
import numpy as np
import soundfile as sf
import tempfile
import wave

app = Flask(__name__)

# model = "openai/whisper-large-v3"
# model = "distil-whisper/distil-large-v3"
# model = "openai/whisper-medium"
# model = "openai/whisper-tiny"

# model = 'large-v3'
# model = "small"
model = "deepdml/faster-whisper-large-v3-turbo-ct2"

# In-memory storage for transcription results and statuses
transcription_results = {}
transcription_statuses = {}
queue = []

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Lock to ensure only one transcription process runs at a time
transcription_lock = threading.Lock()

# Функция для обработки транскрипции в отдельном потоке
def process_transcription(audio_id, audio_stream):
    with transcription_lock:
        print("Устройство обнаружено: ", device)
        transcriber = Transcriber(whisper_model_name=model, language='ru', device=device)

        try:
            # Создаем временный файл
            with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as temp_wav:
                temp_filename = temp_wav.name
                
                # Конвертируем входной поток в WAV формат
                with wave.open(temp_wav, 'wb') as wav_file:
                    wav_file.setnchannels(1)  # моно
                    wav_file.setsampwidth(2)  # 16 бит
                    wav_file.setframerate(16000)  # 16кГц
                    wav_file.writeframes(audio_stream)

            # Загружаем аудио с помощью torchaudio из временного файла
            waveform, sample_rate = torchaudio.load(temp_filename)
            
            # Преобразуем в моно, если необходимо
            if waveform.shape[0] > 1:
                waveform = torch.mean(waveform, dim=0, keepdim=True)

            # Обновляем статус
            transcription_statuses[audio_id] = 'processing'
            
        except Exception as e:
            traceback_str = traceback.format_exc()
            print(f"Произошла ошибка при обработке аудио: {traceback_str}")
            transcription_statuses[audio_id] = 'failed'
            queue.remove(audio_id)
            return
        finally:
            # Удаляем временный файл
            if 'temp_filename' in locals():
                try:
                    os.remove(temp_filename)
                except:
                    pass

        # Выполняем транскрипцию
        try:
            result = transcriber.transcribe_with_speaker_detection({
                "waveform": waveform,
                "sample_rate": sample_rate,
                "BytesIO": BytesIO(audio_stream)
            })
        except Exception as e:
            traceback_str = traceback.format_exc()
            print(f"Произошла ошибка: {traceback_str}")
            transcription_statuses[audio_id] = 'failed'
            queue.remove(audio_id)
            return

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
        # If the status is 'in_queue', include the position in the queue
        if status == 'in_queue' and id in queue:
            position = queue.index(id)
            if position == 0:
                return jsonify({'status': 'processing'})
            return jsonify({'status': f'in_queue, position {position}'})
        else:
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