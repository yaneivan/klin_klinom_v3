from flask import Flask, jsonify, request
from transcriber import Transcriber
from io import BytesIO
import os
import time
import threading
import torch
import torchaudio
import traceback
import numpy as np
import soundfile as sf
import tempfile
import wave
import io
import traceback

app = Flask(__name__)

# model = "openai/whisper-large-v3"
# model = "distil-whisper/distil-large-v3"
# model = "openai/whisper-medium"
# model = "openai/whisper-tiny"

# model = 'large-v3'
# model = "small"
# model = "tiny"
model = "deepdml/faster-whisper-large-v3-turbo-ct2"

# In-memory storage for transcription results and statuses
transcription_results = {}
transcription_statuses = {}
queue = []

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Lock to ensure only one transcription process runs at a time
transcription_lock = threading.Lock()

# Функция для обработки транскрипции в отдельном потоке
def process_transcription(audio_id, audio_data):
    try:
        with transcription_lock:
            print("Устройство обнаружено: ", device)
            transcriber = Transcriber(whisper_model_name=model, language='ru', device=device)

            # Обновляем статус
            transcription_statuses[audio_id] = 'processing'

        # Логирование типа и формы данных
        print(f"Type of audio_data: {type(audio_data)}")
        print(f"Shape of audio_data: {audio_data.shape}")
        print(f"Dtype of audio_data: {audio_data.dtype}")

        # Убедимся, что данные имеют правильную форму для `numpy.ndarray`
        if audio_data.ndim == 1:  # Если одномерный массив
            audio_data = np.expand_dims(audio_data, axis=0)  # Добавляем ось канала
        elif audio_data.ndim == 2 and audio_data.shape[0] > 1:  # Если многоканальный
            audio_data = np.mean(audio_data, axis=0, keepdims=True)  # Преобразуем в моно

        sample_rate = 16000  # Частота дискретизации, используемая для всех данных

        # Выполнение транскрипции
        result = transcriber.transcribe_with_speaker_detection(
            audio_data,
            sample_rate
        )

        # Обновляем результаты
        transcription_results[audio_id] = result
        transcription_statuses[audio_id] = 'completed'

    except Exception as e:
        traceback_str = traceback.format_exc()
        print(f"Ошибка при обработке {audio_id}: {traceback_str}")
        transcription_statuses[audio_id] = 'failed'

    finally:
        with transcription_lock:
            if audio_id in queue:
                queue.remove(audio_id)




from pydub import AudioSegment
from pydub.utils import mediainfo

@app.route('/transcribe', methods=['POST'])
def transcribe():
    # Извлекаем аудиофайл и его ID из запроса
    audio = request.files.get('audio_file')
    if not audio:
        return jsonify({'error': 'No audio file provided'}), 400
    
    audio_id = request.form.get('id')
    if not audio_id:
        return jsonify({'error': 'No audio ID provided'}), 400
    
    # Печатаем информацию о файле для дебага
    print(f"Received audio file: {audio.filename}")
    print(f"Content type: {audio.content_type}")


    try:
        # Определяем MIME-тип файла
        audio_format = audio.content_type
        print(f"Received audio file format: {audio_format}")
        
        # Чтение аудиофайла в байты
        audio_stream = audio.read()

        # Получаем информацию о файле
        # file_info = mediainfo(io.BytesIO(audio_stream))
        # print(f"File info: {file_info}")

        if 'mp3' in audio.filename.lower():  # MP3
            # Преобразуем MP3 в numpy
            print(f"Got mp3! {audio.filename}, audio format: {audio_format}")
            audio_segment = AudioSegment.from_file(io.BytesIO(audio_stream))
            audio_segment = audio_segment.set_frame_rate(16000).set_channels(1)  # Преобразуем в моно 16kHz
            audio_data = np.array(audio_segment.get_array_of_samples(), dtype=np.int16)
            audio_data = audio_data.astype(np.float32) / 32768.0

        elif 'audio/wav' in audio_format:  # WAV
            # Преобразуем WAV в numpy
            audio_segment = AudioSegment.from_file(io.BytesIO(audio_stream), format="wav")
            audio_segment = audio_segment.set_frame_rate(16000).set_channels(1)  # Преобразуем в моно 16kHz
            audio_data = np.array(audio_segment.get_array_of_samples(), dtype=np.int16)

        elif 'audio/flac' in audio_format:  # FLAC
            # Преобразуем FLAC в numpy
            audio_segment = AudioSegment.from_file(io.BytesIO(audio_stream), format="flac")
            audio_segment = audio_segment.set_frame_rate(16000).set_channels(1)  # Преобразуем в моно 16kHz
            audio_data = np.array(audio_segment.get_array_of_samples(), dtype=np.int16)

        else:  # Другие форматы, например OGG
            # Используем librosa для универсальной загрузки
            audio_data, sr = librosa.load(io.BytesIO(audio_stream), sr=16000, mono=True)  # Преобразуем в моно и 16kHz
            audio_data = (audio_data * 32767).astype(np.int16)  # Преобразуем в формат int16

    except Exception as e:
        traceback.print_exc()
        return jsonify({'error': f'Failed to process audio file: {str(e)}'}), 500
    


    
    min_value = np.min(audio_data)
    max_value = np.max(audio_data)
    print(f"Диапазон значений аудио данных: от {min_value} до {max_value}")
    # audio_data она будет размерностями 

    # Добавляем ID аудио в очередь и устанавливаем его статус
    queue.append(audio_id)
    transcription_statuses[audio_id] = 'in_queue'

    # Запускаем новый поток для асинхронной обработки транскрипции
    threading.Thread(target=process_transcription, args=(audio_id, audio_data)).start()

    # Возвращаем статус запроса
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