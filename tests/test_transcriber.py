import pytest
import torch
import torchaudio
import numpy as np
from transcriber import Transcriber
from pyannote.core import Segment, Annotation
import io

@pytest.fixture
def transcriber():
    return Transcriber(whisper_model_name="openai/whisper-tiny", language='ru')

@pytest.fixture
def real_audio_data():
    # Загружаем реальный аудиофайл
    waveform, sample_rate = torchaudio.load("uploads/short.mp3")
    numpy_waveform = waveform.mean(dim=0).numpy()
    
    return {
        "waveform": waveform,
        "sample_rate": sample_rate,
        "raw": numpy_waveform,
        "sampling_rate": sample_rate,
        "BytesIO": io.BytesIO(numpy_waveform.tobytes())
    }

def test_transcriber_initialization(transcriber):
    assert transcriber.device == "cpu"
    assert transcriber.model is not None
    assert transcriber.speaker_segmentation_pipeline is not None

def test_audio_file_exists():
    """Проверяем, что тестовый аудиофайл существует"""
    try:
        waveform, sample_rate = torchaudio.load("uploads/short.mp3")
        assert waveform.shape[0] > 0
        assert sample_rate > 0
    except Exception as e:
        pytest.fail(f"Не удалось загрузить тестовый аудиофайл: {str(e)}")

def test_transcribe_with_speaker_detection(transcriber, real_audio_data):
    result = transcriber.transcribe_with_speaker_detection(real_audio_data)
    
    # Базовые проверки структуры результата
    assert isinstance(result, list)
    assert len(result) > 0
    
    # Проверяем структуру каждого сегмента транскрибации
    for segment in result:
        assert isinstance(segment, dict)
        assert "text" in segment
        assert "timestamp" in segment
        assert "speaker" in segment
        
        # Проверяем формат временных меток
        assert len(segment["timestamp"]) == 2
        assert isinstance(segment["timestamp"][0], (int, float))
        assert isinstance(segment["timestamp"][1], (int, float))
        assert segment["timestamp"][0] <= segment["timestamp"][1]
        
        # Проверяем формат спикера
        assert segment["speaker"].startswith("SPEAKER_")
        
        # Проверяем наличие текста
        assert isinstance(segment["text"], str)
        assert len(segment["text"].strip()) > 0

def test_get_speaker(transcriber):
    # Создаем тестовую аннотацию
    diarization = Annotation()
    diarization[Segment(0, 1)] = "SPEAKER_00"
    diarization[Segment(1, 2)] = "SPEAKER_01"
    
    # Тестируем различные сценарии
    assert transcriber.get_speaker(diarization, 0, 1) == "SPEAKER_00"
    assert transcriber.get_speaker(diarization, 1, 2) == "SPEAKER_01"
    assert transcriber.get_speaker(diarization, 2, 3) == "No_speaker"

def test_transcription_continuity(transcriber, real_audio_data):
    """Проверяем непрерывность транскрибации"""
    result = transcriber.transcribe_with_speaker_detection(real_audio_data)
    
    for i in range(len(result) - 1):
        current_end = result[i]["timestamp"][1]
        next_start = result[i + 1]["timestamp"][0]
        # Проверяем, что между сегментами нет больших разрывов
        assert next_start - current_end < 1.0, "Обнаружен большой разрыв между сегментами"

def test_transcription_language(transcriber, real_audio_data):
    """Проверяем, что текст транскрибации содержит кириллицу"""
    result = transcriber.transcribe_with_speaker_detection(real_audio_data)
    
    def contains_cyrillic(text):
        return bool(re.search('[а-яА-Я]', text))
    
    for segment in result:
        assert contains_cyrillic(segment["text"]), f"Текст не содержит кириллицу: {segment['text']}"


import re

def test_specific_words_in_transcription(transcriber, real_audio_data):
    """Проверяем наличие конкретных слов в транскрипции"""
    result = transcriber.transcribe_with_speaker_detection(real_audio_data)
    
    # Объединяем весь текст из всех сегментов
    full_transcription = " ".join(segment["text"].lower() for segment in result)
    
    expected_words = ["одинаковые", "метка"]
    for word in expected_words:
        assert word.lower() in full_transcription, f"Ожидаемое слово '{word}' не найдено в транскрипции"
    
    print(f"\nПолная транскрипция: {full_transcription}")