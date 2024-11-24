# from transformers import WhisperProcessor, WhisperForConditionalGeneration, pipeline
from transformers import pipeline
from faster_whisper import WhisperModel
import torch
from pyannote.audio import Pipeline
from pyannote.core import Segment
import traceback
import os
import numpy as np

import soundfile as sf
import io

from pydub import AudioSegment



def numpy_to_wav_bytes(numpy_audio_data, sample_rate):
    print(f"Original dtype of audio data: {numpy_audio_data.dtype}")
    print(f"Original shape of audio data: {numpy_audio_data.shape}")
        
    # if numpy_audio_data.ndim == 2 and numpy_audio_data.shape[0] == 1:
    numpy_audio_data = numpy_audio_data[0]  # Убираем лишний измерение


    print(f"Processed dtype of audio data: {numpy_audio_data.dtype}")
    print(f"Processed shape of audio data: {numpy_audio_data.shape}")

    # Create a BytesIO buffer to store the WAV data
    buf = io.BytesIO()
    try:
        # Write the audio data to the buffer in WAV format
        print("Writing audio data to WAV format in the buffer.")
        sf.write(buf, numpy_audio_data, sample_rate, format='WAV')
        buf.seek(0)  # Reset the buffer's position to the beginning
        print("Audio data successfully written to WAV format in the buffer.")
        return buf
    except Exception as e:
        print(f"Error writing audio to WAV: {e}")
        raise


def mp3_to_numpy(mp3_data):
    # Конвертируем MP3 в аудиофайл в памяти
    audio = AudioSegment.from_mp3(io.BytesIO(mp3_data))
    
    # Преобразуем в моно (если стерео)
    audio = audio.set_channels(1)
    
    # Преобразуем в нужную частоту дискретизации (например, 16kHz)
    audio = audio.set_frame_rate(16000)
    
    # Преобразуем в numpy массив
    audio_samples = np.array(audio.get_array_of_samples(), dtype=np.float32)
    
    # Нормализуем данные в диапазоне [-1, 1] для float32
    audio_samples = audio_samples / 32768.0  # для int16 -> float32
    
    return audio_samples

class Transcriber:
    def __init__(self, whisper_model_name = "tiny", language='ru', device = torch.device("cpu") ) -> None:
        # processor = WhisperProcessor.from_pretrained(whisper_model_name, language=language)
        # model = WhisperForConditionalGeneration.from_pretrained(whisper_model_name)

        # self.speech_recognition_pipe = pipeline(
        #     "automatic-speech-recognition",
        #     model=whisper_model_name,
        #     tokenizer=processor.tokenizer,
        #     feature_extractor=processor.feature_extractor,
        #     torch_dtype=torch.bfloat16,
        #     device=device,
        #     return_timestamps = True, 
        #     chunk_length_s=30,
        #     # stride_length_s=5
        # )
        self.device = device

        # download_root = "/app/models"
        download_root = "/models"
        os.makedirs(download_root, exist_ok=True)
        self.model = WhisperModel(whisper_model_name, device=str(device), compute_type="int8", 
                             download_root = download_root)

        self.speaker_segmentation_pipeline = Pipeline.from_pretrained(
        "pyannote/speaker-diarization-3.1",
        use_auth_token="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", 
        )



    def get_speaker(self, diarization, time_start, time_end):
        speaker =  diarization.crop(Segment(time_start, time_end)).argmax()
        if speaker == None:
            return "No_speaker"
        else:
            return speaker
        
    def transcribe_with_speaker_detection(self, numpy_audio_data, sample_rate=16000):
        if not isinstance(numpy_audio_data, np.ndarray):
            raise TypeError(f"Expected np.ndarray, got {type(numpy_audio_data)}")
        
        # Преобразование данных для модели сегментации говорящих (Pyannote)
        waveform = torch.from_numpy(numpy_audio_data).float()
        if waveform.ndimension() == 1:
            waveform = waveform.unsqueeze(0)  # Добавляем измерение канала
        elif waveform.ndimension() == 2 and waveform.shape[0] > 1:
            waveform = torch.mean(waveform, dim=0, keepdim=True)  # Преобразуем в моно
        
        # Перенос на устройство
        self.speaker_segmentation_pipeline.to(self.device)
        
        # Диаризация
        diarization = self.speaker_segmentation_pipeline({"waveform": waveform, "sample_rate": sample_rate}, num_speakers=2)
        self.speaker_segmentation_pipeline.to(torch.device('cpu'))  # Освобождаем устройство
        print('0'*60)
        print("Diarization ran successfully")
        print('0'*60)
        
        # Преобразование данных для Faster Whisper
        audio_bytes = numpy_to_wav_bytes(numpy_audio_data, sample_rate)
        
        # Транскрипция
        segments, info = self.model.transcribe(
            audio_bytes,
            beam_size=5,
            language='ru',
            condition_on_previous_text=False,
            vad_filter=True
        )
        
        # Форматирование результата
        transcription = []
        for segment in segments:
            transcription.append({"text": segment.text, "timestamp": [segment.start, segment.end]})
        
        # Добавление информации о говорящих
        for i in range(len(transcription)):
            speaker = self.get_speaker(diarization, transcription[i]['timestamp'][0], transcription[i]['timestamp'][1])
            try:
                if speaker == 'No_speaker':
                    speaker = transcription[i - 1]['speaker'] if i > 0 else 'SPEAKER_00'
                transcription[i].update({"speaker": speaker})
            except Exception as e:
                traceback_str = traceback.format_exc()
                print(f"An error occurred: {traceback_str}")
                print("i:", i)
                print("Transcription:", transcription)
        
        return transcription



        # print("Before saving:", transcription[0])

        # import pickle

        # with open('transcription.pkl', 'wb') as file:
        #     pickle.dump(transcription, file)

        # for i in range(len(transcription)):
        #     transcription[i]['timestamp'] = list(transcription[i]['timestamp'])

        # concated_transcription = [transcription[0]]
        # for i in range(1, len(transcription)):
        #     if (concated_transcription[-1]['speaker'] == transcription[i]['speaker']) and ( (transcription[i]['timestamp'][1] - concated_transcription[-1]['timestamp'][0]) < 20 ):
        #         concated_transcription[-1]['text'] += ' '
        #         concated_transcription[-1]['text'] += transcription[i]['text']
        #         concated_transcription[-1]['timestamp'][1] = float(transcription[i]['timestamp'][1])
        #     else:
        #         concated_transcription.append(transcription[i])
                
            
        # for i in concated_transcription:
        #     print(i)
            
        # return concated_transcription

if __name__ == "__main__":
    print("Проснулся")
    transcriber = Transcriber(whisper_model_name="openai/whisper-tiny", language='ru')

    import torchaudio

    waveform, sample_rate = torchaudio.load("20s_audio.mp3")
    numpy_waveform = waveform.mean(dim=0).numpy()

    print(transcriber.transcribe_with_speaker_detection({"waveform": waveform, "sample_rate": sample_rate, "raw": numpy_waveform, "sampling_rate": sample_rate}))