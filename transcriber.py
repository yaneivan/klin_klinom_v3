from transformers import WhisperProcessor, WhisperForConditionalGeneration, pipeline
import torch
from pyannote.audio import Pipeline
from pyannote.core import Segment

class Transcriber:
    def __init__(self, whisper_model_name = "openai/whisper-tiny", language='ru' ) -> None:
        processor = WhisperProcessor.from_pretrained(whisper_model_name, language=language)
        model = WhisperForConditionalGeneration.from_pretrained(whisper_model_name)

        self.speech_recognition_pipe = pipeline(
            "automatic-speech-recognition",
            model=whisper_model_name,
            tokenizer=processor.tokenizer,
            feature_extractor=processor.feature_extractor,
            torch_dtype=torch.bfloat16,
            device="cpu",
            return_timestamps = True, 
            chunk_length_s=30,
        )

        self.speaker_segmentation_pipeline = Pipeline.from_pretrained(
        "pyannote/speaker-diarization-3.1",
        use_auth_token="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")



    def get_speaker(self, diarization, time_start, time_end):
        speaker =  diarization.crop(Segment(time_start, time_end)).argmax()
        if speaker == None:
            return "No_speaker"
        else:
            return speaker
        
    def transcribe_with_speaker_detection(self, data_to_transcribe):
        # run the pipeline on an audio file
        diarization = self.speaker_segmentation_pipeline(data_to_transcribe, num_speakers=2)

        transcribtion = self.speech_recognition_pipe(data_to_transcribe)['chunks']

        for i in range(len(transcribtion)):
            transcribtion[i].update({"speaker": self.get_speaker(diarization, transcribtion[i]['timestamp'][0],  transcribtion[i]['timestamp'][1])})

        print(transcribtion[i])
        
        return transcribtion

if __name__ == "__main__":
    print("Проснулся")
    transcriber = Transcriber(whisper_model_name="openai/whisper-tiny", language='ru')

    import torchaudio

    waveform, sample_rate = torchaudio.load("20s_audio.mp3")
    numpy_waveform = waveform.mean(dim=0).numpy()

    print(transcriber.transcribe_with_speaker_detection({"waveform": waveform, "sample_rate": sample_rate, "raw": numpy_waveform, "sampling_rate": sample_rate}))