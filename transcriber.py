from transformers import WhisperProcessor, WhisperForConditionalGeneration, pipeline
import torch
from pyannote.audio import Pipeline
from pyannote.core import Segment
import traceback

class Transcriber:
    def __init__(self, whisper_model_name = "openai/whisper-tiny", language='ru', device = "cpu" ) -> None:
        processor = WhisperProcessor.from_pretrained(whisper_model_name, language=language)
        model = WhisperForConditionalGeneration.from_pretrained(whisper_model_name)

        self.speech_recognition_pipe = pipeline(
            "automatic-speech-recognition",
            model=whisper_model_name,
            tokenizer=processor.tokenizer,
            feature_extractor=processor.feature_extractor,
            torch_dtype=torch.bfloat16,
            device=device,
            return_timestamps = True, 
            chunk_length_s=30,
            # stride_length_s=5
        )

        self.speaker_segmentation_pipeline = Pipeline.from_pretrained(
        "pyannote/speaker-diarization-3.1",
        use_auth_token="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", 
        ).to(device)



    def get_speaker(self, diarization, time_start, time_end):
        speaker =  diarization.crop(Segment(time_start, time_end)).argmax()
        if speaker == None:
            return "No_speaker"
        else:
            return speaker
        
    def transcribe_with_speaker_detection(self, data_to_transcribe):
        # run the pipeline on an audio file
        diarization = self.speaker_segmentation_pipeline(data_to_transcribe, num_speakers=2)

        transcription = self.speech_recognition_pipe(data_to_transcribe, chunk_length_s=30,
                                                    #  return_timestamps="word", 
            batch_size=1)['chunks']
        
        # print("Just from the oven:", transcription[0])

        for i in range(len(transcription)):
            speaker = self.get_speaker(diarization, transcription[i]['timestamp'][0],  transcription[i]['timestamp'][1])

            try:
                if speaker == 'No_speaker':
                    if i == 0:
                        speaker = 'SPEAKER_00'
                    else: 
                        speaker = transcription[i-1]['speaker']
                transcription[i].update({"speaker": speaker})
            except Exception as e:
                traceback_str = traceback.format_exc()
                print(f"Af error occurred: {traceback_str}")
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