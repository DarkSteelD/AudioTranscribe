import os
from pydub import AudioSegment
from transformers import pipeline, WhisperProcessor
from transformers import WhisperForConditionalGeneration
# Параметры и настройки
processor = WhisperProcessor.from_pretrained("openai/whisper-large")
model = WhisperForConditionalGeneration.from_pretrained("openai/whisper-large")

forced_decoder_ids = processor.get_decoder_prompt_ids(language="ru", task="transcribe")

transcriber = pipeline(
    task="automatic-speech-recognition",
    model=model,
    tokenizer=processor.tokenizer,
    feature_extractor=processor.feature_extractor,
    generate_kwargs={"forced_decoder_ids": forced_decoder_ids},
)

output_file = "ans.txt"


def transcribe_acc_file(acc_file_path):
    print(f"Transcribing {acc_file_path}...")
    # Конвертируем ACC в WAV
    audio = AudioSegment.from_file(acc_file_path, format="aac")
    wav_path = os.path.splitext(acc_file_path)[0] + ".wav"
    audio.export(wav_path, format="wav")

    # Распознаем текст из файла WAV
    result = transcriber(wav_path)
    text = result['text']
    with open(output_file, "a") as f:
        f.write(text + "\n\n")
    print("Transcribed text:", text)
    print(f"Summary added from {acc_file_path}")
    print(f"Transcription saved to {output_file}")


transcribe_acc_file("/home/dark/Documents/GitHub/AudioTranscribe/1.aac")
