import asyncio
import tempfile
import torchaudio
import torch
import transformers
import os
import discord
import soundfile as sf
from datasets import load_dataset

# Load the Facebook AI's pre-trained wav2vec2-large-xlsr-53 model for speech-to-text transcription
model_stt = transformers.Wav2Vec2ForCTC.from_pretrained("facebook/wav2vec2-large-xlsr-53")
tokenizer_stt = transformers.Wav2Vec2Tokenizer.from_pretrained("facebook/wav2vec2-large-xlsr-53")

# Load the Twitter-RoBERTa model for emotion detection
model_emotion = transformers.AutoModelForSequenceClassification.from_pretrained("cardiffnlp/twitter-roberta-base-emotion")
tokenizer_emotion = transformers.AutoTokenizer.from_pretrained("cardiffnlp/twitter-roberta-base-emotion")
emotion_labels = load_dataset("emotion")['train'].features['label'].names

client = discord.Client(intents=discord.Intents.all())

async def disconnect(vc):
    await vc.disconnect()

@client.event
async def on_voice_state_update(member, before, after):
    # Check if the member is the bot itself or if they are not in a voice channel
    if member.id == client.user.id or (before.channel == after.channel):
        return

    # Check if the member joined a voice channel
    if before.channel is None:
        # Create a temporary file to store the audio
        temp_file = tempfile.NamedTemporaryFile(delete=False)

        # Start recording the audio
        vc = await after.channel.connect()
        vc.play(discord.PCMVolumeTransformer(discord.FFmpegPCMAudio(temp_file.name), volume=0.5), after=lambda e: asyncio.run_coroutine_threadsafe(disconnect(vc), client.loop))


        # Wait for the member to finish talking
        while vc.is_playing():
            await asyncio.sleep(1)

        # Stop recording
        vc.stop()

        # Disconnect from the voice channel
        await vc.disconnect()

        # Convert the audio file to a format that torchaudio can load
        waveform, sample_rate = sf.read(temp_file.name)
        input_values = torchaudio.transforms.Resample(sample_rate, 16000)(torch.from_numpy(waveform)).squeeze(0)
        input_dict_stt = tokenizer_stt(input_values, return_tensors='pt', padding=True)

        # Transcribe the speech to text
        with torch.no_grad():
            logits_stt = model_stt(input_dict_stt.input_values, attention_mask=input_dict_stt.attention_mask).logits
            predicted_class_stt = torch.argmax(logits_stt, dim=-1).squeeze().cpu().numpy()
            predicted_text = tokenizer_stt.batch_decode(predicted_class_stt)[0]

        # Detect the emotion in the text
        input_dict_emotion = tokenizer_emotion(predicted_text, return_tensors='pt', padding=True)
        with torch.no_grad():
            logits_emotion = model_emotion(**input_dict_emotion)[0]
            predicted_class_emotion = torch.argmax(logits_emotion).item()
            predicted_emotion = emotion_labels[predicted_class_emotion]

        # Write the transcription and predicted emotion to a text file
        with open('transcriptions.txt', 'a') as file:
            file.write(f'{member.name}: {predicted_text}\n')
            file.write(f'Emotion: {predicted_emotion}\n')

        # Delete the audio file
        os.remove(temp_file.name)

client.run("your discord token")

