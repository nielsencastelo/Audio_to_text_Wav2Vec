# -*- coding: utf-8 -*-
"""
Created on Wed Jan 26 09:53:03 2022

@author: Nielsen Castelo Damasceno Dantas
"""

import torch
import librosa
import numpy as np
from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor
from scipy.io import wavfile


LANG = "joorock12/wav2vec2-large-xlsr-portuguese"
MODEL = "joorock12/wav2vec2-large-xlsr-portuguese"

processor = Wav2Vec2Processor.from_pretrained(LANG)
model = Wav2Vec2ForCTC.from_pretrained(MODEL)


file_name = 'audio/audio.wav'

data = wavfile.read(file_name)
framerate = data[0]
sounddata = data[1]
time = np.arange(0,len(sounddata))/framerate

input_audio, _ = librosa.load(file_name, sr=16000)

inputs = processor(input_audio, sampling_rate=16_000, return_tensors="pt", padding=True)

with torch.no_grad():
    logits = model(inputs.input_values, attention_mask=inputs.attention_mask).logits
    
    
predicted_ids = torch.argmax(logits, dim=-1)
predicted_sentences = processor.batch_decode(predicted_ids)

print(predicted_sentences)