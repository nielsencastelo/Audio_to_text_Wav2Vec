# -*- coding: utf-8 -*-
"""
Created on Wed Jan 26 09:53:03 2022

@author: Nielsen Castelo Damasceno Dantas
"""

import torch
import librosa
import numpy as np
from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor


LANG = "lgris/wav2vec2-large-xlsr-open-brazilian-portuguese-v2"
MODEL = "lgris/wav2vec2-large-xlsr-open-brazilian-portuguese-v2"

processor = Wav2Vec2Processor.from_pretrained(LANG)
model = Wav2Vec2ForCTC.from_pretrained(MODEL)


file_name = 'audio/audio.wav'

input_audio, _ = librosa.load(file_name, sr=16000)

inputs = processor(input_audio, sampling_rate=16_000, return_tensors="pt", padding=True)

with torch.no_grad():
    logits = model(inputs.input_values, attention_mask=inputs.attention_mask).logits
    
    
predicted_ids = torch.argmax(logits, dim=-1)
predicted_sentences = processor.batch_decode(predicted_ids)

print(predicted_sentences)