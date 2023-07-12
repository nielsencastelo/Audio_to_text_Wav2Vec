# -*- coding: utf-8 -*-
"""
Created on Wed Jan 26 09:53:03 2022

@author: Nielsen Castelo Damasceno Dantas
"""

from transformers import pipeline

transcriber = pipeline(
  "automatic-speech-recognition", 
  model="jonatasgrosman/whisper-large-pt-cv11"
)

transcriber.model.config.forced_decoder_ids = (
  transcriber.tokenizer.get_decoder_prompt_ids(
    language="pt", 
    task="transcribe"
  )
)


file_name = 'audio/audio.wav'

transcription = transcriber(f"{file_name}")

