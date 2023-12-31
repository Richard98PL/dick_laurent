import numpy as np

from tensorflow.keras import models

from recording_helper import record_audio, terminate
from tf_helper import preprocess_audiobuffer

import os

folder_location = 'data/mini_speech_commands'  # Replace with the actual folder location

folders = []

for root, dirnames, filenames in os.walk(folder_location):
    for dirname in dirnames:
        folder_name = os.path.basename(os.path.join(root, dirname))
        folders.append(folder_name)

print(folders)

commands = folders

loaded_model = models.load_model("saved_model")

def predict_mic():
    audio = record_audio()
    spec = preprocess_audiobuffer(audio)
    prediction = loaded_model(spec)
    label_pred = np.argmax(prediction, axis=1)
    command = commands[label_pred[0]]
    print("Predicted label:", command)
    return command
if __name__ == "__main__":
    from turtle_helper import move_turtle
    while True:
        command = predict_mic()
        if command == 'x':
            command = ''
        move_turtle(command)
        if command == "stop":
            terminate()
            break