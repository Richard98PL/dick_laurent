import tkinter as tk
import threading
import sys
import sounddevice as sd
import numpy as np

sample_rate = 16000
duration = 0.1
threshold = 8
min_duration = 0.5
# Set up the audio stream
start_time = None
print_voice = False
recorded_audio = np.empty((0, 1), dtype=np.float32)


def geTimeDiff(indata):
    global start_time, print_voice, recorded_audio
    xd = time.time() - start_time
    # print(xd)
    recorded_audio = np.concatenate((recorded_audio, indata))
    return xd


def audio_callback(indata, frames, timex, status, hearingEvent, listeningEvent):
    global start_time, print_voice, recorded_audio
    # print('start indata')
    # print(indata)
    # print('end indata')

    threshold = 2

    volume_norm = np.linalg.norm(indata) * 10
    # print(volume_norm)
    if volume_norm > threshold:
        print("voice detected")
        listeningEvent.set()
        hearingEvent.clear()
        if not print_voice:
            start_time = time.time()
            recorded_audio = np.empty((0, indata.shape[1]), dtype=np.float32)
        print_voice = True
        start_time = time.time()
        recorded_audio = np.concatenate((recorded_audio, indata), axis=0)

    elif print_voice and geTimeDiff(indata) >= min_duration:
        print("voice detected" + str(volume_norm))
        print("kurwa" + str(time.time() - start_time))
        print_voice = False
        listeningEvent.clear()

        wavio.write("recorded_audio.wav", recorded_audio, sample_rate, sampwidth=2)
        recorded_audio = np.empty((0, indata.shape[1]), dtype=np.float32)
        print("Audio saved.")
        hearingEvent.set()

    # if rms_db != THRESHOLD_DB:
    #     if hearingEvent.is_set():
    #         hearingEvent.clear()


def on_close():
    global root
    print("im on close function.")
    # global instance
    # if player is not None:
    #     player.stop()
    #     player.get_media().release()
    #     player.release()
    #     player.get_instance().release()
    if speech_thread.is_alive():
        print("speech thread is alive..")
        speech_thread.terminate()
    root.destroy()
    sys.exit()


# Define the window size and line width
window_width = 415
window_height = 221
line_width = 20

# Initialize tkinter
root = tk.Tk()
root.attributes("-topmost", True)
root.title("PROJECT DICK LAURENT IS DEAD")
root.geometry(f"{window_width}x{window_height}")  # Set the window size

root.protocol("WM_DELETE_WINDOW", on_close)

# Create the canvas
canvas = tk.Canvas(root, width=window_width, height=window_height, highlightthickness=0)
canvas.pack()

# Add the background chessboard pattern
for x in range(0, window_width, line_width):
    for y in range(0, window_height, line_width):
        color = "#FFFFFF" if (x + y) // line_width % 2 == 0 else "#ebebeb"
        canvas.create_rectangle(
            x, y, x + line_width, y + line_width, fill=color, outline=""
        )

# Write "DICK LAURENT" in red pixels with shadow
text = "DICK LAURENT"
for i, char in enumerate(text):
    print(char + " -> " + str(ord(char)))

text_size = 40
total_text_width = (
    len(text) - 1
) * text_size + text_size * 0.8  # Adjusted width calculation for spacing
text_x = (
    window_width - total_text_width
) // 2 + text_size * 0.1  # Adjusted x-coordinate for centering
text_y = window_height // 2 - text_size // 2

# Draw the shadow text
shadow_color = "#333333"
shadow_offset = 2
prev_char = ""
initial_fix = 60
for i, char in enumerate(text):
    char_x = (
        initial_fix + text_x + i * (text_size * 0.8)
    )  # Adjusted spacing between characters
    if prev_char in ["D"]:
        char_x -= text_size * 0.9 * 0.2  # Reduce the spacing for 'D' and 'I'
    if prev_char in ["I"]:
        char_x -= text_size * 1.35 * 0.2  # Reduce the spacing for 'D' and 'I'
    if prev_char in ["C"]:
        char_x -= text_size * 0.8 * 0.2  # Reduce the spacing for 'D' and 'I'
    if len(char) != 0 and ord(char) == 32:
        initial_fix = 34
    canvas.create_text(
        char_x + shadow_offset,
        text_y + shadow_offset,
        text=char,
        fill=shadow_color,
        font=("Arial", text_size, "bold"),
    )
    prev_char = char

# Draw the main text
text_color = "#880808"
prev_char = ""
initial_fix = 60

for i, char in enumerate(text):
    char_x = (
        initial_fix + text_x + i * (text_size * 0.8)
    )  # Adjusted spacing between characters
    if prev_char in ["D"]:
        char_x -= text_size * 0.9 * 0.2  # Reduce the spacing for 'D' and 'I'
    if prev_char in ["I"]:
        char_x -= text_size * 1.35 * 0.2  # Reduce the spacing for 'D' and 'I'
    if prev_char in ["C"]:
        char_x -= text_size * 0.8 * 0.2  # Reduce the spacing for 'D' and 'I'
    if len(char) != 0 and ord(char) == 32:
        initial_fix = 34
    canvas.create_text(
        char_x, text_y, text=char, fill=text_color, font=("Arial", text_size, "bold")
    )
    prev_char = char

# Define the LED positions with increased spacing
led_x1 = window_width - 40
led_x2 = window_width - 140
led_x3 = window_width - 240
led_y = text_y + text_size + 50  # Increased distance from text

# Create the LED objects
led_radius = 10
led_shadow_offset = 3
leds = []
led_positions = [(led_x1, led_y), (led_x2, led_y)]  # Store the LED positions
for led_x, led_y in led_positions:
    led_shadow = canvas.create_oval(
        led_x - led_radius + led_shadow_offset,
        led_y - led_radius + led_shadow_offset,
        led_x + led_radius + led_shadow_offset,
        led_y + led_radius + led_shadow_offset,
        fill="black",
    )
    led = canvas.create_oval(
        led_x - led_radius,
        led_y - led_radius,
        led_x + led_radius,
        led_y + led_radius,
        fill="#990a00",
    )
    leds.append((led, led_shadow))

# Define the text for each LED
led_texts = [
    "CONTROL",
    "LISTENING",
    # Add more texts as needed
]

# Create the text objects for LEDs
led_text_objects = []
for i, (led, _) in enumerate(leds):
    text_x = led_positions[i][0] + 3
    text_y = (
        led_positions[i][1] + 2 * led_radius + 5
    )  # Adjust the y-coordinate for text placement
    text_obj = canvas.create_text(
        text_x, text_y, text=led_texts[i], fill="#000015", font=("Arial", 10, "bold")
    )
    led_text_objects.append(text_obj)

powerSwap = True


def doPowerSwap():
    global powerSwap
    powerSwap = not powerSwap
    return powerSwap


listeningEvent = None
hearingEvent = None
hearingEventCaptured = False


def update_power_led():
    if doPowerSwap():
        led_color = "#21c700"
    else:
        led_color = "#990a00"

    canvas.itemconfig(leds[0][0], fill=led_color)

    swap_listening_led()
    global hearingEvent
    global hearingEventCaptured
    if hearingEvent is not None and hearingEvent.is_set():
        hearingEventCaptured = True

    root.after(100, update_power_led)  # Update every 1 second


listenSwap = False


def doListenSwap():
    global listenSwap
    listenSwap = not listenSwap
    return listenSwap


def swap_listening_led():
    global listeningEvent
    if listeningEvent is not None and listeningEvent.is_set():
        led_color = "#21c700"
    else:
        led_color = "#990a00"

    canvas.itemconfig(leds[1][0], fill=led_color)


import speech_recognition as sr
import requests
from bs4 import BeautifulSoup
import webbrowser
import time
from pydub import AudioSegment
import string
import random
import shutil


def generate_random_name(length):
    letters = string.ascii_lowercase
    return "".join(random.choice(letters) for _ in range(length))


def listen(listeningEvent, hearingEvent):
    while True:
        if hearingEvent is not None and hearingEvent.is_set():
            print("no i jestem tutaj")

            mic_index = 0
            mic_list = sr.Microphone.list_microphone_names()

            for i, mic_name in enumerate(mic_list):
                # print(f"{i}: {mic_name}")
                if "Headset Microphone" in mic_name:
                    # print('found it!')
                    mic_index = i

            try:
                audio_file = "recorded_audio.wav"
                with wave.open(audio_file, "rb") as wav:
                    # Get the number of frames in the audio file
                    num_frames = wav.getnframes()

                    # Get the frame rate (sample rate) of the audio file
                    frame_rate = wav.getframerate()

                    # Calculate the duration of the audio file in seconds
                    audio_length = num_frames / frame_rate

                print("Audio Length:", audio_length, "seconds")
                if audio_length < 0.3:
                    print("continue")
                    hearingEvent.clear()
                    continue

                text = ""
                with sr.AudioFile(audio_file) as source:
                    recognizer = sr.Recognizer()
                    audio = recognizer.record(source)  # Read the entire audio file

                    try:
                        text = recognizer.recognize_google(audio, language="pl-PL")
                    except Exception as e:
                        print(e)
                        pass

                    print("Transcribed Text:")
                    print(text)
                
                #text = 'youtube toxic'
                if "youtu" in str(text).lower() or True:
                    text = text.lower()
                    text = text.replace("włącz mi teraz", "")
                    text = text.replace("włącz mi teraz na youtubie", "")
                    text = text.replace("włącz mi teraz na youtube", "")
                    text = text.replace("włącz mi teraz na youtubie proszę", "")
                    text = text.replace("włącz mi teraz na youtube proszę", "")
                    text = text.replace("włącz mi teraz proszę", "")

                    text = text.replace("włącz mi na youtubie", "")
                    text = text.replace("włącz mi na youtubie proszę", "")
                    text = text.replace("włącz mi na youtube", "")
                    text = text.replace("włącz mi na youtube proszę", "")

                    text = text.replace("hej dick", "")
                    text = text.replace("hej dik", "")
                    text = text.replace("heidi", "")
                    text = text.replace("youtube", "")

                    text = text.replace("włącz mi", "")
                    text = text.replace("a teraz", "")
                    text = text.replace("proszę", "")

                    if text.endswith("proszę"):
                        text = text[: -len("proszę")]

                    text = text.replace("  ", "")
                    text = text.strip()

                    print(text)
                    open_first_youtube_video(text)
            except sr.UnknownValueError:
                print("Speech recognition could not understand audio")
            except sr.RequestError as e:
                print(
                    "Could not request results from Google Speech Recognition service; {0}".format(
                        e
                    )
                )
            finally:
                hearingEvent.clear()
                print("end of hearing.")
                pass


def extract_video_id(response_text):
    index = response_text.find("videoId")
    print(index)
    response_text = response_text[int(index) : int(index) + 100]
    print(response_text)
    index = int(index)

    if index != -1:
        colon = response_text.find(":")
        response_text = response_text[colon + 2 : len(response_text) - 1]
        quote = response_text.find('"')
        response_text = response_text[0:quote]

        return response_text
    else:
        return None


import vlc
import yt_dlp


def format_selector(ctx):
    formats = ctx.get("formats")[::-1]
    best_video = next(
        f for f in formats if f["vcodec"] != "none" and f["acodec"] != "none"
    )

    audio_ext = {"mp4": "m4a", "webm": "webm"}[best_video["ext"]]

    best_audio = next(
        f
        for f in formats
        if (f["acodec"] != "none" and f["vcodec"] == "none" and f["ext"] == audio_ext)
    )

    yield {
        "format_id": f'{best_video["format_id"]}+{best_audio["format_id"]}',
        "ext": best_video["ext"],
        "requested_formats": [best_video, best_audio],
        "protocol": f'{best_video["protocol"]}+{best_audio["protocol"]}',
    }


instance = None
player = None

def open_first_youtube_video(search_query):
    base_url = "https://www.youtube.com/results?search_query="
    query = search_query.replace(" ", "+")
    url = base_url + query

    response = requests.get(url)
    video_id = extract_video_id(response.text)

    if video_id:
        url = "https://www.youtube.com/watch?v=" + video_id

        with yt_dlp.YoutubeDL({"format": format_selector}) as ydl:
            info = ydl.extract_info(url, download=False)
            # print(info['requested_formats'])

            format_id = info["requested_formats"][0]["format_id"]

            formats = info["formats"]
            # print(f"Found {len(formats)} formats")
            for i, format in enumerate(formats):
                if format["format_id"] == format_id:
                    # print(format_id)
                    url = format["url"]

        # print(url)
        global instance
        if instance is None:
            instance = vlc.Instance()

        global player
        if player is None:
            player = instance.media_player_new()

        # Create a media object with the stream URL
        media = instance.media_new(url)

        # Set the media player's media

        if sys.platform == "darwin" and False:
            app = QtWidgets.QApplication(sys.argv)
            player = Player()
            player.show()
            player.open_file(media)

            sys.exit(app.exec_())
        else:
            media = instance.media_new(url)

            # Set the media player's media
            player.set_media(media)
            player.video_set_scale(0.75)

            # Start playing the video
            player.play()
    
    else:
        print("No videos found for the given search query.")


update_power_led()

import platform
import os
import sys

from PyQt5 import QtWidgets, QtGui, QtCore
import vlc
class Player(QtWidgets.QMainWindow):
    """A simple Media Player using VLC and Qt
    """

    def __init__(self, master=None):
        QtWidgets.QMainWindow.__init__(self, master)
        self.setWindowTitle("Media Player")

        # Create a basic vlc instance
        self.instance = vlc.Instance()

        self.media = None

        # Create an empty vlc media player
        self.mediaplayer = self.instance.media_player_new()

        self.create_ui()
        self.is_paused = False

    def create_ui(self):
        """Set up the user interface, signals & slots
        """
        self.widget = QtWidgets.QWidget(self)
        self.setCentralWidget(self.widget)

        # In this widget, the video will be drawn
        if platform.system() == "Darwin": # for MacOS
            self.videoframe = QtWidgets.QMacCocoaViewContainer(0)
        else:
            self.videoframe = QtWidgets.QFrame()

        self.palette = self.videoframe.palette()
        self.palette.setColor(QtGui.QPalette.Window, QtGui.QColor(0, 0, 0))
        self.videoframe.setPalette(self.palette)
        self.videoframe.setAutoFillBackground(True)

        self.positionslider = QtWidgets.QSlider(QtCore.Qt.Horizontal, self)
        self.positionslider.setToolTip("Position")
        self.positionslider.setMaximum(1000)
        self.positionslider.sliderMoved.connect(self.set_position)
        self.positionslider.sliderPressed.connect(self.set_position)

        self.hbuttonbox = QtWidgets.QHBoxLayout()
        self.playbutton = QtWidgets.QPushButton("Play")
        self.hbuttonbox.addWidget(self.playbutton)
        self.playbutton.clicked.connect(self.play_pause)

        self.stopbutton = QtWidgets.QPushButton("Stop")
        self.hbuttonbox.addWidget(self.stopbutton)
        self.stopbutton.clicked.connect(self.stop)

        self.hbuttonbox.addStretch(1)
        self.volumeslider = QtWidgets.QSlider(QtCore.Qt.Horizontal, self)
        self.volumeslider.setMaximum(100)
        self.volumeslider.setValue(self.mediaplayer.audio_get_volume())
        self.volumeslider.setToolTip("Volume")
        self.hbuttonbox.addWidget(self.volumeslider)
        self.volumeslider.valueChanged.connect(self.set_volume)

        self.vboxlayout = QtWidgets.QVBoxLayout()
        self.vboxlayout.addWidget(self.videoframe)
        self.vboxlayout.addWidget(self.positionslider)
        self.vboxlayout.addLayout(self.hbuttonbox)

        self.widget.setLayout(self.vboxlayout)

        menu_bar = self.menuBar()

        # File menu
        file_menu = menu_bar.addMenu("File")

        # Add actions to file menu
        open_action = QtWidgets.QAction("Load Video", self)
        close_action = QtWidgets.QAction("Close App", self)
        file_menu.addAction(open_action)
        file_menu.addAction(close_action)

        #open_action.triggered.connect(self.open_file)
        close_action.triggered.connect(sys.exit)

        self.timer = QtCore.QTimer(self)
        self.timer.setInterval(100)
        self.timer.timeout.connect(self.update_ui)

    def play_pause(self):
        """Toggle play/pause status
        """
        if self.mediaplayer.is_playing():
            self.mediaplayer.pause()
            self.playbutton.setText("Play")
            self.is_paused = True
            self.timer.stop()
        else:
            if self.mediaplayer.play() == -1:
                #self.open_file()
                return

            self.mediaplayer.play()
            self.playbutton.setText("Pause")
            self.timer.start()
            self.is_paused = False

    def stop(self):
        """Stop player
        """
        self.mediaplayer.stop()
        self.playbutton.setText("Play")

    def open_file(self, vlc_media):
        self.mediaplayer.set_media(vlc_media)


    def set_volume(self, volume):
        """Set the volume
        """
        self.mediaplayer.audio_set_volume(volume)

    def set_position(self):
        """Set the movie position according to the position slider.
        """

        # The vlc MediaPlayer needs a float value between 0 and 1, Qt uses
        # integer variables, so you need a factor; the higher the factor, the
        # more precise are the results (1000 should suffice).

        # Set the media position to where the slider was dragged
        self.timer.stop()
        pos = self.positionslider.value()
        self.mediaplayer.set_position(pos / 1000.0)
        self.timer.start()

    def update_ui(self):
        """Updates the user interface"""

        # Set the slider's position to its corresponding media position
        # Note that the setValue function only takes values of type int,
        # so we must first convert the corresponding media position.
        media_pos = int(self.mediaplayer.get_position() * 1000)
        self.positionslider.setValue(media_pos)

        # No need to call this function if nothing is played
        if not self.mediaplayer.is_playing():
            self.timer.stop()

            # After the video finished, the play button stills shows "Pause",
            # which is not the desired behavior of a media player.
            # This fixes that "bug".
            if not self.is_paused:
                self.stop()

THRESHOLD_DB = 25  # Adjust this value to set the desired threshold in decibels
lastSavedTime = None
import time
import soundfile as sf
import wavio
import os
import pyaudio
import wave


import pydub


from multiprocessing import Process, Queue, Event
def qt_starter(queue):
    queue.put('test')
        

if __name__ == "__main__":
    window_id = None

    if sys.platform == "darwin" and False:
        window_id_queue = Queue()

        qt_thread = Process(
            target=qt_starter,
            args=(window_id_queue,),
        )
        qt_thread.start()

        window_id = window_id_queue.get()  # Retrieve the window_id from the queue
    print("im here?")
    # Start the audio input stream
    hearingEvent = Event()
    listeningEvent = Event()
    stream = sd.InputStream(
        channels=1,
        samplerate=sample_rate,
        callback=lambda indata, frames, time, status: audio_callback(
            indata, frames, time, status, hearingEvent, listeningEvent
        ),
    )
    stream.start()

    hearingEvent.clear()
    speech_thread = Process(
        target=listen,
        args=(
            listeningEvent,
            hearingEvent,

        ),
    )
    speech_thread.start()

    root.mainloop()
    while True:
        pass
        sd.sleep(int(duration * 2000))
