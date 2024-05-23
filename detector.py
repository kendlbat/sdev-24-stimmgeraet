import pyaudio
import numpy as np
import matplotlib.pyplot as plt
from scipy.fftpack import fft
import datetime as dt
from util import NoteFreq

# Audio stream parameters
CHUNK = 16384  # Higher chunk size for better frequency resolution
FORMAT = pyaudio.paInt16  # Audio format (16-bit PCM)
CHANNELS = 1  # Number of audio channels (1 for mono)
RATE = 44100  # Sampling rate (44.1 kHz)
OVERLAP = 0.5  # Overlap factor

MINIMUM_DETECTION_DURATION = 0.15

# Boolean to enable or disable plotting
ENABLE_PLOT = False

# Initialize PyAudio
p = pyaudio.PyAudio()

# Open the audio stream
stream = p.open(
    format=FORMAT, channels=CHANNELS, rate=RATE, input=True, frames_per_buffer=CHUNK
)

persistant_tone = None


def get_difference_to_detected(wanted_note: float) -> float | None:
    global persistant_tone

    if persistant_tone is None:
        return None

    return wanted_note - persistant_tone


def get_frequency_spectrum(data, rate):
    """Compute the frequency spectrum of the audio data with zero-padding"""
    N = len(data)
    T = 1.0 / rate
    # Zero-padding to increase FFT resolution
    padded_data = np.pad(data, (0, CHUNK * 2 - N), mode="constant")
    yf = fft(padded_data)
    xf = np.linspace(0.0, 1.0 / (2.0 * T), len(padded_data) // 2)
    return xf, 2.0 / len(padded_data) * np.abs(yf[0 : len(padded_data) // 2])


def plot_audio_stream():
    global persistant_tone
    prev_start = dt.datetime.now()
    prev = 0

    """Continuously plot the audio stream frequency spectrum"""
    if ENABLE_PLOT:
        plt.ion()
        fig, ax = plt.subplots()
        x = np.linspace(0.0, RATE / 2.0, (CHUNK * 2) // 2)
        (line,) = ax.plot(x, np.random.rand((CHUNK * 2) // 2))
        # Logarithmic scale for frequency axis
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlim(20, RATE / 2)
        ax.set_ylim(1, 400)

        plt.xlabel("Frequency (Hz)")
        plt.ylabel("Amplitude")

    # Buffer to hold overlapping audio data
    buffer = np.zeros(int(CHUNK * (1 + OVERLAP)), dtype=np.int16)

    selected = NoteFreq.A4

    while True:
        data = np.frombuffer(stream.read(CHUNK), dtype=np.int16)
        buffer[:CHUNK] = buffer[int(CHUNK * OVERLAP) :]
        buffer[int(CHUNK * OVERLAP) :] = data
        xf, yf = get_frequency_spectrum(buffer, RATE)
        if ENABLE_PLOT:
            line.set_xdata(xf)
            line.set_ydata(yf)
            ax.draw_artist(ax.patch)
            ax.draw_artist(line)
            fig.canvas.flush_events()
        # Find the frequency with the maximum amplitude

        loudest_freq = xf[np.argmax(yf)]
        if (loudest_freq > prev * 0.95) and (loudest_freq < prev * 1.05):
            # print(f"Prev detection duration: {dt.datetime.now() - prev_start}")
            if dt.datetime.now() - prev_start > dt.timedelta(
                seconds=MINIMUM_DETECTION_DURATION
            ):
                persistant_tone = prev
                # print(f"Persistent tone detected: {prev:.2f} Hz")

                diff = get_difference_to_detected(selected.value)

                if diff is None:
                    print("No tone detected")
                else:
                    print(
                        f"Detected {selected.name} with a difference of {diff:.2f} Hz"
                    )
                    if selected.value * 0.95 < persistant_tone < selected.value * 1.05:
                        print("Tune is correct")
        else:
            persistant_tone = None
            prev_start = dt.datetime.now()
            prev = loudest_freq
        # print(f"Loudest Frequency: {loudest_freq:.2f} Hz")


try:
    plot_audio_stream()
except Exception as e:
    print(e)
finally:
    # Close the stream gracefully
    stream.stop_stream()
    stream.close()
    p.terminate()
