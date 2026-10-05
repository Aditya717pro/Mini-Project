import speech_recognition as sr
import sounddevice as sd
from scipy.io.wavfile import write
import tempfile
import os


VALID_OBJECTS = {
    "pen": "Pen",
    "pens": "Pen",

    "key": "Keys",
    "keys": "Keys",

    "spectacle": "Spectacles",
    "spectacles": "Spectacles",
    "glasses": "Spectacles",
    "glass": "Spectacles",
}


def record_audio(duration=5, sample_rate=16000):
    """Record audio from the default microphone."""

    print("\n🎤 Listening...")
    print("Speak your command...")

    try:
        audio = sd.rec(
            int(duration * sample_rate),
            samplerate=sample_rate,
            channels=1,
            dtype="int16"
        )

        sd.wait()

        temp_file = tempfile.NamedTemporaryFile(
            suffix=".wav",
            delete=False
        )

        temp_file.close()

        write(
            temp_file.name,
            sample_rate,
            audio
        )

        return temp_file.name

    except Exception as error:
        print("❌ Microphone error:", error)
        return None


def speech_to_text(audio_file):
    """Convert recorded speech into text."""

    recognizer = sr.Recognizer()

    try:
        with sr.AudioFile(audio_file) as source:
            audio_data = recognizer.record(source)

        print("🔄 Recognizing speech...")

        text = recognizer.recognize_google(audio_data)

        print("🗣️ Recognized speech:", text)

        return text.lower()

    except sr.UnknownValueError:
        print("❌ Could not understand the speech.")
        return None

    except sr.RequestError as error:
        print("❌ Speech recognition service error:", error)
        return None

    finally:
        if audio_file and os.path.exists(audio_file):
            os.remove(audio_file)


def extract_target(text):
    """Extract Pen, Spectacles or Keys from the recognized text."""

    if not text:
        return None

    words = text.lower().split()

    for word in words:
        word = word.strip(".,!?")

        if word in VALID_OBJECTS:
            return VALID_OBJECTS[word]

    return None


def get_target_from_voice():
    """Complete voice recognition pipeline."""

    audio_file = record_audio()

    if audio_file is None:
        return None

    text = speech_to_text(audio_file)

    if text is None:
        return None

    target = extract_target(text)

    if target:
        print("🎯 Target object:", target)
        return target

    print("❌ No supported object found.")
    print("Supported objects: Pen, Spectacles, Keys")

    return None


if __name__ == "__main__":
    target = get_target_from_voice()

    if target:
        print("\n✅ Target successfully identified:", target)
    else:
        print("\n⚠️ Please try again.")