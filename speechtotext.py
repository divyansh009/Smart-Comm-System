import speech_recognition as sr

def speechtotext():
    record=sr.Recognizer()
    with sr.Microphone() as s:
        print("Listening..")
        record.pause_threshold = 0.6
        audio = record.listen(s)

    try:
        ask=record.recognize_google(audio,language='en')
        print(f"You said : {ask}")
    except Exception:
        print("Say that again...")
        return ""

    return ask

speechtotext()

