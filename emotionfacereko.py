import cv2

# Try both possible imports (depends on your version of fer)
try:
    from fer import FER
except ImportError:
    from fer.fer import FER

detector = FER(mtcnn=True)

EMOJI_MAP = {
    "happy": ":)))",
    "angry": "8[",
    "sad": ":(",
    "neutral": ": |",
    "surprise": ":)",
    "fear": ":')",
    "disgust": ": {}",
}

def emotion_to_emoji(name):
    return EMOJI_MAP.get(name, "😐")


def main():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Cannot open camera")
        return

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.flip(frame, 1)

        # SAFE: top_emotion may return (None, None)
        try:
            emotion_name, score = detector.top_emotion(frame)
        except:
            emotion_name, score = None, None

        if emotion_name is None:
            emoji = "😐"
            label = "No face detected"
        else:
            emoji = emotion_to_emoji(emotion_name)
            # score is sometimes None → FIX HERE
            if score is None:
                label = f"{emotion_name} (?)"
            else:
                label = f"{emotion_name} ({score:.2f})"

        # UI box
        cv2.rectangle(frame, (0, 0), (350, 80), (0, 0, 0), -1)

        cv2.putText(frame, "Emoji Recognition", (10, 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        cv2.putText(frame, label, (10, 55),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        cv2.putText(frame, emoji, (260, 65),
                    cv2.FONT_HERSHEY_SIMPLEX, 2.0, (255, 255, 255), 3)

        cv2.imshow("Emoji Recognition", frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
