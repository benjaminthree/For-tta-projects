import cv2
import mediapipe as mp

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

# ---------------------------------------------
# Gesture classifier – now also returns pattern
# ---------------------------------------------
def classify_gesture(hand_landmarks):
    lm = hand_landmarks.landmark

    THUMB_TIP = 4
    THUMB_IP = 3

    INDEX_TIP = 8
    INDEX_PIP = 6
    MIDDLE_TIP = 12
    MIDDLE_PIP = 10
    RING_TIP = 16
    RING_PIP = 14
    PINKY_TIP = 20
    PINKY_PIP = 18

    def finger_up(tip, pip):
        return lm[tip].y < lm[pip].y

    thumb_up  = lm[THUMB_TIP].x < lm[THUMB_IP].x
    index_up  = finger_up(INDEX_TIP, INDEX_PIP)
    middle_up = finger_up(MIDDLE_TIP, MIDDLE_PIP)
    ring_up   = finger_up(RING_TIP, RING_PIP)
    pinky_up  = finger_up(PINKY_TIP, PINKY_PIP)

    pattern = [thumb_up, index_up, middle_up, ring_up, pinky_up]
    # convert True/False → 1/0 so it matches your screenshots
    pattern = [1 if f else 0 for f in pattern]

    # name it if you want
    if pattern == [0,0,0,0,0]:
        name = "Fist"
    elif pattern[1:] == [1,1,1,1]:
        name = "Open Palm"
    elif pattern == [1,0,0,0,0]:
        name = "Thumbs Up"
    elif pattern == [0,1,0,0,0]:
        name = "Pointing"
    else:
        name = f"Pattern {pattern}"

    return name, pattern


# ---------------------------------------------
# Main loop – now supports 2 hands + 67 meme
# ---------------------------------------------
def main():
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("❌ Cannot open webcam")
        return

    last_print = ""

    with mp_hands.Hands(
        max_num_hands=2,                     # 🔥 allow 2 hands
        min_detection_confidence=0.6,
        min_tracking_confidence=0.5
    ) as hands:

        print("🎮 Gesture recognition started... (press Q to exit)")

        while True:
            ret, frame = cap.read()
            if not ret:
                print("❌ Camera disconnected")
                break

            frame = cv2.flip(frame, 1)
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = hands.process(rgb)

            overlay_text = "No hand"
            patterns = []

            if results.multi_hand_landmarks:
                # loop through ALL hands
                for idx, hand in enumerate(results.multi_hand_landmarks):
                    # draw skeleton
                    mp_drawing.draw_landmarks(
                        frame,
                        hand,
                        mp_hands.HAND_CONNECTIONS,
                        mp_drawing.DrawingSpec(color=(0,255,0), thickness=2, circle_radius=3),
                        mp_drawing.DrawingSpec(color=(255,0,0), thickness=2)
                    )

                    gesture_name, pattern = classify_gesture(hand)
                    patterns.append(pattern)

                    # show each hand's pattern text slightly lower
                    text = f"H{idx+1}: {gesture_name}"
                    cv2.putText(frame, text, (10, 40 + idx*30),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,0), 2)

                # ------------------------------
                # 67 MEME detection 👇
                # ------------------------------
                if len(patterns) == 2:
                    p1, p2 = patterns

                    ok_pattern   = [0,0,1,1,1]  # your first pic
                    gun_pattern  = [1,0,1,1,1]  # your second pic

                    if ((p1 == ok_pattern and p2 == gun_pattern) or
                        (p1 == gun_pattern and p2 == ok_pattern)):
                        overlay_text = "67 MEME 🤣"
                    else:
                        overlay_text = ""   # no combo
                else:
                    overlay_text = ""       # only 1 hand

            # big text on top for combo / status
            if overlay_text:
                if overlay_text != last_print:
                    print("👉", overlay_text)
                    last_print = overlay_text
                cv2.putText(frame, overlay_text, (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0,255,0), 3)

            cv2.imshow("Hand Gesture Recognition – 67 meme", frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    cap.release()
    cv2.destroyAllWindows()
    print("Camera closed.")


main()
