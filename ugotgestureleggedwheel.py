import cv2
import mediapipe as mp
from ugot import ugot
import time

# ------------------------------
# UGOT SETUP
# ------------------------------
got = ugot.UGOT()
got.initialize("192.168.1.199")   # <-- change if your IP is different

got.wheelleg_start_balancing()
time.sleep(1)
got.wheelleg_set_chassis_height(2)

# ------------------------------
# MEDIAPIPE SETUP
# ------------------------------
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils


# ------------------------------
# HELPER: Count how many fingers are up (for LEFT hand speed)
# Here we only count INDEX, MIDDLE, RING, PINKY.
# Thumb is ignored to make it more reliable.
# ------------------------------
def count_fingers(lm):
    """
    lm = hand_landmarks.landmark
    returns: int number of fingers up (0–4)
    """
    def up(tip, pip):
        # y smaller = higher on screen = finger is up
        return lm[tip].y < lm[pip].y

    index_up  = up(8, 6)
    middle_up = up(12, 10)
    ring_up   = up(16, 14)
    pinky_up  = up(20, 18)

    fingers = [index_up, middle_up, ring_up, pinky_up]
    return sum(1 for f in fingers if f)


# ------------------------------
# RIGHT HAND → DIRECTION 
# We base direction mostly on the FOUR fingers, not the thumb,
# so "1 finger up" works better.
# ------------------------------
def classify_right_hand(lm):
    def up(tip, pip):
        return lm[tip].y < lm[pip].y

    # Thumb: rough check (works ok for most poses)
    thumb_up = lm[4].x < lm[3].x

    index_up  = up(8, 6)
    middle_up = up(12, 10)
    ring_up   = up(16, 14)
    pinky_up  = up(20, 18)

    # Only use index–pinky for pattern
    four = [index_up, middle_up, ring_up, pinky_up]
    four_bits = [1 if f else 0 for f in four]

    # YOUR MAPPING:
    # - Fist          → Left
    # - 1 finger up   → Right
    # - All 4 up      → Forward
    # - Thumb only    → Backward

    # Fist: no fingers (index–pinky) up
    if four_bits == [0, 0, 0, 0]:
        # could be fist (left) or thumb only (backward)
        # If thumb_up too, we treat as Backward
        if thumb_up:
            return "Backward"
        else:
            return "Left"

    # 1 finger up (index only) → Right
    if four_bits == [1, 0, 0, 0]:
        return "Right"

    # All 4 fingers up → Forward
    if four_bits == [1, 1, 1, 1]:
        return "Forward"

    # Thumb only (no index–pinky, thumb up) → Backward
    if four_bits == [0, 0, 0, 0] and thumb_up:
        return "Backward"

    return "None"


# ------------------------------
# LEFT HAND → SPEED CONTROL
# ------------------------------
def speed_from_fingers(fingers):
    """
    Map finger count (0–4) to speed.
    """
    speed_table = {
        0: 0,    # fist → stop
        1: 10,
        2: 20,
        3: 30,
        4: 40,
        5: 50,   # not used, but kept for safety
    }
    return speed_table.get(fingers, 0)


# ------------------------------
# SEND COMMANDS TO UGOT
# ------------------------------
def control_robot(direction, speed):

    if speed == 0 or direction == "None":
        print("🛑 STOP")
        # to stop we can give 0 speeds
        got.wheelleg_move_speed(0, 0)
        got.wheelleg_turn_speed(2, 0)
        return

    if direction == "Forward":
        print(f"⬆ FORWARD speed {speed}")
        got.wheelleg_move_speed(0, speed)

    elif direction == "Backward":
        print(f"⬇ BACKWARD speed {speed}")
        got.wheelleg_move_speed(1, speed)

    elif direction == "Left":
        print(f"⬅ TURN LEFT speed {speed}")
        got.wheelleg_turn_speed(2, speed * 2)

    elif direction == "Right":
        print(f"➡ TURN RIGHT speed {speed}")
        got.wheelleg_turn_speed(3, speed * 2)


# ------------------------------
# MAIN LOOP
# ------------------------------
def main():
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("❌ Cannot open webcam")
        return

    with mp_hands.Hands(
        max_num_hands=2,
        min_detection_confidence=0.6,
        min_tracking_confidence=0.5
    ) as hands:

        print("🤖 TWO-HAND UGOT CONTROL ACTIVE (Press Q to quit)")
        print("➡ Right hand = direction (fist/1 finger/up palm/thumb)")
        print("⬆ Left hand  = speed (0–4 fingers)")

        direction = "None"
        speed = 0

        while True:
            ret, frame = cap.read()
            if not ret:
                print("❌ Camera frame error")
                break

            # Mirror so it feels like a mirror
            frame = cv2.flip(frame, 1)

            # Mediapipe uses RGB
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = hands.process(rgb)

            # Reset each frame
            direction = "None"
            speed = 0
            right_label = "None"
            left_fingers = 0

            if results.multi_hand_landmarks and results.multi_handedness:
                for hand_landmarks, handedness in zip(
                    results.multi_hand_landmarks,
                    results.multi_handedness
                ):
                    label = handedness.classification[0].label  # "Left" or "Right"
                    lm = hand_landmarks.landmark

                    # Draw skeleton on the frame
                    mp_drawing.draw_landmarks(
                        frame,
                        hand_landmarks,
                        mp_hands.HAND_CONNECTIONS
                    )

                    # LEFT hand → speed via finger count
                    if label == "Left":
                        left_fingers = count_fingers(lm)
                        speed = speed_from_fingers(left_fingers)
                        cv2.putText(
                            frame, f"Left fingers: {left_fingers}",
                            (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.8,
                            (255, 255, 0), 2
                        )

                    # RIGHT hand → direction via gesture
                    elif label == "Right":
                        right_label = classify_right_hand(lm)
                        direction = right_label
                        cv2.putText(
                            frame, f"Right: {right_label}",
                            (10, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.8,
                            (0, 255, 0), 2
                        )

            # Apply robot control for this frame
            control_robot(direction, speed)

            # Show overall speed on screen
            cv2.putText(
                frame, f"Speed: {speed}",
                (10, 160), cv2.FONT_HERSHEY_SIMPLEX, 1,
                (0, 0, 255), 2
            )

            cv2.imshow("Two-Hand UGOT Controller", frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    cap.release()
    cv2.destroyAllWindows()
    got.wheelleg_stop_balancing()
    print("❌ Closed.")


if __name__ == "__main__":
    main()
