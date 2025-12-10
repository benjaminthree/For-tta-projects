import cv2
import mediapipe as mp
import numpy as np
import time
from ugot import ugot

# ------------------------------
# UGOT SETUP (MECANUM)
# ------------------------------
got = ugot.UGOT()
got.initialize("192.168.1.105")   # <--- change IP if needed

# No special start function for mecanum, but good to stop everything first
got.mecanum_stop()
time.sleep(1)

# ------------------------------
# MEDIAPIPE SETUP
# ------------------------------
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

# ------------------------------
# MOVEMENT GAINS (TUNE HERE)
# ------------------------------
MAX_LIN_SPEED = 60   # [0-80] recommended range; higher = faster XY motion
TURN_SPEED    = 80   # [0-280] turning speed for left/right spin
DEAD_ZONE     = 0.1  # around center of screen where robot does not move


# ---------------------------------------------
# Gesture classifier (same as your air-drawing one)
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
    pattern = [1 if f else 0 for f in pattern]

    if pattern == [0,0,0,0,0]: return "Fist"
    if pattern[1:] == [1,1,1,1]: return "Open Palm"
    if pattern == [1,0,0,0,0]: return "Thumbs Up"
    if pattern == [0,1,0,0,0]: return "Pointing"

    return f"Pattern {pattern}"


# ------------------------------
# MAIN: Air drawing + UGOT mecanum control
# ------------------------------
def main():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("❌ Cannot open webcam")
        return

    # Drawing state (keep your colours)
    draw_color_list = [
        (0,   0, 255),  # Red
        (0, 255,   0),  # Green
        (255, 0,   0),  # Blue
        (0, 255, 255),  # Yellow
        (255, 0, 255),  # Magenta
        (255, 255, 255) # White
    ]
    color_index = 0
    current_color = draw_color_list[color_index]

    brush_thickness = 6
    eraser_thickness = 40

    last_color_switch_time = 0
    color_switch_cooldown = 0.5  # seconds

    # Right-hand drawing state
    prev_x, prev_y = None, None
    mode = "idle"  # "draw" / "erase" / "idle"

    # Robot control state
    robot_x = 0
    robot_y = 0
    robot_z = 0

    with mp_hands.Hands(
        max_num_hands=2,
        min_detection_confidence=0.6,
        min_tracking_confidence=0.5
    ) as hands:

        canvas = None  # will create after we know frame size

        print("🎨 Air-drawing + 🤖 UGOT Mecanum started!")
        print("Right hand (camera view):")
        print("  👉 Pointing   = draw & robot follows fingertip")
        print("  ✋ Open Palm  = eraser mode (big brush)")
        print("  👍 Thumbs Up  = switch color")
        print("  ✊ Fist       = CLEAR ALL (canvas) + robot idle")
        print("Left hand:")
        print("  ✋ Open Palm  = turn LEFT")
        print("  ✊ Fist       = turn RIGHT")
        print("Keyboard:")
        print("  S = save drawing as 'air_drawing.png'")
        print("  C = clear canvas")
        print("  Q = quit")

        while True:
            ret, frame = cap.read()
            if not ret:
                print("❌ Camera disconnected")
                break

            frame = cv2.flip(frame, 1)
            h, w, _ = frame.shape

            if canvas is None:
                canvas = np.zeros_like(frame)

            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = hands.process(rgb)

            # Reset robot command each frame
            robot_x = 0
            robot_y = 0
            robot_z = 0

            # To show on screen
            right_gesture_text = "None"
            left_gesture_text  = "None"

            # For fingertip position (right hand)
            right_finger_cx = None
            right_finger_cy = None

            now = time.time()

            if results.multi_hand_landmarks and results.multi_handedness:
                for hand_landmarks, handedness in zip(
                    results.multi_hand_landmarks,
                    results.multi_handedness
                ):
                    label = handedness.classification[0].label  # "Left" or "Right"
                    gesture = classify_gesture(hand_landmarks)

                    # Draw skeleton
                    mp_drawing.draw_landmarks(
                        frame,
                        hand_landmarks,
                        mp_hands.HAND_CONNECTIONS,
                        mp_drawing.DrawingSpec(color=(0,255,0), thickness=2, circle_radius=3),
                        mp_drawing.DrawingSpec(color=(255,0,0), thickness=2)
                    )

                    # -----------------------
                    # RIGHT HAND: drawing + translation
                    # -----------------------
                    if label == "Right":
                        right_gesture_text = gesture

                        # Get index fingertip position
                        INDEX_TIP = 8
                        index_tip = hand_landmarks.landmark[INDEX_TIP]
                        cx = int(index_tip.x * w)
                        cy = int(index_tip.y * h)
                        right_finger_cx, right_finger_cy = cx, cy

                        # Visual marker for fingertip
                        cv2.circle(frame, (cx, cy), 8, (255, 255, 255), -1)

                        # Gesture → drawing mode / color / clear
                        if gesture == "Thumbs Up":
                            if now - last_color_switch_time > color_switch_cooldown:
                                color_index = (color_index + 1) % len(draw_color_list)
                                current_color = draw_color_list[color_index]
                                last_color_switch_time = now
                                print(f"🎨 Color changed to index {color_index}, BGR={current_color}")
                            mode = "idle"
                            prev_x, prev_y = None, None

                        elif gesture == "Open Palm":
                            mode = "erase"

                        elif gesture == "Pointing":
                            mode = "draw"

                        elif gesture == "Fist":
                            # CLEAR ALL
                            canvas[:] = 0
                            mode = "idle"
                            prev_x, prev_y = None, None
                            print("🧹 Canvas cleared by RIGHT FIST")

                        # Drawing on canvas
                        if mode in ("draw", "erase") and right_finger_cx is not None:
                            if prev_x is None or prev_y is None:
                                prev_x, prev_y = right_finger_cx, right_finger_cy

                            if mode == "draw":
                                cv2.line(
                                    canvas,
                                    (prev_x, prev_y),
                                    (right_finger_cx, right_finger_cy),
                                    current_color,
                                    brush_thickness
                                )
                            elif mode == "erase":
                                cv2.line(
                                    canvas,
                                    (prev_x, prev_y),
                                    (right_finger_cx, right_finger_cy),
                                    (0, 0, 0),
                                    eraser_thickness
                                )

                            prev_x, prev_y = right_finger_cx, right_finger_cy
                        else:
                            prev_x, prev_y = None, None

                    # -----------------------
                    # LEFT HAND: turning
                    # -----------------------
                    elif label == "Left":
                        left_gesture_text = gesture

                        if gesture == "Open Palm":
                            # Turn left
                            robot_z = -TURN_SPEED
                        elif gesture == "Fist":
                            # Turn right
                            robot_z = TURN_SPEED
                        else:
                            # No turn
                            pass

            # ---------------------------------
            # RIGHT HAND → ROBOT TRANSLATION (follow fingertip)
            # Only when right hand is Pointing (drawing)
            # ---------------------------------
            if right_finger_cx is not None and right_gesture_text == "Pointing":
                # Normalise to [-1, 1]
                norm_x = (right_finger_cx - w / 2) / (w / 2)
                norm_y = (right_finger_cy - h / 2) / (h / 2)

                # Dead zone near center
                if abs(norm_x) < DEAD_ZONE:
                    norm_x = 0
                if abs(norm_y) < DEAD_ZONE:
                    norm_y = 0

                # Map to mecanum speeds
                robot_x = int(norm_x * MAX_LIN_SPEED)    # left/right slide
                robot_y = int(-norm_y * MAX_LIN_SPEED)   # forward/back (screen up = forward)

            # ---------------------------------
            # SEND COMMAND TO MECANUM ROBOT
            # ---------------------------------
            if robot_x == 0 and robot_y == 0 and robot_z == 0:
                got.mecanum_stop()
            else:
                # x_speed, y_speed, z_speed
                got.mecanum_move_xyz(robot_x, robot_y, robot_z)

            # ---------------------------------
            # Combine canvas and camera frame
            # ---------------------------------
            gray_canvas = cv2.cvtColor(canvas, cv2.COLOR_BGR2GRAY)
            _, mask = cv2.threshold(gray_canvas, 10, 255, cv2.THRESH_BINARY)
            mask_inv = cv2.bitwise_not(mask)

            frame_bg = cv2.bitwise_and(frame, frame, mask=mask_inv)
            canvas_fg = cv2.bitwise_and(canvas, canvas, mask=mask)
            output = cv2.add(frame_bg, canvas_fg)

            # HUD text
            cv2.putText(
                output,
                f"Right: {right_gesture_text} | Left: {left_gesture_text} | Mode: {mode}",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 255, 0),
                2
            )

            # Color indicator
            cv2.rectangle(output, (10, 40), (60, 90), current_color, -1)
            cv2.putText(output, "Color", (10, 110),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1)

            # Robot speeds display
            cv2.putText(
                output,
                f"Robot XY: ({robot_x}, {robot_y}) Z: {robot_z}",
                (10, 140),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 0, 255),
                2
            )

            cv2.imshow("Air Drawing + UGOT Mecanum", output)

            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('c'):
                canvas = np.zeros_like(canvas)
                print("🧹 Canvas cleared by keyboard")
            elif key == ord('s'):
                cv2.imwrite("air_drawing.png", canvas)
                print("💾 Saved drawing as air_drawing.png")

    cap.release()
    cv2.destroyAllWindows()
    got.mecanum_stop()
    print("Camera + UGOT closed.")


if __name__ == "__main__":
    main()
