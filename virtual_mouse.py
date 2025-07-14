import cv2
import mediapipe as mp
import pyautogui

# Initialize hand detection
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=1)
mp_draw = mp.solutions.drawing_utils

# Screen size
screen_width, screen_height = pyautogui.size()

# Webcam feed
cap = cv2.VideoCapture(0)

while True:
    success, frame = cap.read()
    if not success:
        break

    # Flip for natural movement and convert to RGB
    frame = cv2.flip(frame, 1)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    result = hands.process(rgb_frame)

    if result.multi_hand_landmarks:
        for hand_landmarks in result.multi_hand_landmarks:
            mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            landmarks = hand_landmarks.landmark

            # Index finger tip
            index_tip = landmarks[8]
            x = int(index_tip.x * frame.shape[1])
            y = int(index_tip.y * frame.shape[0])

            # Move mouse
            screen_x = int(index_tip.x * screen_width)
            screen_y = int(index_tip.y * screen_height)
            pyautogui.moveTo(screen_x, screen_y)

            # Check for pinch gesture (index + thumb)
            thumb_tip = landmarks[4]
            thumb_x = int(thumb_tip.x * frame.shape[1])
            thumb_y = int(thumb_tip.y * frame.shape[0])

            distance = ((x - thumb_x)**2 + (y - thumb_y)**2)**0.5

            if distance < 40:
                pyautogui.click()
                cv2.putText(frame, 'Click!', (x, y - 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    cv2.imshow("Virtual Mouse", frame)
    if cv2.waitKey(1) == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
