import cv2
import numpy as np
import mediapipe as mp
import tensorflow as tf

mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils

# Load trained model
model = tf.keras.models.load_model("trained_model.h5")

# Webcam
cap = cv2.VideoCapture(0)

# Mediapipe Hands
hands = mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.6, min_tracking_confidence=0.6)

labels = [chr(i) for i in range(65, 91)]  # A-Z

while True:
    ret, frame = cap.read()
    if not ret:
        break

    img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(img_rgb)

    h, w, c = frame.shape

    if results.multi_hand_landmarks:

        for handLms in results.multi_hand_landmarks:
            # Extract bounding box
            x_min = w
            y_min = h
            x_max = 0
            y_max = 0

            for lm in handLms.landmark:
                x, y = int(lm.x * w), int(lm.y * h)
                x_min = min(x_min, x)
                y_min = min(y_min, y)
                x_max = max(x_max, x)
                y_max = max(y_max, y)

            # Draw bounding box
            cv2.rectangle(frame, (x_min - 20, y_min - 20), (x_max + 20, y_max + 20), (0, 255, 0), 2)

            # Crop hand region
            hand_img = frame[y_min:y_max, x_min:x_max]

            if hand_img.size > 0:
                hand_img = cv2.resize(hand_img, (28, 28))
                hand_img = cv2.cvtColor(hand_img, cv2.COLOR_BGR2GRAY)
                hand_img = hand_img.reshape(1, 28, 28, 1) / 255.0

                # Prediction
                pred = model.predict(hand_img)[0]
                idx = np.argmax(pred)
                letter = labels[idx]
                accuracy = pred[idx]

                # Show prediction
                cv2.putText(frame, f"{letter} ({accuracy:.2f})", (x_min, y_min - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

            mp_draw.draw_landmarks(frame, handLms, mp_hands.HAND_CONNECTIONS)

    cv2.imshow("ASL Detection", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
