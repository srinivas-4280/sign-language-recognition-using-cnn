import cv2
import numpy as np
from tensorflow.keras.models import load_model

# Load the trained model
model = load_model("trained_model.h5")

# Open webcam
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    resized = cv2.resize(gray, (28, 28))
    img = resized.reshape(1, 28, 28, 1) / 255.0

    # Prediction
    pred = model.predict(img)
    letter = chr(np.argmax(pred) + 65)

    # Display result
    cv2.putText(frame, f"Predicted: {letter}", (10, 50),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    cv2.imshow("Sign Language Detection", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
