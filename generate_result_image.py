import cv2
import numpy as np
from tensorflow.keras.models import load_model

# Load model
model = load_model("trained_model.h5")

# Load your test image
img = cv2.imread("hand_A.jpg")  # <-- replace with your image file

# Preprocessing for model
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
resized = cv2.resize(gray, (28, 28))
input_img = resized.reshape(1, 28, 28, 1) / 255.0

# Prediction
pred = model.predict(input_img, verbose=0)
predicted_class = np.argmax(pred)
confidence = pred[0][predicted_class]

# Convert class number to letter
letter = chr(predicted_class + 65)

# Draw green box
h, w, _ = img.shape
cv2.rectangle(img, (10, 10), (w-10, h-10), (0, 255, 0), 3)

# Put label on top
text = f"{letter} ({confidence:.2f})"
cv2.putText(img, text, (15, 40),
            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

# Show and Save
cv2.imshow("Result", img)
cv2.imwrite(f"result_{letter}.jpg", img)

cv2.waitKey(0)
cv2.destroyAllWindows()
