import cv2
import numpy as np
from tensorflow.keras.models import load_model

# Load trained model
model = load_model("trained_model.h5")

# Load any hand image
img = cv2.imread("hand.jpg")   # <-- put your image file here

# Preprocess for model
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
resized = cv2.resize(gray, (28, 28))
input_img = resized.reshape(1, 28, 28, 1) / 255.0

# Predict
pred = model.predict(input_img, verbose=0)
predicted_class = np.argmax(pred)
confidence = pred[0][predicted_class]

# Convert index to ASL letter
letter = chr(predicted_class + 65)

# Convert confidence to percentage
confidence_percent = confidence * 100

# Draw bounding box
h, w, _ = img.shape
cv2.rectangle(img, (10, 10), (w-10, h-10), (0, 255, 0), 3)

# Add ASL letter + accuracy
text = f"{letter} ({confidence_percent:.2f}%)"
cv2.putText(img, text, (15, 50),
            cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 3)

# Show and save output
cv2.imshow("ASL Prediction", img)
cv2.imwrite(f"ASL_{letter}.jpg", img)

cv2.waitKey(0)
cv2.destroyAllWindows()
