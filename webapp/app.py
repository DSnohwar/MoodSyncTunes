from flask import Flask, request, jsonify
import cv2
import numpy as np
from keras.models import load_model
from keras.preprocessing.image import img_to_array
from flask_cors import CORS

app = Flask(__name__)
CORS(app)  # Allow frontend to call API

# Load pre-trained emotion model
emotion_model = load_model('../model/CNN_Final_Modelv2.h5')

# Define emotion labels
emotion_dict = {
    0: "Angry", 1: "Disgusted", 2: "Fearful",
    3: "Happy", 4: "Neutral", 5: "Sad", 6: "Surprised"
}

# Face detector
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "../haarcascade_frontalface_default.xml")

@app.route("/predict", methods=["POST"])
def predict_emotion():
    try:
        file = request.files["image"]
        image = np.frombuffer(file.read(), np.uint8)
        frame = cv2.imdecode(image, cv2.IMREAD_COLOR)

        # Convert to grayscale
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray_frame, 1.3, 5)

        emotions = []
        for (x, y, w, h) in faces:
            roi_gray = gray_frame[y:y + h, x:x + w]
            cropped_img = cv2.resize(roi_gray, (48, 48))
            cropped_img = cropped_img.astype("float") / 255.0
            cropped_img = img_to_array(cropped_img)
            cropped_img = np.expand_dims(cropped_img, axis=0)

            # Predict emotion
            prediction = emotion_model.predict(cropped_img)
            maxindex = int(np.argmax(prediction))
            emotions.append(emotion_dict[maxindex])

        return jsonify({"emotions": emotions})

    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000)
