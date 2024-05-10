import streamlit as st
import cv2
import numpy as np
import mediapipe as mp
from keras.models import load_model
import webbrowser

# Load emotion recognition model and labels
model = load_model("model.h5")
label = np.load("labels.npy")

# Initialize MediaPipe
holistic = mp.solutions.holistic
hands = mp.solutions.hands
holis = holistic.Holistic()
drawing = mp.solutions.drawing_utils

st.header("Emotion Based Music Recommender")

# Define a flag to control whether to run the emotion detection loop
if "run" not in st.session_state:
    st.session_state["run"] = True

try:
    emotion = np.load("emotion.npy")[0]
except:
    emotion = ""

if not emotion:
    st.session_state["run"] = True
else:
    st.session_state["run"] = False

# Define a function to process frames and perform emotion recognition
def process_frame(frame):
    frm = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

    # Perform emotion recognition
    lst = []
    res = holis.process(frm)

    if res.face_landmarks:
        for i in res.face_landmarks.landmark:
            lst.append(i.x - res.face_landmarks.landmark[1].x)
            lst.append(i.y - res.face_landmarks.landmark[1].y)

        if res.left_hand_landmarks:
            for i in res.left_hand_landmarks.landmark:
                lst.append(i.x - res.left_hand_landmarks.landmark[8].x)
                lst.append(i.y - res.left_hand_landmarks.landmark[8].y)
        else:
            for i in range(42):
                lst.append(0.0)

        if res.right_hand_landmarks:
            for i in res.right_hand_landmarks.landmark:
                lst.append(i.x - res.right_hand_landmarks.landmark[8].x)
                lst.append(i.y - res.right_hand_landmarks.landmark[8].y)
        else:
            for i in range(42):
                lst.append(0.0)

        lst = np.array(lst).reshape(1, -1)

        pred = label[np.argmax(model.predict(lst))]

        print(pred)
        cv2.putText(frm, pred, (50, 50), cv2.FONT_ITALIC, 1, (255, 0, 0), 2)

        np.save("emotion.npy", np.array([pred]))

    drawing.draw_landmarks(frm, res.face_landmarks, holistic.FACEMESH_TESSELATION,
                           landmark_drawing_spec=drawing.DrawingSpec(color=(0, 0, 255), thickness=-1,
                                                                       circle_radius=1),
                           connection_drawing_spec=drawing.DrawingSpec(thickness=1))
    drawing.draw_landmarks(frm, res.left_hand_landmarks, hands.HAND_CONNECTIONS)
    drawing.draw_landmarks(frm, res.right_hand_landmarks, hands.HAND_CONNECTIONS)

    return frm


# Capture video from webcam and process frames
cap = cv2.VideoCapture(0)

while st.session_state["run"]:
    ret, frame = cap.read()

    if ret:
        processed_frame = process_frame(frame)
        st.image(processed_frame, channels="BGR")

cap.release()
