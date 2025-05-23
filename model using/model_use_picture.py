import pickle

import cv2
import mediapipe as mp
import numpy as np
import matplotlib.pyplot as plt
import sys

model = pickle.load(open('model/randomforest_model.p', 'rb'))

if len(sys.argv) < 2:
    print("Usage: python model_use_picture.py <image_path>")
    exit(1)
image_path = sys.argv[1]
img  = cv2.imread(image_path)

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

hands = mp_hands.Hands(static_image_mode=True, min_detection_confidence=0.3)


data_aux = []
x_ = []
y_ = []

H, W, _ = img.shape

frame_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

results = hands.process(frame_rgb)
if results.multi_hand_landmarks:
    for hand_landmarks in results.multi_hand_landmarks:
        mp_drawing.draw_landmarks(
            img,  # image to draw
            hand_landmarks,  # model output
            mp_hands.HAND_CONNECTIONS,  # hand connections
            mp_drawing_styles.get_default_hand_landmarks_style(),
            mp_drawing_styles.get_default_hand_connections_style())

    for hand_landmarks in results.multi_hand_landmarks:
        for i in range(len(hand_landmarks.landmark)):
            x = hand_landmarks.landmark[i].x
            y = hand_landmarks.landmark[i].y

            x_.append(x)
            y_.append(y)

        for i in range(len(hand_landmarks.landmark)):
            x = hand_landmarks.landmark[i].x
            y = hand_landmarks.landmark[i].y
            data_aux.append(x - min(x_))
            data_aux.append(y - min(y_))

    # Pad with zeros if only one hand detected
    if len(data_aux) < 84:
        data_aux.extend([0.0] * (84 - len(data_aux)))

    x1 = int(min(x_) * W) - 10
    y1 = int(min(y_) * H) - 10

    x2 = int(max(x_) * W) - 10
    y2 = int(max(y_) * H) - 10

    prediction = model.predict([np.asarray(data_aux)])

    predicted_character = prediction[0]

    cv2.rectangle(img, (x1, y1), (x2, y2), (0, 0, 0), 4)
    cv2.putText(img, predicted_character, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 0, 0), 3,
                cv2.LINE_AA) # replace the color with (255,255,255) for white text

plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
plt.axis('off')
plt.show()