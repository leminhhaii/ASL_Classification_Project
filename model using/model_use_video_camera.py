import pickle

import cv2
import mediapipe as mp
import numpy as np

model = pickle.load(open("model/randomforest_model.p", 'rb'))

cap = cv2.VideoCapture(0)

# Uncomment the following lines to use a video file instead of the webcam
# video_path = r"C:\Users\Admin\Downloads\ASL_alphabet.mp4"
# cap = cv2.VideoCapture(video_path) 

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

hands = mp_hands.Hands(static_image_mode=True, min_detection_confidence=0.3)

labels_dict = {0: 'A', 1: 'B', 2: 'C', 3: 'D', 4: 'E', 5: 'F', 6: 'G',
                7: 'H', 8: 'I', 9: 'K', 10: 'L', 11: 'M', 12: 'N',
                13: 'O', 14: 'P', 15:'Q', 16: 'R', 17: 'S', 18: 'T',
                19: 'U', 20: 'V', 21: 'W', 22: 'X', 23: 'Y'}

# If you want to save the output video with predictions, uncomment the following lines
# output_path = "output_with_predictions.mp4"
# fourcc = cv2.VideoWriter_fourcc(*'mp4v')
# out = None

while True:

    data_aux = []
    x_ = []
    y_ = []

    ret, frame = cap.read()
    if not ret:
        print("❌ Failed to grab frame. Exiting...")
        break

    H, W, _ = frame.shape
    # H = int(H/2)
    # W = int(W/2)
    # frame = cv2.resize(frame, (W, H))  # Resize the frame to 640x480 (If needed)
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Initialize video writer after knowing frame size
    # if out is None:
    #     out = cv2.VideoWriter(output_path, fourcc, 20.0, (W, H))

    results = hands.process(frame_rgb)
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(
                frame,  # image to draw
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

        if len(data_aux) < 84:
            data_aux.extend([0.0] * (84 - len(data_aux)))

        x1 = int(min(x_) * W) - 10
        y1 = int(min(y_) * H) - 10

        x2 = int(max(x_) * W) - 10
        y2 = int(max(y_) * H) - 10

        prediction = model.predict([np.asarray(data_aux)])

        predicted_character = prediction[0]

        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 0), 4)
        cv2.putText(frame, predicted_character, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0,0,0), 3,
                    cv2.LINE_AA)

    cv2.imshow('frame', frame)
    # out.write(frame)  # Save the frame to the output video
    # if cv2.waitKey(2) & 0xFF == ord('q'):
    #     break
    cv2.waitKey(1)

cap.release()
# if out is not None:
#     out.release()
cv2.destroyAllWindows()