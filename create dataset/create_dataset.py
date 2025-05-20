import os
import pickle

import mediapipe as mp
import cv2
import matplotlib.pyplot as plt
import albumentations as A
from collections import Counter

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

hands = mp_hands.Hands(static_image_mode=True, min_detection_confidence=0.3)

DATA_DIR = './dataset-images'  # Directory of the saved images

data = []
labels = []

import albumentations as A

# Define augmentation pipeline: Only horizontal flip and brightness/contrast, not rotation
# as it may cause the hand to be upside down
augment = A.Compose([
    A.HorizontalFlip(p=0.5),
    A.RandomBrightnessContrast(p=0.2),
])

for dir_ in os.listdir(DATA_DIR):
    for img_path in os.listdir(os.path.join(DATA_DIR, dir_)):
        img = cv2.imread(os.path.join(DATA_DIR, dir_, img_path))
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # Apply augmentation
        augmented = augment(image=img_rgb)
        img_rgb_aug = augmented['image']

        results = hands.process(img_rgb_aug)
        hand_landmarks_list = results.multi_hand_landmarks if results.multi_hand_landmarks else []

        data_aux = []
        for hand_idx in range(2):
            if hand_idx < len(hand_landmarks_list):
                hand_landmarks = hand_landmarks_list[hand_idx]
                x_ = [lm.x for lm in hand_landmarks.landmark]
                y_ = [lm.y for lm in hand_landmarks.landmark]
                min_x, min_y = min(x_), min(y_)
                for lm in hand_landmarks.landmark:
                    data_aux.append(lm.x - min_x)
                    data_aux.append(lm.y - min_y)
            else:
                data_aux.extend([0.0] * 42)

        if len(hand_landmarks_list) > 0:
            data.append(data_aux)
            labels.append(dir_)

f = open('./dataset/full_asl_dataset.pickle', 'wb')
pickle.dump({'data': data, 'labels': labels}, f)
print('Data saved !')
print('Number of samples: {}'.format(len(data)))
print('Number of classes: {}'.format(len(labels)))
label_counts = Counter(labels)
print("Samples per label:")

for label, count in label_counts.items():
    print(f"{label}: {count}")

f.close()