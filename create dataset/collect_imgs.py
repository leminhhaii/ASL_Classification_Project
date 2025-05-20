import os
import cv2


DATA_DIR = "./dataset-images" # Directory to save the images
if not os.path.exists(DATA_DIR):
    os.makedirs(DATA_DIR)

number_of_classes = 13
dataset_size = 1000

alphabet = {
    0: "apple", 1: "can", 2: "get", 3: "thank-you", 4: "have", 5: "want", 6: "how",
    7: "I-me", 8: "like", 9: "love", 10: "my", 11: "no", 12: "sorry", 13: "you", 
    14: "A", 15: "B", 16: "C", 17: "D", 18: "E", 19: "F", 20: "G", 21: "H", 22: "I",
    23: "K", 24: "L", 25: "M", 26: "N", 27: "O", 28: "P", 29: "Q", 30: "R", 31: "S",
    32: "T", 33: "U", 34: "V", 35: "W", 36: "X", 37: "Y"
}

cap = cv2.VideoCapture(0)
for j in range(number_of_classes):
    if not os.path.exists(os.path.join(DATA_DIR, alphabet[j])):
        os.makedirs(os.path.join(DATA_DIR, alphabet[j]))

    print('Collecting data for class {}'.format(alphabet[j]))

    done = False
    while True:
        ret, frame = cap.read()
        cv2.putText(frame, 'Ready? Press "Q" ! :)', (100, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 255, 0), 3,
                    cv2.LINE_AA)
        cv2.imshow('frame', frame)
        if cv2.waitKey(25) == ord('q'):
            break

    counter = 0
    while counter < dataset_size:
        ret, frame = cap.read()
        cv2.imshow('frame', frame)
        cv2.waitKey(25)
        cv2.imwrite(os.path.join(DATA_DIR, alphabet[j], '{}.jpg'.format(counter)), frame)

        counter += 1

cap.release()
cv2.destroyAllWindows()