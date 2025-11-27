import cv2
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
LABEL_MAPPING = {i: str(i) for i in range(10)}

# Preprocessing


def preprocess_image(path):
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        print(f'Khong the doc anh tu duong dan: {path}')
        return None, None
    blur = cv2.GaussianBlur(img, (5, 5), 0)
    _, thresh = cv2.threshold(
        blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    return img, thresh

# segmentation


def segment_characters(thresh):
    contours, _ = cv2.findContours(
        thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    chars = []
    boxes = []
    for c in contours:
        x, y, w, h = cv2.boundingRect(c)
        # Loc nhieu nho
        if w*h > 50:
            char = thresh[y:y+h, x:x+w]
            char = cv2.resize(char, (28, 28))
            char = char.astype('float32') / 255.0
            char = char.reshape(28, 28, 1)
            chars.append(char)
            boxes.append((x, y, w, h))
    chars = [x for _, x in sorted(
        zip(boxes, chars), key=lambda pair: pair[0][0])]
    boxes = sorted(boxes, key=lambda b: b[0])
    return chars, boxes


if __name__ == "__main__":
    try:
        model = load_model('handwritten_recognizer_model.h5')
        print("Mo hinh da duoc tai thanh cong.")
    except Exception as e:
        print("Khong the tai mo hinh. Vui long dam bao mo hinh da duoc huan luyen va luu vao 'handwritten_recognizer_model.h5'.")
        exit()

    IMAGE_PATH = './image/text.jpg'
    raw, thresh = preprocess_image(IMAGE_PATH)

    if raw is None:
        exit()

    chars, boxes = segment_characters(thresh)

    predictions = []
    if not chars:
        print('Khong tim thay ky tu nao trong anh.')
    else:
        for ch in chars:
            ch_input = np.expand_dims(ch, axis=0)

            prediction_vector = model.predict(ch_input, verbose=0)[0]
            predicted_index = np.argmax(prediction_vector)

            predicted_char = LABEL_MAPPING.get(predicted_index, '?')
            predictions.append(predicted_char)

        result = "".join(predictions)
        print('\n==============================')
        print("Kết quả Nhận dạng (Chữ số):", result)
        print('==============================\n')

    debug = cv2.cvtColor(raw, cv2.COLOR_GRAY2BGR)
    for (x, y, w, h), char_label in zip(boxes, predictions):
        cv2.rectangle(debug, (x, y), (x+w, y+h), (0, 255, 0), 2)
        cv2.putText(debug, char_label, (x, y-10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

    plt.figure(figsize=(10, 4))
    plt.subplot(1, 2, 1)
    plt.title("Anh goc")
    plt.imshow(raw, cmap='gray')
    plt.axis('off')

    plt.subplot(1, 2, 2)
    plt.title("Ket qua phat hien ky tu")
    plt.imshow(debug)
    plt.axis('off')
    plt.show()
