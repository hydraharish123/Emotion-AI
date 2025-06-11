import os
import cv2
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt


def load_unlabelled_data(path, target_size=(64,64)):
    images = []
    for person_name in os.listdir(path):
        person_dir = os.path.join(path, person_name)
        if not os.path.isdir(person_dir):
            continue
        for img_file in os.listdir(person_dir):
            img_path = os.path.join(person_dir, img_file)
            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            if img is not None:
                img = cv2.resize(img, target_size)
                images.append(img)
    print(images)
    return np.array(images)



## Load labelled and unlabelled dataset

unlabelled_data = load_unlabelled_data("../data/lfw-deepfunneled") ## (13233, 64, 64)
flattened_unlabelled_data = unlabelled_data.reshape(unlabelled_data.shape[0], -1) ## (13233, 4096)

labelled_data = pd.read_csv('../data/ckextended.csv')

emotion_map = {
    0 : "Anger", 
    1 : "Disgust", 
    2 : "Fear", 
    3 : "Happiness", 
    4 : "Sadness", 
    5 : "Surprise", 
    6 : "Neutral", 
    7 : "Contempt"
}


## Prepare input and output for labeled data

labelled_data['emotion'] = labelled_data['emotion'].map(emotion_map)
print(labelled_data.head())
labelled_data.drop(columns=['Usage'], inplace=True)

emotion = list(labelled_data['emotion'])
emotion = np.array(emotion) ## (920,)
pixels = labelled_data['pixels']

X_labelled = []
for i in range(len(pixels)):
    X_labelled.append(pixels[i])

X_labelled_final = []
for x in X_labelled:
    X_labelled_final.append(np.array(list(map(int, x.split()))))
X_labelled_final = np.array(X_labelled_final) ## (920, 2304)


## Preprocess unlabelled dataset
original_size = (48, 48)
target_size = (64, 64)

X_resized = []

for img_flat in X_labelled_final:
    img_2d = img_flat.reshape(original_size).astype(np.uint8)
    resized_img = cv2.resize(img_2d, target_size, interpolation=cv2.INTER_AREA)
    X_resized.append(resized_img.flatten())

X_resized = np.array(X_resized) ## (920, 4096)


## Plot 5 images from labelled dataset

plt.figure(figsize=(10, 2)) 
for i in range(5):
    plt.subplot(1, 5, i+1) 
    img = X_resized[i].reshape(64, 64)
    plt.imshow(img, cmap='gray')
    plt.title(f'Image {i+1}')
    plt.axis('off')
plt.tight_layout()
plt.show()

## Plot 5 images from unlabelled dataset

plt.figure(figsize=(10, 2)) 
for i in range(5):
    plt.subplot(1, 5, i+1) 
    img = flattened_unlabelled_data[i].reshape(64, 64)
    plt.imshow(img, cmap='gray')
    plt.title(f'Image {i+1}')
    plt.axis('off')
plt.tight_layout()
plt.show()


## Preprocess conclusion

print(f"The dimensions of unlabelled data is {flattened_unlabelled_data.shape}")
print(f"The dimensions of labelled data is {X_resized.shape}")

## save data so that we can use it in the next script

np.save('X_resized_labelled.npy', X_resized)
np.save('X_flattened_unlabelled.npy', flattened_unlabelled_data)
np.save('emotion_labels.npy', emotion)