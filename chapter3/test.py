import warnings
warnings.filterwarnings("ignore", category = FutureWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning)

import numpy as np
from sklearn.utils import shuffle

from sklearn.datasets import fetch_openml
mnist = fetch_openml('mnist_784', version=1)
mnist.keys()

X, y = mnist["data"], mnist["target"]


X_train, X_test, y_train, y_test = X[:60000], X[60000:], y[:60000], y[60000:]

from scipy.ndimage.interpolation import shift

def shift_image(image, dx, dy):
    image = image.reshape((28, 28))
    shifted_image = shift(image, [dy, dx], cval=0, mode="constant")
    return shifted_image.reshape([-1])

X_train_augmented = [image for image in X_train]
y_train_augmented = [label for label in y_train]

for dx, dy in ((0,0), (1,0), (0,1), (1,1)):
	for image, label in zip(X_train, y_train):
		X_train_augmented.append(shift_image(image, dx, dy))
		y_train_augmented.append(label)

X_train_augmented = np.array(X_train_augmented)
y_train_augmented = np.array(y_train_augmented)

X_train_augmented, y_train_augmented = shuffle(X_train_augmented, y_train_augmented, random_state=0)

from sklearn.neighbors import KNeighborsClassifier

knn_clf = KNeighborsClassifier()

knn_clf.fit(X_train_augmented, y_train_augmented)

from sklearn.metrics import accuracy_score

y_pred = knn_clf.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))