import os
import numpy as np
import cv2
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report


def load_data(path):
    data = []
    labels = []

    for folder in os.listdir(path):
        for filename in os.listdir(path + '/' + folder):
            img = cv2.imread(path + '/' + folder + '/' + filename, 0)
            img = cv2.resize(img, (200, 200))
            data.append(img.flatten())
            if folder == 'Tumor':
                labels.append(1)
            else:
                labels.append(0)

    return np.array(data), np.array(labels)


def main():
    data, labels = load_data("C:/Users/mridu/Desktop/Paper Images/SVMNew/CLAHE-RMSHE")
    X_train, X_test, y_train, y_test = train_test_split(data, labels, test_size=0.2, random_state=42)
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=42)

    clf = svm.SVC(kernel='linear')
    clf.fit(X_train, y_train)

    y_pred_train = clf.predict(X_train)
    y_pred_test = clf.predict(X_test)
    y_pred_val = clf.predict(X_val)

    print("Train Accuracy: ", accuracy_score(y_train, y_pred_train))
    print("Test Accuracy: ", accuracy_score(y_test, y_pred_test))
    print("Validation Accuracy: ", accuracy_score(y_val, y_pred_val))

    print("Train Confusion Matrix: \n", confusion_matrix(y_train, y_pred_train))
    print("Test Confusion Matrix: \n", confusion_matrix(y_test, y_pred_test))
    print("Validation Confusion Matrix: \n", confusion_matrix(y_val, y_pred_val))

    print("Train Classification Report: \n", classification_report(y_train, y_pred_train))
    print("Test Classification Report: \n", classification_report(y_test, y_pred_test))
    print("Validation Classification Report: \n", classification_report(y_val, y_pred_val))


if __name__ == '__main__':
    main()
