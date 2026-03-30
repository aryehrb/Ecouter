import pytest
import os
import wave
import numpy as np
from sklearn import neighbors, svm as sv, model_selection
import project
from project import transform, train, score

def test_transform(tmp_path):
    test_dir = os.path.join(tmp_path, "rarrf")
    os.makedirs(test_dir)

    test_path = os.path.join(test_dir, "rarrf_00.wav")
    with wave.open(str(test_path), "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(22050)
        wav.writeframes(b'\x00\x00' * 22050)

    project.data_folder = str(tmp_path)

    features, labels = transform(["rarrf"])

    assert labels[0] == "rarrf"
    assert features.shape == (1, 13)

def test_train():
    X_test = np.random.rand(50, 13)
    Y_test = np.array(["ruff"]*25 + ["meow"]*25)

    knn, svm, knn_results, svm_results, Y_eval = train(X_test, Y_test)

    assert isinstance(knn, neighbors.KNeighborsClassifier)
    assert isinstance(svm, sv.SVC)
    assert len(knn_results) == 10
    assert len(svm_results) == 10
    assert len(Y_eval) == 10
    
def test_score():
    
    knn_predictions = np.array(["bark", "meow", "hiss", "cluck", "growl", "ribbit"])
    svm_predictions = np.array(["bark", "meow", "tweet", "cluck", "moo", "moo"])
    gt_labels =       np.array(["bark", "hiss", "tweet", "cluck", "growl", "ribbit"])
    
    knn_score, svm_score = score(knn_predictions, svm_predictions, gt_labels)

    assert np.isscalar(knn_score)
    assert np.isscalar(svm_score)
    assert knn_score == 4/6
    assert svm_score == 0.5