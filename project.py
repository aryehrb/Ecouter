
import sounddevice as sd
import numpy as np
import wave
import os
import sys
import librosa
from sklearn import neighbors, svm as sv, model_selection


def main():

    # choose where your sound files will be stored
    global data_folder
    data_folder = "sound_data"

    # comment out the following line if you are using a separate dataset
    #record()
    words = os.listdir(data_folder)
    print(f"Words: {words}")

    # extract features from the sound data
    features, labels = transform(words)

    # train 2 ML models on the dataset
    knn_model, svm_model, knn_res, svm_res, eval_labels = train(features, labels)

    # evaluate the results of the models on the "test" part of the dataset
    knn_score, svm_score = score(knn_res, svm_res, eval_labels)
    print(f"K Nearest Neighbors score on test dataset: {knn_score*100:.2f}%",
          f"Support Vector Machines score on test dataset: {svm_score*100:.2f}%", sep="\n")
    
    test(knn_model, svm_model)

## record audio samples
def record():
    
    while True:

        word = input("Word to record (press Enter to exit): ")
        
        if not word:
            sys.exit("Done recording")
        
        else:

            sample_count = int(input("Number of samples desired: "))
            duration = input("Time needed to record each sample (how many seconds?): ")

            if not duration:
                duration = 2 # seconds
            else:
                duration = float(duration)

            fs = 44100 # Hz

            # create folder to save audio files
            output_path = os.path.join(data_folder, word)
            print(output_path)
            # output_path = f"test/{word}/"

            if not os.path.exists(output_path):
                os.makedirs(output_path)

            # record the desired number of audio samples
            n = 0
            while n < sample_count:

                input("Press Enter to record.")
                print("***recording***")

                sound = sd.rec(int(duration*fs), samplerate=fs, channels=1)
                sd.wait()

                # exponent operator is ** in python
                sound_int16 = np.int16(sound*(2**15-1))
                # add zero-padding to file name
                with wave.open(os.path.join(output_path, f"{word}_{n:02d}.wav"), "wb") as wav_file:
                    wav_file.setnchannels(1)
                    wav_file.setsampwidth(2)
                    wav_file.setframerate(fs)

                    wav_file.writeframes(sound_int16.tobytes())

                    n+=1


        print("+ Recording Complete. +\n")

## transform the sound data
def transform(words):
    signal = []
    features = []
    labels = []

    # use librosa.load to load in the .wav files
    for word in words:
        for _, _, samples in os.walk(os.path.join(data_folder, word)):
            for sample in sorted(samples):
                sample_path = os.path.join(data_folder, word, sample)

                # librosa.load automatically resamples the audio to 22050 Hz from 44100 Hz
                signal, sr = librosa.load(sample_path)

                signal, _ = librosa.effects.trim(signal, top_db=30)
                signal = librosa.util.normalize(signal)


            # use librosa.features.mccf to perform Fourier Transform on the audio files; returns 2D matrix of coefficients
            # flatten 2D matrix into 1D

                # Mel-frequency cepstral coefficients extraction
                mfccs = librosa.feature.mfcc(y=signal, sr=sr, n_mfcc=13)

                # flatten the 2d array  into 1d (each column stores the mfccs for a single time period; # of rows may vary)
                mfcc_processed = np.mean(mfccs, axis=1)
            
                features.append(mfcc_processed)
                labels.append(word)

    # create large, continuous dataset with all processed values in one vector, and all gt labels in the other
    X = np.array(features)
    Y = np.array(labels)

    # to save time while working on the project; can be commented out before submission
    # np.save("features.npy", X)
    # np.save("labels.npy", Y)

    return X, Y

# train two ML models using bespoke dataset
def train(X, Y):


    # partition dataset into train (80%) and test (20%)
    X_train, X_eval, Y_train, Y_eval = model_selection.train_test_split(X, Y, test_size=0.2)

    # train KNN model
    knn = neighbors.KNeighborsClassifier(n_neighbors=5)
    knn.fit(X_train, Y_train)

    # train SVM model
    svm = sv.SVC()
    svm.fit(X_train, Y_train)

    # evalute both model on test data
    results_knn = knn.predict(X_eval)
    results_svm = svm.predict(X_eval)

    return knn, svm, results_knn, results_svm, Y_eval        

# evaluate the models' performance on the test data
def score(a, b, l):
    knn_score = np.mean(a == l)
    svm_score = np.mean(b == l)

    return knn_score, svm_score

# create new data to test the models in real-time
def test(knn, svm):
    # record new audio and feed to one or both ML models to see how they perform
    while True:
        if input("\nTo test the models, record yourself saying one of the following words (press Enter to continue or Any Key + Enter to exit).\n"):
            sys.exit()
        else:
            print(f"Words: {os.listdir(data_folder)}")
            input("\nPress Enter to start recording ")
            print("***recording***\n")

            fs = 44100 # Hz
            sound = sd.rec(int(2*fs), samplerate=fs, channels=1)
            sd.wait()

            # unravel the column vector into a flat 1D array
            sound = sound.ravel()
            # resample the audio to match the trained frequency
            sound = librosa.resample(sound, orig_sr=fs, target_sr=22050)
            # trim away dead noise
            sound, _ = librosa.effects.trim(sound, top_db=30)
            # normalize audio to avoid issues due to recording volume
            sound = librosa.util.normalize(sound)

            mfccs = librosa.feature.mfcc(y=sound, sr=22050, n_mfcc=13)
            # flatten the 2d array (each column stores the mfccs for a single time period; # of rows may vary) into 1D
            mfcc_processed = np.mean(mfccs, axis=1)

            print(f"KNN Prediction: {knn.predict([mfcc_processed])}")
            print(f"SVM Prediction: {svm.predict([mfcc_processed])}")


if __name__ == "__main__":
    main()
