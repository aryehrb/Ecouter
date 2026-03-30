# Ecouter
by Aryeh Rothenberg

### Video Demo: 

### About
This projects uses fundamental machine learning (ML) tools to create a lightweight model that can recognize sounds from pure audio input. The project is entirely vertically-integrated, meaning the user starts by building their own dataset, training 2 models with it, evaluating the models' performance, and testing them with brand new data in real-time. The project comprises 5 primary components, each of which is organized in its own Python function. The following is a breakdown of what each function accomplishes and why:

1. Record()
Record() captures individual recordings of user-defined sounds and saves them to the local file structure. The user specifies the name of the sound they wish to record and the quantity of recordings they wish to make for that sound, and then they are prompted to record the sounds using their device's microphone.
This step is important because it creates the dataset upon which the ML models will be trained.

2. Transform()
Transform() takes the raw audio (WAV) files and transforms them into a clean, organized arrays of values that represent the sound features of the audio. This step uses the Python library librosa to extract the key features from the audio signals using the Mel-Frequency Cepstral Coefficients method. These coefficients are then averaged across all timestamps to produce an array of uniform values.
This step is important because it processes the raw audio data into a form that enables the ML models to be trained.

3. Train()
Train() relies entirely on the Python library scikit-learn to create lightweight ML models using the newly processed dataset. Using two different methods (K Nearest Neighbor and Structure Vector Machines), the function creates two lightweight models that ideally learn to relate the features of the audio files to the user-defined "ground truth" (GT) labels (defined in the Record() function). For ease of evaluation, the bespoke dataset is separated into two pieces, one used for training the models (80% of the data) and the other for evaluating (20%).

4. Score()
Score() uses the two models from Train() to predict the labels of each of the audio samples in the "evaluate" dataset. Importantly, this data was not used for training the models; therefore, as unseen data, the label predictions are purely based on the recognition abilities of the models. To evaluate the performance of a model, its predicted label for each audio file is compared to the GT label, and then binary score is determined by whether the prediction is correct or incorrect. The mean of the individual results reveals the model's overall score.

5. Test()
Test() allows the user to test the performance of the models in real-time. Going through each of the steps again, the user records a sound using their microphone and gets to see the model output its prediction. For each recorded sound, the user will see the prediction made by the KNN model and the SVM mode, from which they can see which predictions were right or wrong. Testing continues until the user chooses to stop.

One of the challenges encountered during this project was the ambiguity of audio recordings taken at different times, in different settings, and at different volumes/distances from the microphone. With all these different fluctuations, the models were completely unable to reason on the input sounds and would therefore return just a random guess (or in many cases, the same guess every try). The solution to fluctuations in background noise, etc. proved trimming the dead noise from all recordings, leaving the primary audio as the only component of the file. To solve the discrepancies in recording volume, the audio signals were normalized to a scale of 0-1 before being passed into the training or testing modules. These two solutions greatly enhanced the accuracy of the two models, specifically the Structure Vector Machines (SVM) model, which jumped from ~8% accuracy before trimming to ~60% after.

Of the two ML approaches evaluated, the K Nearest Neighbors (KNN) model proved more accurate across all trials than the SVM model, which aligned with my expectations. KNN uses proximity to predict similar features, and it each time it seeks a prediction it refers back to the training data to determine distance matches. In this way, the prediction is heavily tied to the training data, and each prediction relies on the sampling of individual data points from the dataset instead of relying on a single generalization made during training. SVM takes this second approach by drawing distinct boundaries between the training data points and then discarding the training data during inference to purely rely on the derived boundaries. This approach is weaker in the case of sound recognition because many sounds (e.g. spoken words) can vary greatly without losing their accepted meaning; therefore, trying to separate sounds into rigid bins to compare to during inference time can prove less efficient than directly comparing an input signal to the labeled training data. Updating the scope of the project to include 2 distinct ML approaches was a great choice because it helped me understand the underlying differences between approaches, and how certain approaches can be better tailored for certain kinds of data.

Thank you for a great course!
This has been Av.