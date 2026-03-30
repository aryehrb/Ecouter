# Final Project Proposal: Signal-Based Voice Recognition (Intermediate FT)

## Goal
Build a voice recognition system that manually processes audio signals using the Fourier Transform to identify frequency patterns in spoken words.

## Tech Stack
- **`numpy`**: For Fast Fourier Transform (`np.fft.rfft`) and array manipulation.
- **`scipy`**: For audio file I/O and basic signal filtering.
- **`matplotlib`**: To visualize the "Power Spectrum" of different words.
- **`scikit-learn`**: For a simple classifier (e.g., KNN) to recognize the frequency patterns.

## Core Features
1. **Windowing Logic**: Break 1-2 second audio clips into 50ms overlapping frames using `numpy` slicing.
2. **Manual Feature Extraction**:
   - Apply `np.fft.rfft` to each frame to get the frequency domain representation.
   - Calculate the "Power Spectrum" using `np.abs()`.
   - **Binning**: Aggregate hundreds of frequency points into ~10-20 meaningful frequency bins (e.g., Low, Mid, High frequencies) by averaging.
3. **Training**: Use the 2D "Spectrogram" (Time x Frequency Bins) as a feature vector for a simple classifier.
4. **Visualization**: Create a tool that plots the "signature" of a word, showing why "Stop" looks different from "Go" in the frequency domain.

## Why this is manageable
You leverage `numpy.fft` for the heavy math while writing the "intermediate" logic for windowing and binning. This provides deep insight into signal processing without the risk of an "infinite" scope.
