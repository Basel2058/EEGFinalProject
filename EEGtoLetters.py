import os
import scipy.io
import numpy as np
import scipy.signal as signal

#qq
# Function to map frequency to letters
def map_to_letter(frequency):
    # Define a mapping from frequency ranges to letters
    mapping = {
        (0.5, 4): 'D',
        (4, 8): 'T',
        (8, 12): 'A',
        (12, 30): 'B',
        (30, 50): 'G'
    }
    for freq_range, letter in mapping.items():
        if freq_range[0] <= frequency <= freq_range[1]:
            return letter
    return 'F'  # Default letter for frequencies outside defined ranges


# Convert EEG data into sequence of letters
def eeg_to_letters(eeg_data, segment_size, overlap, sampling_freq):
    num_samples = eeg_data.shape[1]
    letters = []
    start_idx = 0
    end_idx = segment_size
    while end_idx <= num_samples:
        segment = eeg_data[:, start_idx:end_idx]
        f, psd = signal.welch(segment, fs=sampling_freq, nperseg=min(segment_size, segment.shape[1]))
        dominant_frequency = f[np.argmax(np.mean(psd, axis=0))]
        letter = map_to_letter(dominant_frequency)
        letters.append(letter)
        start_idx += int(segment_size * (1 - overlap))
        end_idx += int(segment_size * (1 - overlap))
    return letters


# Directory containing the cleaned .mat files
cleaned_directory = "Cleaned_EEG_data"
output_text_directory = "EEG_Letter_Sequences"
if not os.path.exists(output_text_directory):
    os.makedirs(output_text_directory)

# Convert cleaned EEG data into sequence of letters for each file
for file_name in os.listdir(cleaned_directory):
    if file_name.endswith('.mat'):
        file_path = os.path.join(cleaned_directory, file_name)
        mat = scipy.io.loadmat(file_path)
        variable_name = list(mat.keys())[-1]  # Dynamically get the variable name
        cleaned_data = mat[variable_name]
        sfreq = 128  # Sampling frequency
        segment_duration = 3  # Segment duration in seconds
        segment_size = segment_duration * sfreq  # Segment size in samples
        overlap = 0.5  # Adjust as needed
        letters = eeg_to_letters(cleaned_data, segment_size, overlap, sfreq)
        sequence_str = ''.join(letters)

        # Save the sequence of letters to a text file
        base_name = os.path.splitext(file_name)[0]
        output_text_file_path = os.path.join(output_text_directory, f"{base_name}.txt")
        with open(output_text_file_path, 'w') as f:
            f.write(sequence_str)

        print(f"Sequence of letters for {file_name} saved to {output_text_file_path}")
