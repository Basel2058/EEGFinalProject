import os
import scipy.io
import numpy as np


# Function to map quantized values to letters
def map_to_letter(value):
    # Define a more granular mapping from value ranges to letters
    if 0 <= value < 43:
        return 'A'
    elif 43 <= value < 86:
        return 'B'
    elif 86 <= value < 129:
        return 'C'
    elif 129 <= value < 172:
        return 'D'
    elif 172 <= value < 215:
        return 'E'
    elif 215 <= value <= 256:
        return 'F'
    else:
        return 'G'  # Default letter for unexpected values


# Function to normalize and quantize the EEG data
def quantize_eeg_data(eeg_data):
    # Normalize the EEG data using z-score normalization
    mean = np.mean(eeg_data)
    std = np.std(eeg_data)
    z_normalized_signal = (eeg_data - mean) / std

    # Clip the values to be within a specific range to avoid extreme outliers
    clipped_signal = np.clip(z_normalized_signal, -3, 3)

    # Scale to the range 0-256
    scaled_signal = ((clipped_signal + 3) / 6) * 256

    # Quantize to integer values
    quantized_signal = np.round(scaled_signal).astype(int)

    return quantized_signal

#e3ee
# Directory containing the cleaned .mat files
cleaned_directory = "Cleaned_EEG_data"
output_text_directory = "EEG_Quantized_Sequences"
if not os.path.exists(output_text_directory):
    os.makedirs(output_text_directory)

# Convert cleaned EEG data into quantized sequences for each file
for file_name in os.listdir(cleaned_directory):
    if file_name.endswith('.mat'):
        file_path = os.path.join(cleaned_directory, file_name)
        mat = scipy.io.loadmat(file_path)
        variable_name = list(mat.keys())[-1]  # Dynamically get the variable name
        cleaned_data = mat[variable_name]

        # Segment size in seconds
        segment_duration = 3  # Segment duration in seconds
        sfreq = 128  # Sampling frequency
        segment_size = segment_duration * sfreq  # Segment size in samples

        sequence_str = ""
        for start in range(0, cleaned_data.shape[1], segment_size):
            segment = cleaned_data[:, start:start + segment_size]

            if segment.shape[1] < segment_size:
                continue  # Skip incomplete segment

            # Quantize the EEG data segment
            quantized_data = quantize_eeg_data(segment)

            # Flatten the quantized data to a 1D array
            quantized_sequence = quantized_data.flatten()

            # Convert quantized values to a string of letters
            letters_sequence = ''.join(map(map_to_letter, quantized_sequence))

            # Append the letters to the overall sequence string
            sequence_str += letters_sequence

        # Save the sequence of letters to a text file
        base_name = os.path.splitext(file_name)[0]
        output_text_file_path = os.path.join(output_text_directory, f"{base_name}.txt")
        with open(output_text_file_path, 'w') as f:
            f.write(sequence_str)

        print(f"Quantized sequence for {file_name} saved to {output_text_file_path}")
