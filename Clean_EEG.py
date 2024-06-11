import os
import scipy.io
import mne

# Function to clean EEG data
def clean_eeg_data(file_path, output_directory):
    mat = scipy.io.loadmat(file_path)
    variable_name = list(mat.keys())[-1]  # Dynamically get the variable name
    data = mat[variable_name]
    sfreq = 128  # Sampling frequency
    ch_names = ['Fz', 'Cz', 'Pz', 'C3', 'T3', 'C4', 'T4', 'Fp1', 'Fp2', 'F3', 'F4', 'F7', 'F8', 'P3', 'P4', 'T5', 'T6', 'O1', 'O2']
    info = mne.create_info(ch_names=ch_names, sfreq=sfreq, ch_types='eeg')

    # Create Raw object
    raw = mne.io.RawArray(data.T, info)  # Transpose data to match MNE's format (n_channels, n_times)

    # Set montage
    montage = mne.channels.make_standard_montage('standard_1020')
    raw.set_montage(montage)

    # Filter the data (0.5-40 Hz band-pass filter)
    raw.filter(0.5, 40., fir_design='firwin')

    # Re-reference to average
    raw.set_eeg_reference('average', projection=True)

    # Use ICA for artifact removal
    ica = mne.preprocessing.ICA(n_components=15, random_state=97, max_iter=800)
    ica.fit(raw)

    # Manually inspect the ICA components to find those representing artifacts
    ica.plot_components()

    # Manually select artifact components
    # After inspecting the plots, you would select the appropriate components
    # For demonstration, let's assume components [0, 1] are artifacts
    ica.exclude = [0, 1]

    # Apply ICA to remove the selected components
    ica.apply(raw)

    cleaned_data = raw.get_data()

    # Save the cleaned data
    base_name = os.path.basename(file_path)
    output_file_path = os.path.join(output_directory, base_name)
    scipy.io.savemat(output_file_path, {variable_name: cleaned_data})
#ggg
# Directory containing the .mat files
directory = "DB"
output_directory = "Cleaned_EEG_data"
if not os.path.exists(output_directory):
    os.makedirs(output_directory)

# Process each .mat file in the directory
for file_name in os.listdir(directory):
    if file_name.endswith('.mat'):
        file_path = os.path.join(directory, file_name)
        clean_eeg_data(file_path, output_directory)
        print(f"Processed and saved cleaned data for {file_name}")
