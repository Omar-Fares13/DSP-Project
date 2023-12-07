import tkinter as tk
from tkinter import filedialog, simpledialog
from tkinter import ttk
import matplotlib.pyplot as plt
import numpy as np
import os

# Global variables to store data
data1 = None
data2 = None

def open_file_1():
    global data1
    file_path = filedialog.askopenfilename(filetypes=[("Text Files", "*.txt")])
    if file_path:
        with open(file_path, 'r') as file:
            lines = file.readlines()
            signal_type = int(lines[0])
            is_periodic = int(lines[1])
            num_samples = int(lines[2])
            data_lines = lines[3:]

            if signal_type == 0:  # Time domain
                data1 = [(int(index), float(amplitude)) for index, amplitude in (line.split() for line in data_lines)]
            elif signal_type == 1:  # Frequency domain
                data1 = [(float(freq), float(amplitude), float(phase)) for freq, amplitude, phase in (line.split() for line in data_lines)]
        print("data components loaded successfully.")

def open_file_2():
    global data2
    file_path = filedialog.askopenfilename(filetypes=[("Text Files", "*.txt")])
    if file_path:
        with open(file_path, 'r') as file:
            lines = file.readlines()
            signal_type = int(lines[0])
            is_periodic = int(lines[1])
            num_samples = int(lines[2])
            data_lines = lines[3:]

            if signal_type == 0:  # Time domain
                data2 = [(int(index), float(amplitude)) for index, amplitude in (line.split() for line in data_lines)]
            elif signal_type == 1:  # Frequency domain
                data2 = [(float(freq), float(amplitude), float(phase)) for freq, amplitude, phase in (line.split() for line in data_lines)]
        print("data components loaded successfully.")


def generate_signal():
    # Create a new pop-up window
    popup = tk.Toplevel(root)
    popup.title("Signal Parameters")

    # Signal Type
    signal_type_label = ttk.Label(popup, text="Signal Type:")
    signal_type_label.grid(row=0, column=0, pady=5, sticky='w')
    signal_type = ttk.Combobox(popup, values=['Sine', 'Cosine'], state='readonly')
    signal_type.grid(row=0, column=1, pady=5)

    # Amplitude
    amplitude_label = ttk.Label(popup, text="Amplitude (A):")
    amplitude_label.grid(row=1, column=0, pady=5, sticky='w')
    amplitude_entry = ttk.Entry(popup)
    amplitude_entry.grid(row=1, column=1, pady=5)

    # Phase Shift
    phase_shift_label = ttk.Label(popup, text="Phase Shift (θ) in degrees:")
    phase_shift_label.grid(row=2, column=0, pady=5, sticky='w')
    phase_shift_entry = ttk.Entry(popup)
    phase_shift_entry.grid(row=2, column=1, pady=5)

    # Analog Frequency
    analog_frequency_label = ttk.Label(popup, text="Analog Frequency (Hz):")
    analog_frequency_label.grid(row=3, column=0, pady=5, sticky='w')
    analog_frequency_entry = ttk.Entry(popup)
    analog_frequency_entry.grid(row=3, column=1, pady=5)

    # Sampling Frequency
    sampling_frequency_label = ttk.Label(popup, text="Sampling Frequency (Hz):")
    sampling_frequency_label.grid(row=4, column=0, pady=5, sticky='w')
    sampling_frequency_entry = ttk.Entry(popup)
    sampling_frequency_entry.grid(row=4, column=1, pady=5)

    def close_popup():
        popup.destroy()

    def generate_and_plot():
        A = float(amplitude_entry.get())
        theta = float(phase_shift_entry.get())
        analog_freq = float(analog_frequency_entry.get())
        sampling_freq = float(sampling_frequency_entry.get())

        t = np.linspace(0, 1, int(sampling_freq), endpoint=False)
        signal = A * np.cos(2 * np.pi * analog_freq * t + np.radians(theta)) if signal_type.get() == 'Cosine' else A * np.sin(2 * np.pi * analog_freq * t + np.radians(theta))

        plt.plot(t, signal)
        plt.title('Generated Signal')
        plt.xlabel('Time (s)')
        plt.ylabel('Amplitude')

        plt.axhline(0, color='black', linewidth=0.5)  # Horizontal axis
        plt.axvline(0, color='black', linewidth=0.5)  # Vertical axis

        plt.show()

        close_popup()

    # Generate Button in the pop-up
    generate_button_popup = ttk.Button(popup, text="Generate and Plot", command=generate_and_plot)
    generate_button_popup.grid(row=5, column=0, columnspan=2, pady=10)

    # Show the pop-up window
    popup.mainloop()

def perform_addition():
    global data1, data2
    if data1 is None:
        print("Please load the first signal before performing addition.")
        return

    if data2 is None:
        print("Please load the second signal for addition.")
        open_file_2()
        if data2 is None:
            return

    result = []

    if len(data1) != len(data2):
        print("Signals must have the same length for addition.")
        return

    for i in range(len(data1)):
        index1, amplitude1 = data1[i]
        index2, amplitude2 = data2[i]
        result.append((index1, amplitude1 + amplitude2))

    print(f"Addition result: {result}")
    plot_signal(result)

def perform_subtraction():
    global data1, data2
    if data1 is None:
        print("Please load the first signal before performing subtraction.")
        return

    if data2 is None:
        print("Please load the second signal for subtraction.")
        open_file_2()
        if data2 is None:
            return

    result = []

    if len(data1) != len(data2):
        print("Signals must have the same length for subtraction.")
        return

    for i in range(len(data1)):
        index1, amplitude1 = data1[i]
        index2, amplitude2 = data2[i]
        result.append((index1, amplitude1 - amplitude2))

    print(f"Subtraction result: {result}")
    plot_signal(result)

def perform_multiplication():
    global data1
    if data1 is None:
        print("Please load the signal before performing multiplication.")
        return

    try:
        constant = float(simpledialog.askstring("Input", "Enter a constant for multiplication:"))
        result = [(index, amplitude * constant) for index, amplitude in data1]
        print(f"Multiplication result: {result}")
        plot_signal(result)
    except ValueError:
        print("Invalid input. Please enter a number.")    

def perform_squaring():
    global data1
    if data1 is None:
        print("Please load the signal before performing squaring.")
        return

    result = [(index, amplitude**2) for index, amplitude in data1]
    print(f"Squaring result: {result}")
    plot_signal(result)

def perform_shifting():
    global data1
    if data1 is None:
        print("Please load the signal before performing shifting.")
        return

    try:
        constant = float(simpledialog.askstring("Input", "Enter a constant for shifting:"))
        result = [(index, amplitude + constant) for index, amplitude in data1]
        print(f"Shifting result: {result}")
        plot_signal(result)
    except ValueError:
        print("Invalid input. Please enter a number.")

def perform_normalization():
    global data1
    if data1 is None:
        print("Please load the signal before performing normalization.")
        return

    try:
        normalization_type = simpledialog.askstring("Input", "Enter '0' for -1 to 1 normalization or '1' for 0 to 1 normalization:")
        if normalization_type not in ('0', '1'):
            print("Invalid input. Please enter '0' or '1'.")
            return

        if normalization_type == '0':
            max_value = max([abs(amplitude) for _, amplitude in data1])
            result = [(index, amplitude / max_value) for index, amplitude in data1]
        else:
            max_value = max([amplitude for _, amplitude in data1])
            min_value = min([amplitude for _, amplitude in data1])
            result = [(index, (amplitude - min_value) / (max_value - min_value)) for index, amplitude in data1]

        print(f"Normalization result: {result}")
        plot_signal(result)
    except ValueError:
        print("Invalid input. Please enter a number.")

def perform_accumulation():
    global data1
    if data1 is None:
        print("Please load the signal before performing accumulation.")
        return

    result = []
    accumulated_value = 0

    for index, amplitude in data1:
        accumulated_value += amplitude
        result.append((index, accumulated_value))

    print(f"Accumulation result: {result}")
    plot_signal(result)

def plot_signal(data):
    plt.figure(figsize=(8, 4))
    indices, amplitudes = zip(*data)
    plt.plot(indices, amplitudes, 'bo-')
    plt.title('Signal')
    plt.xlabel('Index/Sample')
    plt.ylabel('Amplitude')
    plt.grid(True, linestyle='--', alpha=0.7)  # Add horizontal grid
    plt.show()


def perform_quantization():
    global data1
    if data1 is None:
        print("Please load the signal before performing quantization.")
        return

    try:
        # Create a dialog window for user choice
        choice_window = tk.Toplevel(root)
        choice_window.title("Quantization Choice")

        def choose_levels():
            choice_window.destroy()
            num_levels = int(simpledialog.askstring("Input", "Enter the number of levels:"))
            if num_levels <= 0:
                print("Invalid input. Please enter a positive number of levels.")
                return
            perform_quantization_with_levels(num_levels)

        def choose_bits():
            choice_window.destroy()
            num_bits = int(simpledialog.askstring("Input", "Enter the number of bits:"))
            if num_bits <= 0:
                print("Invalid input. Please enter a positive number of bits.")
                return
            num_levels = 2**num_bits
            perform_quantization_with_levels(num_levels)

        levels_button = tk.Button(choice_window, text="Choose Levels", command=choose_levels)
        levels_button.pack(pady=10)

        bits_button = tk.Button(choice_window, text="Choose Bits", command=choose_bits)
        bits_button.pack(pady=10)

    except ValueError:
        print("Invalid input. Please enter a valid number of levels or bits.")

def perform_quantization_with_levels(num_levels):
    global data1

    # Find the range of the signal
    max_amplitude = max([amplitude for _, amplitude in data1])
    min_amplitude = min([amplitude for _, amplitude in data1])
    signal_range = max_amplitude - min_amplitude

    # Calculate the quantization step size
    step_size = signal_range / num_levels

    # Quantize the signal
    quantized_signal = [(index, round(amplitude / step_size) * step_size) for index, amplitude in data1]

    # Calculate quantization error
    quantization_error = [(index, amplitude - quantized_amplitude) for (index, amplitude), (_, quantized_amplitude) in zip(data1, quantized_signal)]

    # Plot original signal, quantized signal, and quantization error
    plt.figure(figsize=(12, 6))

    # Original Signal
    plt.subplot(131)
    indices, amplitudes = zip(*data1)
    plt.plot(indices, amplitudes, 'bo-')
    plt.title('Original Signal')
    plt.xlabel('Index/Sample')
    plt.ylabel('Amplitude')
    plt.grid(True, linestyle='--', alpha=0.7)

    # Quantized Signal
    plt.subplot(132)
    indices, amplitudes = zip(*quantized_signal)
    plt.plot(indices, amplitudes, 'ro-')
    plt.title('Quantized Signal')
    plt.xlabel('Index/Sample')
    plt.ylabel('Amplitude')
    plt.grid(True, linestyle='--', alpha=0.7)

    # Quantization Error
    plt.subplot(133)
    indices, amplitudes = zip(*quantization_error)
    plt.plot(indices, amplitudes, 'go-')
    plt.title('Quantization Error')
    plt.xlabel('Index/Sample')
    plt.ylabel('Error')
    plt.grid(True, linestyle='--', alpha=0.7)

    plt.tight_layout()
    plt.show()

def custom_dft(s):
    N = len(s)
    X = np.zeros(N, dtype=np.complex128)
    for k in range(N):
        X[k] = sum(s[n] * np.exp(-1j * 2 * np.pi * k * n / N) for n in range(N))
    return X


def DFT():
    global data1
    if data1 is None:
        print("Please load the signal.")
        return
    
    try:
        HZ_input = simpledialog.askstring("Input", "Enter a frequency in HZ:")
        if HZ_input is None:
            # User clicked Cancel
            return

        HZ = float(HZ_input)
        signal = [(sequence) for index, sequence in data1]
        dft_result = custom_dft(signal)
        num_samples = len(signal)
        frequencies = [k * HZ / num_samples for k in range(num_samples)]
        amplitude = np.abs(dft_result)
        phase = np.angle(dft_result)

        # Save amplitude and phase to a text file
        with open("frequency_components.txt", "w") as file:
            # Write signal type, is_periodic, and num_samples
            file.write("0\n")
            file.write("0\n")
            file.write(f"{num_samples}\n")

            # Write amplitude and phase
            for a, p in zip(amplitude, phase):
                file.write(f"{a} {p}\n")

        # Plot frequency spectrum (same as before)
        plt.subplot(211)
        plt.stem(frequencies, amplitude)
        plt.xlabel('Frequency')
        plt.ylabel('Amplitude')
        plt.title('Frequency / Amplitude')

        plt.subplot(212)
        plt.stem(frequencies, phase)
        plt.xlabel('Frequency')
        plt.ylabel('Phase')
        plt.title('Frequency / Phase')
        
        plt.tight_layout()
        plt.show()

        print("Amplitude and phase saved to 'frequency_components.txt'.")
        
        # Call the modify function
        modify(num_samples, frequencies, amplitude, phase)

    except ValueError:
        print("Invalid input. Please enter a valid number.")


def modify(num_samples, frequencies, amplitude, phase): 
    print("\nModify Amplitude and Phase:")
    for k in range(num_samples):
        try:
            new_amp = float(simpledialog.askstring("Input", f"Enter new amplitude for component at {frequencies[k]} Hz: "))
            new_phase = float(simpledialog.askstring("Input", f"Enter new phase for component at {frequencies[k]} Hz: "))
        
            amplitude[k] = new_amp
            phase[k] = new_phase
        except ValueError:
            print("ERROR!!!")  
    plt.subplot(211)
    plt.stem(frequencies, amplitude)
    plt.xlabel('Frequency')
    plt.ylabel('Amplitude')
    plt.title('Frequency / Amplitude')
    # Plot frequency versus phase
    plt.subplot(212)
    plt.stem(frequencies, phase)
    plt.xlabel('Frequency')
    plt.ylabel('Phase')
    plt.title('Frequency / Phase')
    plt.tight_layout()
    plt.show()

def IDFT():
    global data1
    try:
        if data1 is None:
            print("Please load the frequency components.")
            return

        num_samples = len(data1)
        time_values = np.linspace(0, 1, num_samples, endpoint=False)
        signal_values = np.zeros(num_samples)

        for n in range(num_samples):
            for k in range(num_samples):
                amplitude, phase = data1[k]
                signal_values[n] += amplitude * np.cos(2 * np.pi * k * time_values[n] + phase)/num_samples

        # Plot the reconstructed signal
        plt.figure(figsize=(8, 4))
        plt.plot(time_values, signal_values, 'b-')
        plt.title('Reconstructed Signal')
        plt.xlabel('Time')
        plt.ylabel('Amplitude')
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.show()

        # Print the reconstructed signal
        print(f"Reconstructed Signal: {signal_values}")

    except Exception as e:
        print(f"An error occurred: {e}")


def compute_dct():
    global data1
    if data1 is None:
        print("Please load the signal.")
        return

    try:
        N = len(data1)
        dct_result = np.zeros(N)
        signal_values = [amplitude for  _,amplitude in data1]

        for k in range(N):
            dct_result[k] = np.sqrt(2/N) * np.sum(signal_values * np.cos(np.pi/(4*N) * (2*np.arange(0, N) - 1) * (2*k - 1)))

        # Get user input for the number of coefficients to save
        m = int(simpledialog.askstring("Input", "Enter the number of coefficients to save: "))

        if m > N:
            print("Error: Number of coefficients to save exceeds the length of the DCT result.")
            return

        # Save the first m coefficients in a text file
        with open("coefficients.txt", 'w') as file:
            file.write("0\n")  # Signal type (0 for time domain)
            file.write("1\n")  # Is periodic (1 for yes, 0 for no)
            file.write(f"{m}\n")  # Number of samples

    # Write amplitude and phase
            for coeff in dct_result[:m]:
                file.write(f"0 {coeff}\n")

        print(f"First {m} DCT coefficients saved to coefficients.txt")

        # Display DCT result
        plt.figure(figsize=(8, 4))
        plt.stem(dct_result)
        plt.title('Discrete Cosine Transform (DCT)')
        plt.xlabel('Coefficient Index (k)')
        plt.ylabel('DCT Coefficient Value')
        plt.show()

    except ValueError as e:
        print(f"Error: {e}")


def remove_dc_component():
    global data1
    if data1 is None:
        print("Please load the signal.")
        return

    try:
        # Extract the signal values
        signal_values = [amplitude for _, amplitude in data1]

        # Remove DC component by subtracting the mean value
        mean_value = np.mean(signal_values)
        signal_without_dc = signal_values - mean_value

        # Round the modified signal values to 3 digits
        signal_without_dc_rounded = np.round(signal_without_dc, 3)

        # Update data1 with the modified signal
        data1 = list(enumerate(signal_without_dc_rounded))

        # Print the modified signal
        print("Modified Signal:")
        for index, amplitude in data1:
            print(f"{index} {amplitude:.3f}")

        # Print the modified signal values
        modified_signal_values = [amplitude for _, amplitude in data1]
        print(f"Modified Signal Values: {modified_signal_values}")

        # Plot the original and modified signals
        plt.figure(figsize=(10, 5))

        plt.subplot(2, 1, 1)
        plt.plot(signal_values, label='Original Signal')
        plt.title('Original Signal')
        plt.xlabel('Sample Index')
        plt.ylabel('Amplitude')
        plt.legend()

        plt.subplot(2, 1, 2)
        plt.plot(signal_without_dc_rounded, label='Signal without DC Component')
        plt.title('Signal without DC Component')
        plt.xlabel('Sample Index')
        plt.ylabel('Amplitude')
        plt.legend()

        plt.tight_layout()
        plt.show()

        print("DC component removed successfully.")

    except ValueError as e:
        print(f"Error: {e}")


def Smoothing():
    global data1
    if data1 is None:
        print("Please load the signal before performing smoothing.")
        return

    try:
        # Get the window size from the user
        window_size = simpledialog.askinteger("Input", "Enter the number of points included in averaging:")

        if window_size is None or window_size <= 0:
            print("Invalid window size. Please enter a positive integer.")
            return

        num_points = len(data1)
        moving_avg = []

        for n in range(num_points):
            start = max(0, n - window_size + 1)
            end = n + 1
            window = data1[start:end]
            avg_value = sum(amplitude for _, amplitude in window) / len(window)
            moving_avg.append((n, avg_value))

        print(f"Smoothing result: {moving_avg}")
        plot_signal(moving_avg)

    except ValueError:
        print("Invalid input. Please enter a valid number.")


def DerivativeSignal():
    InputSignal = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0, 17.0, 18.0, 19.0, 20.0, 21.0, 22.0, 23.0, 24.0, 25.0, 26.0, 27.0, 28.0, 29.0, 30.0, 31.0, 32.0, 33.0, 34.0, 35.0, 36.0, 37.0, 38.0, 39.0, 40.0, 41.0, 42.0, 43.0, 44.0, 45.0, 46.0, 47.0, 48.0, 49.0, 50.0, 51.0, 52.0, 53.0, 54.0, 55.0, 56.0, 57.0, 58.0, 59.0, 60.0, 61.0, 62.0, 63.0, 64.0, 65.0, 66.0, 67.0, 68.0, 69.0, 70.0, 71.0, 72.0, 73.0, 74.0, 75.0, 76.0, 77.0, 78.0, 79.0, 80.0, 81.0, 82.0, 83.0, 84.0, 85.0, 86.0, 87.0, 88.0, 89.0, 90.0, 91.0, 92.0, 93.0, 94.0, 95.0, 96.0, 97.0, 98.0, 99.0, 100.0]
    expectedOutput_first = [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
    expectedOutput_second = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]

    FirstDrev = [InputSignal[i] - InputSignal[i - 1] for i in range(1, len(InputSignal))]
    
    SecondDrev = [InputSignal[i + 1] - 2 * InputSignal[i] + InputSignal[i - 1] for i in range(1, len(InputSignal) - 1)]
    
    if (len(FirstDrev) != len(expectedOutput_first)) or (len(SecondDrev) != len(expectedOutput_second)):
        print("mismatch in length")
        return
    first = second = True
    for i in range(len(expectedOutput_first)):
        if abs(FirstDrev[i] - expectedOutput_first[i]) < 0.01:
            continue
        else:
            first = False
            print("1st derivative wrong")
            return
    for i in range(len(expectedOutput_second)):
        if abs(SecondDrev[i] - expectedOutput_second[i]) < 0.01:
            continue
        else:
            second = False
            print("2nd derivative wrong")
            return
    if first and second:
        print("Derivative Test case passed successfully")
    else:
        print("Derivative Test case failed")
    return



def delay_signal():
    global data1
    if data1 is None:
        print("Please load the signal.")
        return
    
    try:
        # Get user input for delay steps
        k = int(simpledialog.askstring("Input", "Enter the number of steps for delaying:"))

        print("Original Signal:", amplitude_values)

        # Extract the amplitude values from data1
        amplitude_values = np.array([amplitude for _, amplitude in data1])
        for i in range(len(amplitude_values)):
            amplitude_values[i]+=k


        # Print the original and delayed signals
        print("Delayed Signal:", amplitude_values)

        # Plot the original and delayed signals
        plt.figure(figsize=(10, 5))

        plt.subplot(2, 1, 1)
        plt.plot(amplitude_values, label='Original Signal')
        plt.title('Original Signal')
        plt.xlabel('Sample Index')
        plt.ylabel('Amplitude')
        plt.legend()

        plt.subplot(2, 1, 2)
        plt.plot(delayed_signal, label=f'Delayed Signal (by {k} steps)')
        plt.title(f'Delayed Signal (by {k} steps)')
        plt.xlabel('Sample Index')
        plt.ylabel('Amplitude')
        plt.legend()

        plt.tight_layout()
        plt.show()

        print(f"Signal delayed successfully by {k} steps.")

    except ValueError as e:
        print(f"Error: {e}")




def advance_signal():
    global data1
    if data1 is None:
        print("Please load the signal.")
        return
    
    try:
        # Get user input for advance steps
        k = int(simpledialog.askstring("Input", "Enter the number of steps for advancing:"))

        # Extract the amplitude values from data1
        amplitude_values = np.array([amplitude for _, amplitude in data1])
        for i in range(len(amplitude_values)):
            amplitude_values[i]-=k

        # Perform the advancement by adding k zeros at the end
        advanced_signal = np.concatenate((amplitude_values[k:], [0.0] * k))

        # Print the original and advanced signals
        print("Original Signal:", amplitude_values)
        print("Advanced Signal (by {} steps):".format(k), advanced_signal)

        # Plot the original and advanced signals
        plt.figure(figsize=(10, 5))

        plt.subplot(2, 1, 1)
        plt.plot(amplitude_values, label='Original Signal')
        plt.title('Original Signal')
        plt.xlabel('Sample Index')
        plt.ylabel('Amplitude')
        plt.legend()

        plt.subplot(2, 1, 2)
        plt.plot(advanced_signal, label=f'Advanced Signal (by {k} steps)')
        plt.title(f'Advanced Signal (by {k} steps)')
        plt.xlabel('Sample Index')
        plt.ylabel('Amplitude')
        plt.legend()

        plt.tight_layout()
        plt.show()

        print(f"Signal advanced successfully by {k} steps.")

    except ValueError as e:
        print(f"Error: {e}")


def fold_signal():
    global data1
    if data1 is None:
        print("Please load the signal.")
        return

    # Original signal
    original_signal = np.array(data1)
    
    # Fold the signal
    folded_signal = np.array([(-y, x) for y, x in reversed(data1)])
    data1 = folded_signal

    # Plot the original and folded signals
    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    plt.plot(original_signal[:, 1], original_signal[:, 0], marker='o', linestyle='-', color='b')
    plt.title('Original Signal')
    plt.xlabel('Index')
    plt.ylabel('Amplitude')

    plt.subplot(1, 2, 2)
    plt.plot(folded_signal[:, 1], folded_signal[:, 0], marker='o', linestyle='-', color='r')
    plt.title('Folded Signal')
    plt.xlabel('Index')
    plt.ylabel('Amplitude')

    plt.tight_layout()
    plt.show()
    print(folded_signal)
    return folded_signal

def delayFoldedSignal():

    fold_signal()
    advance_signal()


def advanceFoldedSignal():

    fold_signal()
    delay_signal()


def custom_idft(signal):
    N = len(signal)
    idft_result = np.zeros(N, dtype=np.complex128)

    for n in range(N):
        for k in range(N):
            idft_result[n] += signal[k] * np.exp(2j * np.pi * k * n / N)

    idft_result /= N  # Normalize by the number of samples

    return idft_result

def remove_dc_component2():
    global data1
    if data1 is None:
        print("Please load the signal.")
        return

    try:
        # Convert the signal to the frequency domain using DFT
        signal = [sequence for index, sequence in data1]
        dft_result = custom_dft(signal)

        # Set the DC component to zero
        dft_result[0] = 0

        # Apply the inverse DFT to get the modified signal
        modified_signal = custom_idft(dft_result)

        # Update the data1 list with the modified signal
        data1 = [(index, value) for index, value in enumerate(modified_signal)]

        print("DC component removed successfully.")
        # Plot the modified signal
        plt.figure(figsize=(8, 4))
        plt.plot(*zip(*data1), marker='o', linestyle='-', color='b')
        plt.title('Signal after Removing DC Component')
        plt.xlabel('Index')
        plt.ylabel('Amplitude')
        plt.grid(True)
        plt.show()
    except ValueError as e:
        print(f"Error: {e}")


def convolve_signals():
    global data1, data2
    if data1 is None or data2 is None:
        print("Please load both signals.")
        return

    try:
        # Extract amplitude values from data1 and data2
        signal1_values = np.array([amplitude for _, amplitude in data1])
        signal2_values = np.array([amplitude for _, amplitude in data2])

        # Length of signals
        len_signal1 = len(signal1_values)
        len_signal2 = len(signal2_values)

        # Length of the result
        len_result = len_signal1 + len_signal2 - 1

        # Perform convolution manually
        convolution_result = [
            sum(
                signal1_values[i] * signal2_values[j - i]
                for i in range(max(0, j - len_signal2 + 1), min(len_signal1, j + 1))
            )
            for j in range(len_result)
        ]

        # Update data1 with the convolution result
        data1 = list(enumerate(convolution_result))

        # Print the convolution result
        print("Convolution Result:", convolution_result)
        # Plot the original signals and the convolution result
        plt.figure(figsize=(12, 5))

        plt.subplot(3, 1, 1)
        plt.plot(signal1_values, label='Signal 1')
        plt.title('Signal 1')
        plt.xlabel('Sample Index')
        plt.ylabel('Amplitude')
        plt.legend()

        plt.subplot(3, 1, 2)
        plt.plot(signal2_values, label='Signal 2')
        plt.title('Signal 2')
        plt.xlabel('Sample Index')
        plt.ylabel('Amplitude')
        plt.legend()

        plt.subplot(3, 1, 3)
        plt.plot(convolution_result, label='Convolution Result')
        plt.title('Convolution Result')
        plt.xlabel('Sample Index')
        plt.ylabel('Amplitude')
        plt.legend()

        plt.tight_layout()
        plt.show()

        print("Signals convolved successfully.")

    except ValueError as e:
        print(f"Error: {e}")


def correlation():
    global data1, data2
    if data1 is None or data2 is None:
        print("Please load both signals.")
        return

    try:
        # Extract amplitude values from data1 and data2
        signal1_values = np.array([amplitude for _, amplitude in data1])
        signal2_values = np.array([amplitude for _, amplitude in data2])

        N = len(signal2_values)
    
        # Compute the cross-correlation function
        r_12 = np.zeros(N)
        for n in range(N):
            for j in range(N):
                r_12[j] += signal1_values[n] * signal2_values[(n + j)%N]/N

    
        # Compute the root mean square (RMS) values
        rms_x1 = np.sum(signal1_values**2)
        rms_x2 = np.sum(signal2_values**2)
    
        # Compute the denominator
        denominator = np.sqrt(rms_x1 * rms_x2)/N
    
        # Normalize the cross-correlation function
        rho_12 = r_12 / denominator

        # Plot the result
        plt.stem(rho_12)
        plt.xlabel('Shift')
        plt.ylabel('Normalized Cross-Correlation')
        plt.title('Normalized Cross-Correlation of Two Signals')
        plt.show()

        print("Normalized Cross-Correlation Result:", rho_12)

    except ValueError as e:
        print(f"Error: {e}")


def time_delay_analysis():

    global data1, data2
    if data1 is None or data2 is None:
        print("Please load both signals.")
        return

    try:
        # Extract amplitude values from signal1 and signal2
        signal1_values = np.array([amplitude for _, amplitude in data1])
        signal2_values = np.array([amplitude for _, amplitude in data2])

        N = len(signal2_values)
    
        # Compute the cross-correlation function
        r_12 = np.zeros(N)
        for n in range(N):
            for j in range(N):
                r_12[j] += signal1_values[n] * signal2_values[(n + j)%N]/N

    
        # Compute the root mean square (RMS) values
        rms_x1 = np.sum(signal1_values**2)
        rms_x2 = np.sum(signal2_values**2)
    
        # Compute the denominator
        denominator = np.sqrt(rms_x1 * rms_x2)/N
    
        # Normalize the cross-correlation function
        rho_12 = r_12 / denominator

        # Find the time delay (index of the maximum value in the cross-correlation)
        sampling_period = float(simpledialog.askstring("Input", "Enter the sampling period:"))
        ts = 1 / sampling_period 
        delay_index = np.argmax(rho_12)
        delay = delay_index * ts

        # Plot the cross-correlation result
        plt.stem(rho_12)
        plt.xlabel('Time Shift')
        plt.ylabel('Cross-Correlation')
        plt.title('Cross-Correlation for Time Delay Analysis')
        plt.show()

        print("Excpected output:", delay)
    except ValueError as e:
        print(f"Error: {e}")



def custom_correlation(x1 , x2):
    try:
        N = len(x2)
    
        # Compute the cross-correlation function
        r_12 = np.zeros(N)
        for n in range(N):
            for j in range(N):
                r_12[j] +=x1[n] *x2[(n + j)%N]/N

    
        # Compute the root mean square (RMS) values
        rms_x1 = np.sum(x1**2)
        rms_x2 = np.sum(x2**2)
    
        # Compute the denominator
        denominator = np.sqrt(rms_x1 * rms_x2)/N
    
        # Normalize the cross-correlation function
        rho_12 = r_12 / denominator

        return rho_12

    except ValueError as e:
        print(f"Error: {e}")


def calculate_average_signal(folder_path):
    files = os.listdir(folder_path)
    num_files = len(files)

    # Read the first file to get the number of rows
    first_file_path = os.path.join(folder_path, files[0])
    with open(first_file_path, 'r') as f:
        first_signal = np.array([float(value) for value in f.read().splitlines()])

    num_rows = len(first_signal)

    # Initialize variables to store sum of signals
    sum_signal = np.zeros(num_rows)

    for file in files:
        file_path = os.path.join(folder_path, file)

        # Read signal values from the file
        with open(file_path, 'r') as f:
            signal = np.array([float(value) for value in f.read().splitlines()])

        # Update sum
        sum_signal += signal

    # Calculate the average signal
    average_signal = sum_signal / num_files

    return average_signal

def browse_button(entry):
    folder_path = filedialog.askdirectory()
    entry.delete(0, tk.END)
    entry.insert(0, folder_path)

def template_matching_popup():
    # Create a pop-up window
    popup = tk.Toplevel()
    popup.title("Template Matching")

    # Entry widgets for folder paths
    class1_entry = tk.Entry(popup, width=40)
    class2_entry = tk.Entry(popup, width=40)
    test_entry = tk.Entry(popup, width=40)

    # Browse buttons for selecting folders
    class1_button = tk.Button(popup, text="Browse", command=lambda: browse_button(class1_entry))
    class2_button = tk.Button(popup, text="Browse", command=lambda: browse_button(class2_entry))
    test_button = tk.Button(popup, text="Browse", command=lambda: browse_button(test_entry))

    # Place widgets in the pop-up window
    tk.Label(popup, text="Class 1 Folder:").grid(row=0, column=0)
    class1_entry.grid(row=0, column=1)
    class1_button.grid(row=0, column=2)

    tk.Label(popup, text="Class 2 Folder:").grid(row=1, column=0)
    class2_entry.grid(row=1, column=1)
    class2_button.grid(row=1, column=2)

    tk.Label(popup, text="Test Folder:").grid(row=2, column=0)
    test_entry.grid(row=2, column=1)
    test_button.grid(row=2, column=2)

    # Run the template matching function
    run_button = tk.Button(popup, text="Run Template Matching", command=lambda: template_matching(class1_entry, class2_entry, test_entry, popup))
    run_button.grid(row=3, column=0, columnspan=3, pady=10)

def template_matching(class1_entry, class2_entry, test_entry, popup):
    try:
        # Get folder paths from entry widgets
        class1_folder_path = class1_entry.get()
        class2_folder_path = class2_entry.get()
        test_folder_path = test_entry.get()

        # Calculate average signals for each class
        avg_signal_class1 = calculate_average_signal(class1_folder_path)
        avg_signal_class2 = calculate_average_signal(class2_folder_path)

        # Get the list of test files
        test_files = os.listdir(test_folder_path)

        for test_file in test_files:
            test_file_path = os.path.join(test_folder_path, test_file)

            # Read test signal values from the file
            with open(test_file_path, 'r') as f:
                test_signal = np.array([float(value) for value in f.read().splitlines()])

            # Calculate correlations with average signals
            correlation_class1 = custom_correlation(test_signal , avg_signal_class1)
            correlation_class2 = custom_correlation(test_signal , avg_signal_class2)

            # Determine the class based on correlation
            if max(correlation_class1) > max(correlation_class2):
             print(f"{test_file} down movement of EOG signal")
            else:
             print(f"{test_file} up movement of EOG signal")
        # Close the pop-up window
        popup.destroy()

    except Exception as e:
        print(f"Error: {e}")




# Create a main window
root = tk.Tk()
root.title("Signal Viewer")

# Create a notebook (tabs)
notebook = ttk.Notebook(root)
notebook.pack(fill='both', expand=True)

# Frame for file-related buttons
file_frame = ttk.Frame(notebook)
notebook.add(file_frame, text='File')

open_button_1 = tk.Button(file_frame, text="Open File 1", command=open_file_1, padx=10, pady=5, width=15, height=2)
open_button_1.pack(side=tk.LEFT, padx=5, anchor='center')

open_button_2 = tk.Button(file_frame, text="Open File 2", command=open_file_2, padx=10, pady=5, width=15, height=2)
open_button_2.pack(side=tk.LEFT, padx=5, anchor='center')

generate_signal_button = tk.Button(file_frame, text="Generate Signal", command=generate_signal, padx=10, pady=5, width=15, height=2)
generate_signal_button.pack(side=tk.LEFT, padx=5, anchor='center')

# Frame for arithmetic operation buttons
arithmetic_frame = ttk.Frame(notebook)
notebook.add(arithmetic_frame, text='Arithmetic')

addition_button = tk.Button(arithmetic_frame, text="Perform Addition", command=perform_addition, padx=10, pady=5, width=15, height=2)
addition_button.pack(side=tk.LEFT, padx=5, anchor='center')

subtraction_button = tk.Button(arithmetic_frame, text="Perform Subtraction", command=perform_subtraction, padx=10, pady=5, width=15, height=2)
subtraction_button.pack(side=tk.LEFT, padx=5, anchor='center')

multiplication_button = tk.Button(arithmetic_frame, text="Perform Multiplication", command=perform_multiplication, padx=10, pady=5, width=15, height=2)
multiplication_button.pack(side=tk.LEFT, padx=5, anchor='center')

squaring_button = tk.Button(arithmetic_frame, text="Perform Squaring", command=perform_squaring, padx=10, pady=5, width=15, height=2)
squaring_button.pack(side=tk.LEFT, padx=5, anchor='center')

# Frame for other operations
operation_frame = ttk.Frame(notebook)
notebook.add(operation_frame, text='Other Operations')

shifting_button = tk.Button(operation_frame, text="Perform Shifting", command=perform_shifting, padx=10, pady=5, width=15, height=2)
shifting_button.pack(side=tk.LEFT, padx=5, anchor='center')

normalization_button = tk.Button(operation_frame, text="Perform Normalization", command=perform_normalization, padx=10, pady=5, width=15, height=2)
normalization_button.pack(side=tk.LEFT, padx=5, anchor='center')

accumulation_button = tk.Button(operation_frame, text="Perform Accumulation", command=perform_accumulation, padx=10, pady=5, width=15, height=2)
accumulation_button.pack(side=tk.LEFT, padx=5, anchor='center')

# Frame for plotting buttons
plotting_frame = ttk.Frame(notebook)
notebook.add(plotting_frame, text='Plotting')

plot_button = tk.Button(plotting_frame, text="Plot Signal", command=lambda: plot_signal(data1), padx=10, pady=5, width=15, height=2)
plot_button.pack(side=tk.LEFT, padx=5, anchor='center')

# Frame for frequency domain operations
frequency_frame = ttk.Frame(notebook)
notebook.add(frequency_frame, text='Frequency Domain')

dft_button = tk.Button(frequency_frame, text="DFT", command=DFT, padx=10, pady=5, width=15, height=2)
dft_button.pack(side=tk.LEFT, padx=5, anchor='center')

idft_button = tk.Button(frequency_frame, text="IDFT", command=IDFT, padx=10, pady=5, width=15, height=2)
idft_button.pack(side=tk.LEFT, padx=5, anchor='center')

# Frame for frequency domain operations
dct_frame = ttk.Frame(notebook)
notebook.add(dct_frame, text='DCT Operations')

dct_button = tk.Button(dct_frame, text="DCT", command=compute_dct, padx=10, pady=5, width=15, height=2)
dct_button.pack(side=tk.LEFT, padx=5, anchor='center')

remove_dc_button = tk.Button(dct_frame, text="Remove DC Component", command=remove_dc_component, padx=10, pady=5, width=15, height=2)
remove_dc_button.pack(side=tk.LEFT, padx=5, anchor='center')

# Frame for time domain operations
time_frame = ttk.Frame(notebook)
notebook.add(time_frame, text='Time Domain')

# Create two frames within the time domain frame
row1_frame = tk.Frame(time_frame)
row1_frame.pack(side=tk.TOP, pady=5)

row2_frame = tk.Frame(time_frame)
row2_frame.pack(side=tk.TOP, pady=5)

# Buttons for the first row
smoothing_button = tk.Button(row1_frame, text="Smoothing", command=Smoothing, padx=10, pady=5, width=15, height=2)
smoothing_button.pack(side=tk.LEFT, padx=5, anchor='center')

derivative_button = tk.Button(row1_frame, text="Derivative", command=DerivativeSignal, padx=10, pady=5, width=15, height=2)
derivative_button.pack(side=tk.LEFT, padx=5, anchor='center')

delay_button = tk.Button(row1_frame, text="Delaying", command=delay_signal, padx=10, pady=5, width=15, height=2)
delay_button.pack(side=tk.LEFT, padx=5, anchor='center')

advance_button = tk.Button(row1_frame, text="Advancing", command=advance_signal, padx=10, pady=5, width=15, height=2)
advance_button.pack(side=tk.LEFT, padx=5, anchor='center')

# Buttons for the second row
fold_button = tk.Button(row2_frame, text="Fold", command=fold_signal, padx=10, pady=5, width=15, height=2)
fold_button.pack(side=tk.LEFT, padx=5, anchor='center')

delay_folded_button = tk.Button(row2_frame, text="Delay Folded Signal", command=delayFoldedSignal, padx=10, pady=5, width=15, height=2)
delay_folded_button.pack(side=tk.LEFT, padx=5, anchor='center')

advance_folded_button = tk.Button(row2_frame, text="Advance Folded Signal", command=advanceFoldedSignal, padx=10, pady=5, width=15, height=2)
advance_folded_button.pack(side=tk.LEFT, padx=5, anchor='center')

remove_dc_component2_button = tk.Button(row2_frame, text="Remove DC Component 2", command=remove_dc_component2, padx=10, pady=5, width=15, height=2)
remove_dc_component2_button.pack(side=tk.LEFT, padx=5, anchor='center')

convolve_signals_button = tk.Button(row2_frame, text="Convolve Two Signals", command=convolve_signals, padx=10, pady=5, width=15, height=2)
convolve_signals_button.pack(side=tk.LEFT, padx=5, anchor='center')

# Frame for correlation operations
correlation_frame = ttk.Frame(notebook)
notebook.add(correlation_frame, text='Correlation')

correlation_button = tk.Button(correlation_frame, text="Correlation", command=correlation, padx=10, pady=5, width=15, height=2)
correlation_button.pack(side=tk.LEFT, padx=5, anchor='center')

time_delay_analysis_button = tk.Button(correlation_frame, text="time delay analysis", command=time_delay_analysis, padx=10, pady=5, width=15, height=2)
time_delay_analysis_button.pack(side=tk.LEFT, padx=5, anchor='center')

template_matching_button = tk.Button(correlation_frame, text="template matching", command=template_matching_popup, padx=10, pady=5, width=15, height=2)
template_matching_button.pack(side=tk.LEFT, padx=5, anchor='center')

# Start the main event loop
root.mainloop()