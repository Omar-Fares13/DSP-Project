import tkinter as tk
from tkinter import filedialog, simpledialog
import matplotlib.pyplot as plt
import numpy as np

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
                data1 = [(float(amplitude), float(phase)) for amplitude, phase in (line.split() for line in data_lines)]
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
                data2 = [(float(amplitude), float(phase)) for amplitude, phase in (line.split() for line in data_lines)]
        print("data components loaded successfully.")

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
            file.write("1\n")
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
        signal_values = [amplitude for amplitude, _ in data1]

        for k in range(N):
            dct_result[k] = np.sqrt(2/N) * np.sum(signal_values * np.cos(np.pi/(4*N) * (2*np.arange(1, N+1) - 1) * (2*k + 1)))

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
import tkinter as tk
from tkinter import filedialog, simpledialog

# Function definitions...

# Create a main window
root = tk.Tk()
root.title("Signal Viewer")

# Frame for file-related buttons
file_frame = tk.Frame(root)
file_frame.pack(pady=10)

open_button_1 = tk.Button(file_frame, text="Open File 1", command=open_file_1)
open_button_1.pack(side=tk.LEFT, padx=5)

open_button_2 = tk.Button(file_frame, text="Open File 2", command=open_file_2)
open_button_2.pack(side=tk.LEFT, padx=5)

# Frame for arithmetic operation buttons
arithmetic_frame = tk.Frame(root)
arithmetic_frame.pack(pady=10)

addition_button = tk.Button(arithmetic_frame, text="Perform Addition", command=perform_addition)
addition_button.pack(side=tk.LEFT, padx=5)

subtraction_button = tk.Button(arithmetic_frame, text="Perform Subtraction", command=perform_subtraction)
subtraction_button.pack(side=tk.LEFT, padx=5)

multiplication_button = tk.Button(arithmetic_frame, text="Perform Multiplication", command=perform_multiplication)
multiplication_button.pack(side=tk.LEFT, padx=5)

squaring_button = tk.Button(arithmetic_frame, text="Perform Squaring", command=perform_squaring)
squaring_button.pack(side=tk.LEFT, padx=5)

# Frame for other operations
operation_frame = tk.Frame(root)
operation_frame.pack(pady=10)

shifting_button = tk.Button(operation_frame, text="Perform Shifting", command=perform_shifting)
shifting_button.pack(side=tk.LEFT, padx=5)

normalization_button = tk.Button(operation_frame, text="Perform Normalization", command=perform_normalization)
normalization_button.pack(side=tk.LEFT, padx=5)

accumulation_button = tk.Button(operation_frame, text="Perform Accumulation", command=perform_accumulation)
accumulation_button.pack(side=tk.LEFT, padx=5)

# Frame for plotting buttons
plotting_frame = tk.Frame(root)
plotting_frame.pack(pady=10)

plot_button = tk.Button(plotting_frame, text="Plot Signal", command=lambda: plot_signal(data1))
plot_button.pack(side=tk.LEFT, padx=5)

# Frame for frequency domain operations
frequency_frame = tk.Frame(root)
frequency_frame.pack(pady=10)

dft_button = tk.Button(frequency_frame, text="DFT", command=DFT)
dft_button.pack(side=tk.LEFT, padx=5)

idft_button = tk.Button(frequency_frame, text="IDFT", command=IDFT)
idft_button.pack(side=tk.LEFT, padx=5)

dct_button = tk.Button(frequency_frame, text="DCT", command=compute_dct)
dct_button.pack(side=tk.LEFT, padx=5)

remove_dc_button = tk.Button(frequency_frame, text="Remove DC Component", command=remove_dc_component)
remove_dc_button.pack(side=tk.LEFT, padx=5)

# Start the main event loop
root.mainloop()
