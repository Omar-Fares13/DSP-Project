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
                data1 = [(float(freq), float(amplitude), float(phase_shift)) for freq, amplitude, phase_shift in (line.split() for line in data_lines)]

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
                data2 = [(float(freq), float(amplitude), float(phase_shift)) for freq, amplitude, phase_shift in (line.split() for line in data_lines)]
def open_file_IDFT():
    global data1
    file_path = filedialog.askopenfilename(filetypes=[("Text Files", "*.txt")])
    if file_path:
        with open(file_path, 'r') as file:
            lines = file.readlines()
            data_lines = lines[3:]
            data1 = data1 = [complex(line.split(',')[0].replace("f", "j")) + complex(line.split(',')[1].replace("f", "j")) for line in data_lines if line.strip()]
            print(data1)

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
        print("Please load the signal .")
        return
    
    try:
        HZ = float(simpledialog.askstring("Input", "Enter a frequency in HZ:"))
        signal = [(sequence) for index ,sequence in data1 ] 
        print(signal)
        dft_result = custom_dft(signal)
        print(dft_result)
        num_samples = len(signal)
        frequencies = [k * HZ / num_samples for k in range(num_samples)]
        amplitude = np.abs(dft_result)
        phase = np.angle(dft_result)
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
        plt.title('Frequency /  Phase')
        plt.tight_layout()
        plt.show()
        
        def modify() : 
            print("\nModify Amplitude and Phase:")
            for k in range(num_samples):
                try:
                    new_amp = float(simpledialog.askstring("Input",f"Enter new amplitude for component at {frequencies[k]} Hz: "))
                    new_phase = float(simpledialog.askstring("Input",f"Enter new phase for component at {frequencies[k]} Hz: "))
                
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
            plt.title('Frequency /  Phase')
            plt.tight_layout()
            plt.show()
        

        # Create buttons to modify amp/phase
        
        root = tk.Tk()
        root.title("modify ")
        modifyPtn = tk.Button(root, text="modify amp/phase", command=modify)
        modifyPtn.pack(pady=10)
        

    except ValueError:
        print("Invalid input. Please enter a number.")  




def coustem_IDFT(X):
    N = len(X)
    x = np.zeros(N, dtype=np.complex128)
    
    for n in range(N):
        x[n] = sum(X[k] * np.exp(1j * 2 * np.pi * k * n / N) for k in range(N))
    
    return x / N

def IDFT():
    global data1
    if data1 is None:
        print("Please load the signal .")
        return
    try:
        d = coustem_IDFT(data1)
        real_parts = [complex_number.real for complex_number in d]
        print(real_parts)

    except ValueError:
        print("Invalid input. Please enter a number.")  



# Create a main window
root = tk.Tk()
root.title("Signal Viewer")

# Create buttons to open files
open_button_1 = tk.Button(root, text="Open File 1", command=open_file_1)
open_button_1.pack(pady=10)

open_button_2 = tk.Button(root, text="Open File 2", command=open_file_2)
open_button_2.pack(pady=10)

open_button_IDFT = tk.Button(root, text="Open File IDFT", command=open_file_IDFT)
open_button_IDFT.pack(pady=10)

# Create buttons to perform operations
addition_button = tk.Button(root, text="Perform Addition", command=perform_addition)
addition_button.pack(pady=10)

subtraction_button = tk.Button(root, text="Perform Subtraction", command=perform_subtraction)
subtraction_button.pack(pady=10)

# Create buttons to perform multiplcation
multiplication_button = tk.Button(root, text="Perform Multiplication", command=perform_multiplication)
multiplication_button.pack(pady=10)

# Create buttons to perform Squaring
squaring_button = tk.Button(root, text="Perform Squaring", command=perform_squaring)
squaring_button.pack(pady=10)

# Create buttons to perform shifting
shifting_button = tk.Button(root, text="Perform Shifting", command=perform_shifting)
shifting_button.pack(pady=10)

# Create buttons to perform normalization
normalization_button = tk.Button(root, text="Perform Normalization", command=perform_normalization)
normalization_button.pack(pady=10)

# Create buttons to perform accumulation
accumulation_button = tk.Button(root, text="Perform Accumulation", command=perform_accumulation)
accumulation_button.pack(pady=10)

# Create buttons to plot signal
plot_button = tk.Button(root, text="Plot Signal", command=lambda: plot_signal(data1))
plot_button.pack(pady=10)

# Create buttons to perform quantization
plot_button = tk.Button(root, text="Perform quantization", command=perform_quantization)
plot_button.pack(pady=10)

# Create buttons to Frequency Domain
plot_button = tk.Button(root, text="DFT", command=DFT)
plot_button.pack(pady=10)

# Create buttons to Frequency Domain
plot_button = tk.Button(root, text="IDFT", command=IDFT)
plot_button.pack(pady=10)


# Start the main event loop
root.mainloop()