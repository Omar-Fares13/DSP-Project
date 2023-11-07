import tkinter as tk
from tkinter import ttk
import numpy as np
import matplotlib.pyplot as plt

def generate_signal():
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

root = tk.Tk()
root.title('Sinusoidal Signal Generator')

# Signal Type
signal_type_label = ttk.Label(root, text="Signal Type:")
signal_type_label.grid(row=0, column=0, sticky='w')
signal_type = ttk.Combobox(root, values=['Sine', 'Cosine'], state='readonly')
signal_type.grid(row=0, column=1, pady=5)

# Amplitude
amplitude_label = ttk.Label(root, text="Amplitude (A):")
amplitude_label.grid(row=1, column=0, sticky='w')
amplitude_entry = ttk.Entry(root)
amplitude_entry.grid(row=1, column=1, pady=5)

# Phase Shift
phase_shift_label = ttk.Label(root, text="Phase Shift (θ) in degrees:")
phase_shift_label.grid(row=2, column=0, sticky='w')
phase_shift_entry = ttk.Entry(root)
phase_shift_entry.grid(row=2, column=1, pady=5)

# Analog Frequency
analog_frequency_label = ttk.Label(root, text="Analog Frequency (Hz):")
analog_frequency_label.grid(row=3, column=0, sticky='w')
analog_frequency_entry = ttk.Entry(root)
analog_frequency_entry.grid(row=3, column=1, pady=5)

# Sampling Frequency
sampling_frequency_label = ttk.Label(root, text="Sampling Frequency (Hz):")
sampling_frequency_label.grid(row=4, column=0, sticky='w')
sampling_frequency_entry = ttk.Entry(root)
sampling_frequency_entry.grid(row=4, column=1, pady=5)

# Generate Button
generate_button = ttk.Button(root, text="Generate Signal", command=generate_signal)
generate_button.grid(row=5, column=0, columnspan=2, pady=10)

root.mainloop()
