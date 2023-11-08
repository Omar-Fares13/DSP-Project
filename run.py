import tkinter as tk
import subprocess

def run_program_1():
    # Replace 'program_1.py' with the actual name of the program you want to run for option 1
    subprocess.run(['python', 'signal1.py'])

def run_program_2():
    # Replace 'program_2.py' with the actual name of the program you want to run for option 2
    subprocess.run(['python', 'trig_signal.py'])

# Create the main window
root = tk.Tk()
root.title("Program Selector")

# Create and configure the buttons
button1 = tk.Button(root, text="Read data from file", command=run_program_1)
button1.pack(pady=10)

button2 = tk.Button(root, text="generate trig functions", command=run_program_2)
button2.pack(pady=10)

# Start the main event loop
root.mainloop()