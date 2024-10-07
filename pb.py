from tkinter import ttk
import tkinter as tk
from tkinter.messagebox import showinfo
import time
import threading

root = tk.Tk()
root.geometry("960x480")
root.title("Progressbar learn")

started = False

def update_progress_label():
    return f"Progress: {pb['value']}%"
    
    
def progress():
    global started
    
    def _progress():
        while pb['value'] < 100:
            pb['value'] += 20
            value_label['text'] = update_progress_label()
            time.sleep(1)
        showinfo(message='The progress completed!')
        
        started = False
        
    if not started:
        started = True
        progress_task = threading.Thread(target=_progress)
        progress_task.start()
        
        
def stop():
    pb.stop()
    value_label['text'] = update_progress_label()


pb = ttk.Progressbar(
    root,
    orient='horizontal',
    mode='determinate',
    length=280
)

pb.grid(column=0, row=0, columnspan=2, padx=10, pady=20)

value_label = ttk.Label(root, text=update_progress_label())
value_label.grid(column=0, row=1, columnspan=2)

start_button = ttk.Button(
    root,
    text='Progress',
    command=progress
)

start_button.grid(column=0, row=2, padx=10, pady=10, sticky=tk.E)

stop_button = ttk.Button(
    root,
    text='Stop',
    command=stop
)

stop_button.grid(column=1, row=2, padx=10, pady=10, sticky=tk.W)

root.mainloop()