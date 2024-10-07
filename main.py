from pathlib import Path
import math
from functools import reduce
import time
import threading

from PIL import ImageTk, Image, ImageFont, ImageDraw
import cv2 as cv
import numpy as np

from tkinter import ttk
import tkinter as tk
from tkinter.messagebox import showinfo
from tkinter import filedialog, simpledialog


# configurations
grayscale_string = "$@B%8&WM#*oahkbdpqwmZO0QLCJUYXzcvunxrjft/\|()1{}[]?-_ +~<>i!lI;:,\"^`'."
video_file = "bad_apple.mp4"
image_file = "images/neptunia.jpg"
ratio = 0.5
font_file = "./fonts/CrimsonText-Bold.ttf"
output_video_name = "my_video.mp4"
output_image_name = "my_image.jpg"
font_point = 16
default_size = (512, 512)
background_color = (200, 200, 200)
font_color = (0, 0, 0)
target_size = (1960, 1080)
output_file_name = "output.mp4"
windows_geometry = "720x720"
windows_title = "Grayscale Characters Video Converter"
#isForceMono = True

g_str = "$@B%8&WM#*oahkbdpqwmZO0QLCJUYXzcvunxrjft/\|()1{}[]?-_ +~<>i!lI;:,\"^`'."
g_str_len = len(g_str)
g_interval = g_str_len / 256


font = ImageFont.truetype(font_file, font_point)

# image to be processed
img_grayscale = cv.imread(image_file, 0)
pil_img = Image.fromarray(img_grayscale)

# temporary canvas
image = Image.new("RGBA", pil_img.size, background_color)

draw = ImageDraw.Draw(image)


gs_length = font.getlength(grayscale_string)
print(f"length = {gs_length}")

max_char_width = reduce(lambda value,element: font.getlength(element) if value < font.getlength(element) else value, grayscale_string, 0)
print(max_char_width)
bound = font.getbbox(grayscale_string)
max_char_height = bound[3] - bound[1]


#chars_per_line = math.floor(image.size[0]/max_char_width)

#print(chars_per_line)

#lines = math.floor(len(grayscale_string) / chars_per_line)

def update_font_info():
    global max_char_width, max_char_height, font, proc_win_size
    
    font = ImageFont.truetype(font_file, font_point)
    max_char_width = reduce(lambda value,element: font.getlength(element) if value < font.getlength(element) else value, grayscale_string, 0)
    print(max_char_width)
    bound = font.getbbox(grayscale_string)
    max_char_height = bound[3] - bound[1]
    # kernel size
    proc_win_size = (max_char_width, max_char_height)

    

# image is a PIL.Image object with mode=L, bbox is (left, top, right, bottom) bounding box
def ave_grayness(image, bbox):
    cols = int(bbox[2] - bbox[0])
    rows = int(bbox[3] - bbox[1])
    value = 0
    for x in range(cols):
        for y in range(rows):
            pixel = image.getpixel((x+bbox[0],y+bbox[1]))
            value += pixel
    value = value / (cols*rows)
    return value
    
# val: [0, 255]
def lightness_to_char(val):
    index = math.floor(val * g_interval)
    return g_str[index]


# processing kernel size
proc_win_size = (max_char_width, max_char_height)
# processing grid size
proc_grid_size = (math.floor(image.size[0]/max_char_width), math.floor(image.size[1]/max_char_height))


#total_frame = cap.get(cv.CAP_PROP_FRAME_COUNT)
#print(f"total frame={total_frame}")

"""
# processing image
for x in range(proc_grid_size[0]):
    for y in range(proc_grid_size[1]):
        box = (x*proc_win_size[0], y*proc_win_size[1], (x+1)*proc_win_size[0], (y+1)*proc_win_size[1])
        gray = ave_grayness(pil_img, box)
        draw.text((box[0], box[1]), lightness_to_char(gray), font=font, fill=font_color)
"""
  


def convert_image(
    image,
    font,
    font_color,
    background_color,
    win_size,
    grid_size,
    isForceMono=True):
    
    buffer_img = Image.new("RGBA", image.size, background_color)
    draw = ImageDraw.Draw(buffer_img)
    for x in range(grid_size[0]):
        for y in range(grid_size[1]):
            box = (x*win_size[0], y*win_size[1], (x+1)*win_size[0], (y+1)*win_size[1])
            gray = ave_grayness(image, box)
            draw.text((box[0], box[1]), lightness_to_char(gray), font=font, fill=font_color)
    return buffer_img
    

"""
image2 = convert_image(pil_img, font, font_color, background_color, proc_win_size, proc_grid_size)

image2.show()
"""

# vidoe_file = path to video file
# target_size = (width, height) of target video size
# 



current_progress = 0

def convert_video(stop_event):
    #video_file
    #target_size,
    #font_file,
    #font_color,
    #font_size,
    #background_color,
    #isForceMono=True):
    #):
    try:
        cap = cv.VideoCapture(video_file)
        ret, frame = cap.read()
        gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
        gray_img = Image.fromarray(gray)
        proc_grid_size = (math.floor(target_size[0]/max_char_width), math.floor(target_size[1]/max_char_height))
        fps = cap.get(cv.CAP_PROP_FPS)
        total_frames = cap.get(cv.CAP_PROP_FRAME_COUNT)
        outfile = cv.VideoWriter(output_file_name,  
                                 cv.VideoWriter_fourcc(*'mp4v'), 
                                 fps, gray_img.size)

        processed_frames = 0

        while True:
            ret, frame = cap.read()
            if not ret:
                break;
            
            gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
            gray = Image.fromarray(gray)
            gray = gray.resize(target_size)
            gray = convert_image(gray, font, font_color, background_color, proc_win_size, proc_grid_size)
            gray = np.asarray(gray)
            cv.imshow('Current Output (press q to quit)', gray)
            
            processed_frames += 1
            global current_progress
            
            current_progress = processed_frames / total_frames
            #print(f"{current_progress}, {processed_frames}, {total_frames}")
            update_progress_label()
            
            rgb = cv.cvtColor(gray, cv.COLOR_RGB2BGR)
            outfile.write(rgb)
            if cv.waitKey(1) == ord('q') or stop_event.is_set():
                break;
    except:
        cap.release()
        outfile.release()
        cv.destroyAllWindows()
    finally:
        current_progress = 1
        cap.release()
        outfile.release()
        cv.destroyAllWindows()
        
        
def preview():
    cap = cv.VideoCapture(video_file)
    ret, frame = cap.read()
    gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    gray_img = Image.fromarray(gray)
    gray_img = gray_img.resize(target_size)
    proc_grid_size = (math.floor(target_size[0]/max_char_width), math.floor(target_size[1]/max_char_height))
    
    # preview 
    gray_img = convert_image(gray_img, font, font_color, background_color, proc_win_size, proc_grid_size)
    cv.imshow('Output Preview', np.array(gray_img))
    cap.release()




# create UI


root = tk.Tk()
root.geometry(windows_geometry)
root.title(windows_title)



pb = ttk.Progressbar(
    root,
    orient='horizontal',
    mode='determinate',
    length=280
)

pb.grid(column=0, row=0, columnspan=2, padx=10, pady=20)

value_label = ttk.Label(root, text="Progress: 0%")
value_label.grid(column=0, row=1, columnspan=2)



def update_progress_label():
    # print(current_progress)
    pb['value'] = current_progress*100
    value_label['text'] = "Progress: {:.2f}%".format(current_progress*100)
    # return f"Progress: {current_progress*100}%"
    
task = None
stop_event = threading.Event()
    
def progress():
    stop_event.clear()
    global task
    if task == None:
        task = threading.Thread(target=lambda: convert_video(stop_event))
        task.start()


def stop():
    pb.stop()
    value_label['text'] = "progress: 0%"
    global task
    if task != None:
        stop_event.set()
        task = None
        

start_button = ttk.Button(
    root,
    text='Convert',
    command=progress
)

start_button.grid(column=0, row=2, padx=10, pady=10, sticky=tk.E)

stop_button = ttk.Button(
    root,
    text='Stop',
    command=stop
)

stop_button.grid(column=1, row=2, padx=10, pady=10, sticky=tk.W)

preview_button = ttk.Button(
    root,
    text='Preview',
    command=preview
)

preview_button.grid(column=2, row=2, padx=10, pady=10, sticky=tk.W)


       
       
# video input
ttk.Label(root, text="Input file: ").grid(column=0, row=3, padx=10, pady=10, sticky=tk.W)
input_file_view = ttk.Label(root, text="<Input video file name>")
input_file_view.grid(column=0, row=4, padx=10, pady=5, sticky=tk.W)


def open_input_file():
    name = filedialog.askopenfilename(filetypes=[("Videos", "*.mp4")])
    if name != "":
        global video_file
        video_file = name
        input_file_view['text'] = Path(video_file).name
        

input_file_btn = ttk.Button(root, text="Open", command=open_input_file)
input_file_btn.grid(column=0, row=5, padx=10, pady=5, sticky=tk.W)


# output
ttk.Label(root, text="Output file: ").grid(column=0, row=6, padx=10, pady=10, sticky=tk.W)
output_file_view = ttk.Label(root, text="<output video file name>")
output_file_view.grid(column=0, row=7, padx=10, pady=5, sticky=tk.W)


def open_input_file():
    name = filedialog.asksaveasfilename(filetypes=[("Videos", "*.mp4")])
    if name != "":
        global output_file_name
        output_file_name = name
        output_file_view['text'] = Path(output_file_name).name
        

output_file_btn = ttk.Button(root, text="Save to", command=open_input_file)
output_file_btn.grid(column=0, row=8, padx=10, pady=5, sticky=tk.W)


# target size
target_size_width, target_size_height = tk.IntVar(), tk.IntVar()
target_size_width.set(1920)
target_size_height.set(1080)

ttk.Label(root, text="Output size:").grid(column=0, row=9, padx=10, pady=10, sticky=tk.W)
output_canvas_size = ttk.Label(root, text="<size>")
output_canvas_size.grid(column=0, row=10, padx=10, pady=10, sticky=tk.W)

ttk.Label(root, text="Width: ").grid(column=0, row=11, padx=10, pady=0, sticky=tk.W)
target_size_width_entry = tk.Entry(root, textvariable=target_size_width)
target_size_width_entry.grid(column=1, row=11, padx=0, pady=0, sticky=tk.W)
target_size_height_entry = tk.Entry(root, textvariable=target_size_height)
target_size_height_entry.grid(column=3, row=11, padx=0, pady=0, sticky=tk.W)
ttk.Label(root, text="Height: ").grid(column=2, row=11, padx=0, pady=0, sticky=tk.W)

def update_output_size():
    global target_size
    target_size = (target_size_width.get(), target_size_height.get())
    output_canvas_size['text'] = f"{target_size[0]} x {target_size[1]}"

output_canvas_size_button = ttk.Button(root, text="Update", command=update_output_size)
output_canvas_size_button.grid(column=0, row=12, padx=10, pady=0, sticky=tk.W)

# font
ttk.Label(root, text="Font:").grid(column=0, row=13, padx=10, pady=0, sticky=tk.W)
font_label = ttk.Label(root, text="<font>")
font_label.grid(column=0, row=14, padx=10, pady=5, sticky=tk.W)

def update_font():
    font_name = filedialog.askopenfilename(filetypes=[("Font", "*.ttf")])
    if font_name != "":
        global font_file
        font_file = font_name
        update_font_info()
        font_label['text'] = Path(font_file).stem
        
font_button = ttk.Button(root, text="Update", command=update_font)
font_button.grid(column=0, row=15, padx=10, pady=5, sticky=tk.W)

# font size
ttk.Label(root, text="Font point:").grid(column=1, row=13, padx=0,pady=0, sticky=tk.W)
font_size_label = ttk.Label(root, text="<size>")
font_size_label.grid(column=2, row=13, padx=10, pady=5, sticky=tk.W)

font_size_var = tk.IntVar()
font_size_var.set(font_point)

def update_font_size():
    global font_point
    font_point = font_size_var.get()
    update_font_info()
    font_size_label['text'] = font_point

tk.Entry(root, textvariable=font_size_var).grid(column=1, row=14, padx=0, pady=0, sticky=tk.W)

font_size_button = ttk.Button(root, text="Update", command=update_font_size)
font_size_button.grid(column=1, row=15, padx=10, pady=5, sticky=tk.W)

# color
ttk.Label(root, text="Font color:").grid(column=0, row=16, padx=10, pady=0, sticky=tk.W)
font_color_label = ttk.Label(root, text="(R,G,B)")
font_color_label.grid(column=1, row=16, padx=0, pady=0, sticky=tk.W)

font_R, font_G, font_B = tk.IntVar(), tk.IntVar(), tk.IntVar()
tk.Entry(root, textvariable=font_R).grid(column=0, row=17, padx=10, pady=0, sticky=tk.W)
tk.Entry(root, textvariable=font_G).grid(column=1, row=17, padx=0, pady=0, sticky=tk.W)
tk.Entry(root, textvariable=font_B).grid(column=2, row=17, padx=0, pady=0, sticky=tk.W)

def update_font_color():
    r, g, b = font_R.get(), font_G.get(), font_B.get()
    r, g, b = min(max(r,0),255), min(max(g,0),255), min(max(b,0),255)
    font_R.set(r)
    font_G.set(g)
    font_B.set(b)
    global font_color
    font_color = (r, g, b)
    font_color_label['text'] = f"({r}, {g}, {b})"

font_color_button = ttk.Button(root, text="Update", command=update_font_color)
font_color_button.grid(column=0, row=18, padx=10, pady=0, sticky=tk.W)


# background color
ttk.Label(root, text="Background color:").grid(column=0, row=19, padx=10, pady=0, sticky=tk.W)
bg_color_label = ttk.Label(root, text="(R,G,B)")
bg_color_label.grid(column=1, row=19, padx=0, pady=0, sticky=tk.W)

bg_R, bg_G, bg_B = tk.IntVar(), tk.IntVar(), tk.IntVar()
tk.Entry(root, textvariable=bg_R).grid(column=0, row=20, padx=10, pady=0, sticky=tk.W)
tk.Entry(root, textvariable=bg_G).grid(column=1, row=20, padx=0, pady=0, sticky=tk.W)
tk.Entry(root, textvariable=bg_B).grid(column=2, row=20, padx=0, pady=0, sticky=tk.W)

def update_background_color():
    r, g, b = bg_R.get(), bg_G.get(), bg_B.get()
    r, g, b = min(max(r,0),255), min(max(g,0),255), min(max(b,0),255)
    bg_R.set(r)
    bg_G.set(g)
    bg_B.set(b)
    global background_color
    background_color = (r, g, b)
    bg_color_label['text'] = f"({r}, {g}, {b})"

bg_color_button = ttk.Button(root, text="Update", command=update_background_color)
bg_color_button.grid(column=0, row=21, padx=10, pady=0, sticky=tk.W)



root.mainloop()