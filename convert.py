import cv2
import PIL

import math

g_str = "$@B%8&WM#*oahkbdpqwmZO0QLCJUYXzcvunxrjft/\|()1{}[]?-_ +~<>i!lI;:,\"^`'."
g_str_len = len(g_str)
g_interval = g_str_len / 256

# val: [0, 255]
def lightness_to_char(val):
    index = math.floor(val * g_interval)
    return g_str[index]
    
for i in range(256):
    print("{}: {}".format(i, lightness_to_char(i)))