import pandas as pd
from colorthief import ColorThief
import cv2 as cv
import numpy as np
import matplotlib
# import colormath
from PIL import Image, ImageDraw
from tqdm import tqdm

from colormath.color_objects import sRGBColor, LabColor

from colormath.color_conversions import convert_color

from colormath.color_diff import delta_e_cie2000
import os

from itertools import product
from math import isqrt
import random
from PIL import Image

def create(path):
    """"
    opens and resizes the image.

    Returns:
    image_resize : resized image
    pathname : pathname
    """
    # pathname = "minion"
    # pathext = "jpg"
    # pathname = "minnie"
    # pathext = "jpg"
    pathname, _ = os.path.splitext(path)

    image = cv.imread(path)
    height, width, _ = image.shape

    # size = int(input("How many inches across?"))
    # size = 14*size
    size = 30
    factor = size/width


    height = int(height * factor)
    width = int(width * factor)

    image_resize =cv.resize(image, (width, height), interpolation=cv.INTER_NEAREST)

    return image_resize, pathname
def symbol_init():
    """
    initializing the symbols library to be overlaid. Returns list of filenames.
    """
    symbols=[]
    for imgname in os.listdir("symbols"):
        # print(imgname)
        symbols.append(r"symbols/"+imgname)
        # symbols.append(imgname.replace(".png", ""))
    return symbols

def create_grid(image, white):
    """create a grid over the image or over a white screen (for the white bool)
    
    Returns:
    image : image with grid over it
    scale : the scale of the larger image (to account for grid size)
    """

    image_rgb = image[:, :, ::-1] #flipping the cv2 bgr to rgb

    image = Image.fromarray(image_rgb)
    pixel_width = image.width
    pixel_height = image.height
    scale = 40
    if white:
        image = Image.new('RGBA', (pixel_width*scale, pixel_height*scale) , 'white')
    else:     
        image = image.resize((pixel_width*scale, pixel_height*scale), resample=Image.Resampling.NEAREST)
    # Draw grid
    draw = ImageDraw.Draw(image)
    y_start = 0
    y_end = image.height
    step_size = scale

    for x in range(0, image.width, step_size):
        line = ((x, y_start), (x, y_end))
        draw.line(line, fill='black', width=0)

    x_start = 0
    x_end = image.width

    for y in range(0, image.height, step_size):
        line = ((x_start, y), (x_end, y))
        draw.line(line, fill='black', width=0)

    del draw

    return image, scale

def patch_asscalar(a):
    """ patch for pixelate (colorthief)"""
    return a.item()
    
def pixelate(image, palette):
    """Pixelates the image into a set of colors
    
    Returns:
    pixelated image
    """
    

    # for each pixel, calculate its difference from each of the palette colors and pick the closest difference
    reshaped = image.reshape(-1, image.shape[-1])
    image_list = np.unique(reshaped, axis=0)

    unique=[]
    for pixel in tqdm(image_list):

            closest = 10000000
            pixel_rgb = sRGBColor(pixel[2], pixel[1], pixel[0])
            closecolor = (0, 0, 0)
            pixelcolor = convert_color(pixel_rgb, LabColor)
            for color in palette:
                color_rgb = sRGBColor(color[0], color[1], color[2])
                color_lab = convert_color(color_rgb, LabColor)
                closer = delta_e_cie2000(color_lab, pixelcolor)
                unique.append(closer)
                if closer<closest:
                    # print(closer, closest)
                    closest = closer
                    closecolor = color
            
            
            mask = np.all(image==(pixel[0], pixel[1], pixel[2]), axis=-1)
            
            # print(symb_dict[closecolor])
        
            # image[mask] = symb_dict[closecolor]
            image[mask] = (closecolor[2], closecolor[1], closecolor[0])

    #remove single pixels (where all neighbors are different?)
    return image

def create_pattern(image, image_matlike, scale, pathname, white, symbols, palette):
    """
    Creates the cross stitch pattern to use for projects
    
    Returns the dictionary of symbols used to create the pattern
    """
    image_pix = image.copy()
    width = image_pix.width
    height = image_pix.height
    pixel_width = image_matlike.shape[1] #this may be unnecessary
    pixel_height = image_matlike.shape[0]

    # n_images = int((width*height)/(scale**2))
    # image_pix.show()
    # Dummy sample list of images
    sample_list = []

    #replace symbol list with images
    #replace all colors with symbols

    image2= image_matlike.astype(str)

    palette = np.array(palette, dtype=str)

    symb = random.sample(symbols, len(palette))

    symb_dict={}

    for i in range(len(palette)):
        color = palette[i]
        symb_dict[str(color)]=symb[i]
    #create new np array and use mask on it instead
    image_names= np.zeros((image2.shape[0], image2.shape[1])).astype(str)

    for color in palette:
        color_reverse = [color[2], color[1], color[0]]


        mask = np.all(image2==color_reverse, axis=-1)
        # print(np.count_nonzero(mask))
        # image2[mask]=symb_dict[str(color)]
        image_names[mask]=symb_dict[str(color)]

    sample_list = image_names.flatten('F')
    # print(f"{len(sample_list)}")
    # Determine positions from given parameters
    x_pos = [(i * scale+ pixel_width//4)  for i in range(pixel_width)]
    y_pos = [(i * scale+ pixel_width//4)  for i in range(pixel_height)]
    # positions = list(product(y_pos, x_pos))
    positions = list(product(x_pos, y_pos))

    # Build and save final image
    a = image_pix.copy().convert('RGBA')

    # a.save('im1.png')
    for i in tqdm(range(len(positions))): #positions and sample list aren't the same size... need to figure that out

        symbol = Image.open(sample_list[i]).resize((width//int(scale*1.5), height//int(scale*1.5)), resample=Image.Resampling.NEAREST).convert('RGBA')
        a.paste(symbol, positions[i], mask=symbol)
    if white:
        saving = f"whitepattern/img{pathname}.png"
        a.save(saving)
    else: 
        saving = f"patterns/img{pathname}.png"
        a.save(saving)
    print(saving)
    return symb_dict

def create_key(symb_dict):
    #TODO: implement
    return 0

def full_fx(path):
    symbols = symbol_init()

    #split into pathname and path ext

    resized_image, pathname = create(path)

    setattr(np, "asscalar", patch_asscalar)

    img_ct = ColorThief(path)

    # num_colors = input("How many colors?") 
    # TODO: update to be input
    num_colors = 10

    palette = img_ct.get_palette(color_count=num_colors)
    # could create a code where if there's a color close to white don't add white else add white...
    palette.append((255, 255, 255))

    pixel_img = pixelate(resized_image, palette)
    # white = input("white or no? (y/n)")
    # white = True if 'y' else False
    white = False #TODO: update
    grid_img, scale = create_grid(pixel_img, white)

    symb_pattern = create_pattern(grid_img, resized_image, scale, pathname, white, symbols, palette)
    print(symb_pattern)

full_fx("minnie.jpg")