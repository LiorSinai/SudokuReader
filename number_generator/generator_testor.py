from PIL import Image, ImageDraw, ImageFont, ImageFilter
import os
import numpy as np
import cv2 as cv

fontSize = 30
imgSize = (28,28)


def stretch_image(image, ch_width):
    # first crop middle strip
    width, height = image.size
    centre = (width//2, height//2)
    left = max(0, centre[0] - ch_width//2)
    right = min(width, centre[0] + ch_width//2)
    image = image.crop((left, 0, right, height))
    # then resize back to original size
    image = image.resize((width, height)) 
    return image


if __name__ == '__main__':
    # extract system font
    outpath = ""
    sys_path = 'C:\\Windows\\Fonts'
    font_paths = os.listdir(sys_path)
    font_paths = [s for s in font_paths if s.endswith('ttf')]

    font = ImageFont.truetype(os.path.join(sys_path, font_paths[3]), fontSize)
    font_name = font.font.family + "-" + font.font.style
    font_name = font_name.replace(" ", "")
    print("using font " + font_name)
    for digit in range(10):
        print("generating data for digit {:d}".format(digit))
        ch = str(digit)
        image = Image.new("L", imgSize, color=0)
        text_size = font.font.getsize(ch)
        centre = (imgSize[0]/2 - text_size[0][0]/2, imgSize[1]/2 - text_size[0][1]/2 - text_size[1][1])
        ImageDraw.Draw(image).text(centre, ch, fill=255, font=font)
        #image = stretch_image(image, pixSize[0][0])
        image = image.filter(ImageFilter.GaussianBlur(1))
        image.save(os.path.join(outpath, ch + "-tmp.png"))
    print("done")
