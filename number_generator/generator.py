from PIL import Image, ImageDraw, ImageFont, ImageFilter
import os

fontSize = 30
imgSize = (28,28)


def stretch_horizontal(image, ch_width):
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
    sys_path = 'C:\\Windows\\Fonts'
    font_paths = os.listdir(sys_path)
    font_paths = [s for s in font_paths if s.endswith('ttf')]
    outpath = "../datasets/generated_data"
    exclude_fonts = {
        "HoloLensMDL2Assets", 
        "Marlett", 
        "SegoeMDL2Assets", "Symbol", "SymbolPropBT",
        "Webdings", "Wingdings"
    }

    transforms = {'rotate': False, 'stretch': True, 'blur': True}

    for digit in range(10):
        print("generating data for digit {:d} ... ".format(digit), end='')
        num_generated = 0
        ch = str(digit)
        # get folder ready
        if not os.path.isdir(os.path.join(outpath, ch)):
            os.mkdir(os.path.join(outpath, ch))
        for i, font_path in enumerate(font_paths):
            font = ImageFont.truetype(os.path.join(sys_path, font_path), fontSize)
            font_name = font.font.family + "-" + font.font.style
            font_name = font_name.replace(" ", "")
            if (font.font.family.replace(" ", "") in exclude_fonts) or \
                font_name.find("UltraBold") != -1:
                continue        
            image = Image.new("L", imgSize, color=0)
            pixSize = font.font.getsize(ch)
            centre = (imgSize[0]/2 - pixSize[0][0]/2, imgSize[1]/2 - pixSize[0][1]/2 - pixSize[1][1])
            ImageDraw.Draw(image).text(centre, ch, fill=255, font=font)
            image_name = ch + "-" + font_name

            if transforms['rotate']:
                for r in range(-10, 11, 5):
                    if r == 0:
                        continue
                    image_r = image.rotate(r)
                    image_r.save(os.path.join(outpath, ch, image_name + "-" + str(r) + ".png"))
                    num_generated += 1
            if transforms['stretch']:
                image = stretch_horizontal(image, pixSize[0][0] + 4)
                image_name += '-stretch'
            if transforms['blur']:
                image = image.filter(ImageFilter.GaussianBlur(1))
                image_name += '-blur'
            num_generated += 1
            image.save(os.path.join(outpath, ch, image_name + ".png"))
        print(' saved {:d} images'.format(num_generated))

    print("done")
