from PIL import Image, ImageOps
import os

def process_images(input_dir, output_dir):
    subdirs = os.listdir(input_dir)
    for subdir in subdirs:
        ch = str(int(subdir[len(subdir)-2:len(subdir)]) - 1)
        print(f"Reading images for {ch} in {subdir}")
        if not os.path.isdir(os.path.join(output_dir, ch)):
            os.mkdir(os.path.join(output_dir, ch))
        for path_in in os.listdir(os.path.join(input_dir, subdir)):
            image = Image.open(os.path.join(input_dir, subdir, path_in))
            image = ImageOps.invert(image)
            image = image.resize((28, 28))
            image.save(os.path.join(output_dir, ch, path_in))
    print("done")



if __name__ == '__main__':
    input_dir = "../datasets/74k_numbers"
    output_dir = "../datasets/74k_numbers_28x28"

    process_images(input_dir, output_dir)
