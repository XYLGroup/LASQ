import os
import random

def generate_image_pairs(low_res_folder, high_res_folder, output_txt_file):
    low_res_images = os.listdir(low_res_folder)
    high_res_images = os.listdir(high_res_folder)

    if len(low_res_images) != len(high_res_images):
        raise ValueError("The number of images in the low and high resolution folders are not the same.")

    random.shuffle(low_res_images)
    random.shuffle(high_res_images)

    with open(output_txt_file, 'w') as f:
        for low_img, high_img in zip(low_res_images, high_res_images):
            low_img_path = os.path.join(low_res_folder, low_img)
            high_img_path = os.path.join(high_res_folder, high_img)
            f.write(f"{low_img_path} {high_img_path}\n")

    print(f"Successfully saved image pairs to {output_txt_file}")

low_res_folder = "pathA"
high_res_folder = "pathB"
output_txt_file = './output.txt'


generate_image_pairs(low_res_folder, high_res_folder, output_txt_file)