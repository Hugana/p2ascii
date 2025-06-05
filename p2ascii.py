import sys
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np


class P2Ascii:
    def __init__(self):
        self.ascii_list = [" ", ".", ":", "c", "o", "P", "B", "O", "?", "&", "#"]
        self.ascii_letters = cv2.imread("1x0 8x8 2.png", cv2.IMREAD_GRAYSCALE)

    def get_ascii_index_for_pixel(self, pixel: int) -> int:
        num_levels = len(self.ascii_list)
        step = 256 // num_levels
        index = min(pixel // step, num_levels - 1)
        return index

    def get_ascii_image_by_index(self, index: int):
        if index == 0:
            return np.zeros((8, 8), dtype=np.uint8)
        else:
            start = (index - 1) * 8
            end = index * 8
            return self.ascii_letters[0:8, start:end]

    def convert_image_to_asscii_simple(self, image_path):
        image = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)

        if image is None:
            print("Error: image could not be loaded.")
            return

        rows, cols = image.shape
        d_width = cols // 8
        d_height = rows // 8
        d_dim = (d_width, d_height)
        u_dim = (cols, rows)
        image = cv2.resize(image, d_dim, interpolation=cv2.INTER_AREA)
        image = cv2.resize(image, u_dim, interpolation=cv2.INTER_NEAREST)
        f_image = np.zeros([rows, cols], dtype=np.uint8)

        for i in range(d_height):
            for j in range(d_width):
                pixel = image[i * 8, j * 8]
                index = self.get_ascii_index_for_pixel(pixel)
                ascii_text_image = self.get_ascii_image_by_index(index)
                f_image[i * 8 : (i + 1) * 8, j * 8 : (j + 1) * 8] = ascii_text_image

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"image_{timestamp}.png"
        save_path = Path.cwd() / filename
        cv2.imwrite(str(save_path), f_image)

    def run(self):
        if len(sys.argv) < 2:
            print("Missing command. Try 'p2ascii help'")
            return

        command = sys.argv[1]
        print(command)
        match command:
            case "sc2image":
                if len(sys.argv) < 3:
                    print("No image path specified")
                    return
                print(sys.argv[2])
                self.convert_image_to_asscii_simple(sys.argv[2])


if __name__ == "__main__":
    program = P2Ascii()
    program.run()
