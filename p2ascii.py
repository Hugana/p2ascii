import sys
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np


class P2Ascii:
    def __init__(self):
        self.ascii_list = [" ", ".", ":", "c", "o", "P", "B", "O", "?", "&", "#"]
        self.ascii_orientation_list = [" ", "|", "—", "/", "\\"]
        self.ascii_letters_img = cv2.imread("1x0 8x8 2.png", cv2.IMREAD_GRAYSCALE)
        self.ascii_orientation_img = cv2.imread("edgesASCII.png", cv2.IMREAD_GRAYSCALE)

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
            return self.ascii_letters_img[0:8, start:end]

    def get_ascii_orientation_image_by_index(self, index: int):
        start = index * 8
        end = (index + 1) * 8
        return self.ascii_orientation_img[0:8, start:end]

    def get_ascii_orientation_index_for_angle(self, angle: float) -> int:
        if angle <= 45:
            return 1
        elif angle <= 90:
            return 2
        elif angle <= 135:
            return 3
        elif angle <= 180:
            return 4
        else:
            return 0

    def convert_image_to_ascii_image_simple(self, image_path):
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

    def convert_image_to_ascii_text_simple(self, image_path):
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
        result = "\n"
        for i in range(d_height):
            for j in range(d_width):
                pixel = image[i * 8, j * 8]
                index = self.get_ascii_index_for_pixel(pixel)
                result += self.ascii_list[index]
            result += "\n"

        return result

    def convert_image_to_ascii_image_complex(self, image_path):
        image = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)

        if image is None:
            print("Error: image could not be loaded.")
            return

        gX = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=3)
        gY = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=3)

        magnitude = np.sqrt((gX**2) + (gY**2))
        orientation = (((np.arctan2(gY, gX) / np.pi) * 0.5 + 0.5) * 360) % 180

        rows, cols = image.shape
        d_width = cols // 8
        d_height = rows // 8
        d_dim = (d_width, d_height)
        u_dim = (cols, rows)

        downscaled_img = cv2.resize(image, d_dim, interpolation=cv2.INTER_AREA)
        downscaled_img = cv2.resize(
            downscaled_img, u_dim, interpolation=cv2.INTER_NEAREST
        )

        orientation_downscaled = cv2.resize(
            orientation, d_dim, interpolation=cv2.INTER_AREA
        )
        orientation_downscaled = cv2.resize(
            orientation_downscaled, u_dim, interpolation=cv2.INTER_NEAREST
        )

        magnitude_downscaled = cv2.resize(
            magnitude, d_dim, interpolation=cv2.INTER_AREA
        )
        magnitude_downscaled = cv2.resize(
            magnitude, u_dim, interpolation=cv2.INTER_NEAREST
        )

        nonzero_magnitudes = magnitude_downscaled[magnitude_downscaled > 0]
        threshold = np.percentile(nonzero_magnitudes, 90)

        f_image = np.zeros([rows, cols], dtype=np.uint)

        for i in range(d_height):
            for j in range(d_width):
                pixel = downscaled_img[i * 8, j * 8]
                index = self.get_ascii_index_for_pixel(pixel)
                ascii_text_image = self.get_ascii_image_by_index(index)
                f_image[i * 8 : (i + 1) * 8, j * 8 : (j + 1) * 8] = ascii_text_image

        for i in range(d_height):
            for j in range(d_width):
                block_mag = magnitude[i * 8 : (i + 1) * 8, j * 8 : (j + 1) * 8]
                if np.mean(block_mag) < 30:
                    continue
                angle = orientation_downscaled[i * 8, j * 8]
                index = self.get_ascii_orientation_index_for_angle(angle)
                ascii_text_image = self.get_ascii_orientation_image_by_index(index)
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

        if command == "help":
            pass

        if len(sys.argv) < 3:
            print("No image specified")
            return

        match command:
            case "sc2image":
                self.convert_image_to_ascii_image_simple(sys.argv[2])
            case "sc2text":
                print(self.convert_image_to_ascii_text_simple(sys.argv[2]))
            case "cc2image":
                self.convert_image_to_ascii_image_complex(sys.argv[2])


if __name__ == "__main__":
    program = P2Ascii()
    program.run()
