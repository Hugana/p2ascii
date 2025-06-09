import sys
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np


class P2Ascii:
    def __init__(self):
        self.ascii_list = [" ", ".", ":", "c", "o", "P", "B", "O", "?", "&", "#"]
        self.ascii_orientation_list = [" ", "|", "—", "/", "\\"]
        self.ascii_letters_img = cv2.imread("Images/1x0 8x8 2.png", cv2.IMREAD_GRAYSCALE)
        self.ascii_orientation_img = cv2.imread("Images/edgesASCII.png", cv2.IMREAD_GRAYSCALE)
        self.ascii_orientation_img_color = cv2.imread("Images/edgesASCII.png", cv2.IMREAD_COLOR)
        self.ascii_letters_img_color = cv2.imread("Images/1x0 8x8 2.png", cv2.IMREAD_COLOR)

    def get_ascii_index_for_pixel(self, pixel: int) -> int:
        num_levels = len(self.ascii_list)
        step = 256 // num_levels
        index = min(pixel // step, num_levels - 1)
        return index

    def get_ascii_image_by_index(self, index: int, color: bool):
        if index == 0:
            if color:
                return self.ascii_letters_img_color[0:8, 0:8]
            return self.ascii_letters_img[0:8, 0:8]
        else:
            start = (index - 1) * 8
            end = index * 8
            if color:
                return self.ascii_letters_img_color[0:8, start:end]
            return self.ascii_letters_img[0:8, start:end]

    def get_ascii_orientation_image_by_index(self, index: int, color: bool):
        start = index * 8
        end = (index + 1) * 8
        if color:
            return self.ascii_orientation_img_color[0:8, start:end]
        return self.ascii_orientation_img[0:8, start:end]

    def get_ascii_orientation_index_for_angle(self, angle: float) -> int:
        if angle <= 45:
            return 2
        elif angle <= 90:
            return 3
        elif angle <= 135:
            return 4
        elif angle <= 180:
            return 1
        else:
            return 0

    def set_color_of_ascii_image(self, ascii_image, color):
        rows, cols, _ = ascii_image.shape
        for i in range(rows):
            for j in range(cols):
                if np.array_equal(ascii_image[i, j], [255, 255, 255]):
                    ascii_image[i, j] = color

        return ascii_image

    def prepare_images(self, image_path, with_color=True):
        gray = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
        color = cv2.imread(str(image_path), cv2.IMREAD_COLOR) if with_color else None

        if gray is None:
            raise ValueError("Image could not be loaded.")

        rows, cols = gray.shape
        d_width, d_height = cols // 8, rows // 8
        d_dim, u_dim = (d_width, d_height), (cols, rows)

        resized_gray = cv2.resize(gray, d_dim, interpolation=cv2.INTER_AREA)
        resized_gray = cv2.resize(resized_gray, u_dim, interpolation=cv2.INTER_NEAREST)

        if color is not None:
            color = cv2.resize(color, d_dim, interpolation=cv2.INTER_AREA)
            color = cv2.resize(color, u_dim, interpolation=cv2.INTER_NEAREST)

        return gray, resized_gray, color, d_width, d_height, rows, cols

    def compute_gradients(self, gray, rows, cols):

        gX = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
        gY = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
        magnitude = np.sqrt(gX**2 + gY**2)
        orientation = (((np.arctan2(gY, gX) / np.pi) * 0.5 + 0.5) * 360) % 180

        d_width, d_height = cols // 8, rows // 8
        d_dim, u_dim = (d_width, d_height), (cols, rows)

        magnitude_downscaled = cv2.resize(magnitude, d_dim, interpolation=cv2.INTER_AREA)
        orientation_downscaled = cv2.resize(orientation, d_dim, interpolation=cv2.INTER_AREA)

        magnitude_upscaled = cv2.resize(magnitude_downscaled, u_dim, interpolation=cv2.INTER_NEAREST)
        orientation_upscaled = cv2.resize(orientation_downscaled, u_dim, interpolation=cv2.INTER_NEAREST)

        return magnitude_upscaled, orientation_upscaled, magnitude


    def convert_image_to_ascii_image_simple(self, image_path):

        gray, image, color, d_width, d_height, rows, cols = self.prepare_images(image_path,False)

        f_image = np.zeros([rows, cols], dtype=np.uint8)

        for i in range(d_height):
            for j in range(d_width):
                pixel = gray[i * 8, j * 8]
                index = self.get_ascii_index_for_pixel(pixel)
                ascii_text_image = self.get_ascii_image_by_index(index, False)
                f_image[i * 8 : (i + 1) * 8, j * 8 : (j + 1) * 8] = ascii_text_image

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"image_{timestamp}.png"
        save_path = Path.cwd() / filename
        cv2.imwrite(str(save_path), f_image.astype(np.uint8))

    def convert_image_to_ascii_text_simple(self, image_path):
        gray, image, color, d_width, d_height, rows, cols = self.prepare_images(image_path,False)

        result = "\n"
        for i in range(d_height):
            for j in range(d_width):
                pixel = gray[i * 8, j * 8]
                index = self.get_ascii_index_for_pixel(pixel)
                result += self.ascii_list[index] * 2
            result += "\n"

        return result

    def convert_image_to_ascii_image_simple_color(self, image_path):

        image_gray, image, image_color, d_width, d_height, rows, cols = self.prepare_images(image_path,True)

        f_image = np.zeros([rows, cols, 3], dtype=np.uint8)
        ascii_text_image_color = 0
        for i in range(d_height):
            for j in range(d_width):
                pixel_color = image_color[i * 8, j * 8]
                pixel_gray = image_gray[i * 8, j * 8]
                index_gray = self.get_ascii_index_for_pixel(pixel_gray)
                ascii_text_image = self.get_ascii_image_by_index(index_gray, True).copy()
                ascii_text_image_color = self.set_color_of_ascii_image(ascii_text_image, pixel_color)
                f_image[i * 8 : (i + 1) * 8, j * 8 : (j + 1) * 8] = (ascii_text_image_color)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"image_{timestamp}.png"
        save_path = Path.cwd() / filename
        cv2.imwrite(str(save_path), f_image.astype(np.uint8))

    def convert_image_to_ascii_text_simple_color(self, image_path):
        image_gray, image, color, d_width, d_height, rows, cols = self.prepare_images(image_path,True)

        result = "\n"

        for i in range(d_height):
            for j in range(d_width):
                pixel_color = color[i * 8, j * 8]
                pixel_gray = image_gray[i * 8, j * 8]
                index_gray = self.get_ascii_index_for_pixel(pixel_gray)
                b = pixel_color[0]
                g = pixel_color[1]
                r = pixel_color[2]
                result += (
                    "\033[38;2;"
                    + str(r)
                    + ";"
                    + str(g)
                    + ";"
                    + str(b)
                    + ";m"
                    + self.ascii_list[index_gray] * 2
                    + "\033[0m"
                )
            result += "\n"

        return result

    def convert_image_to_ascii_text_complex(self, image_path, mag_threshold):
        gray, image, color, d_width, d_height, rows, cols = self.prepare_images(image_path,False)

        magnitude_downscaled, orientation_downscaled, magnitude = self.compute_gradients(gray, rows, cols)

        nonzero_magnitudes = magnitude_downscaled[magnitude_downscaled > 0]
        threshold = np.percentile(nonzero_magnitudes, 90)

        value = 30
        if str(mag_threshold).lower() == "auto":
            value = threshold
        else:
            try:
                int_value = int(mag_threshold)
                if 0 <= int_value <= 255:
                    value = int_value
                else:
                    print("Error: magnitude must be between 0 and 255")
                    return
            except ValueError:
                print("Error: invalid magnitude parameter (must be 'auto' or an integer between 0 and 255)")
                return

        result = "\n"

        for i in range(d_height):
            for j in range(d_width):
                block_mag = magnitude[i * 8 : (i + 1) * 8, j * 8 : (j + 1) * 8]

                if np.mean(block_mag) < value:
                    pixel = gray[i * 8, j * 8]
                    index = self.get_ascii_index_for_pixel(pixel)
                    result += self.ascii_list[index] * 2
                    continue

                angle = orientation_downscaled[i * 8, j * 8]
                index = self.get_ascii_orientation_index_for_angle(angle)
                result += self.ascii_orientation_list[index] * 2
            result += "\n"

        return result

    def convert_image_to_ascii_image_complex(self, image_path, mag_threshold):
        gray, image, color, d_width, d_height, rows, cols = self.prepare_images(image_path,False)

        magnitude_downscaled, orientation_downscaled, magnitude = self.compute_gradients(gray, rows, cols)

        nonzero_magnitudes = magnitude_downscaled[magnitude_downscaled > 0]
        threshold = np.percentile(nonzero_magnitudes, 90)

        f_image = np.zeros([rows, cols], dtype=np.uint8)

        value = 30
        if str(mag_threshold).lower() == "auto":
            value = threshold
        else:
            try:
                int_value = int(mag_threshold)
                if 0 <= int_value <= 255:
                    value = int_value
                else:
                    print("Error: magnitude must be between 0 and 255")
                    return
            except ValueError:
                print("Error: invalid magnitude parameter (must be 'auto' or an integer between 0 and 255)")
                return

        for i in range(d_height):
            for j in range(d_width):
                block_mag = magnitude[i * 8 : (i + 1) * 8, j * 8 : (j + 1) * 8]

                if np.mean(block_mag) < value:
                    pixel = gray[i * 8, j * 8]
                    index = self.get_ascii_index_for_pixel(pixel)
                    ascii_text_image = self.get_ascii_image_by_index(index, False)
                    f_image[i * 8 : (i + 1) * 8, j * 8 : (j + 1) * 8] = ascii_text_image
                    continue

                angle = orientation_downscaled[i * 8, j * 8]
                index = self.get_ascii_orientation_index_for_angle(angle)
                ascii_text_image = self.get_ascii_orientation_image_by_index(index, False)
                f_image[i * 8 : (i + 1) * 8, j * 8 : (j + 1) * 8] = ascii_text_image

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"image_{timestamp}.png"
        save_path = Path.cwd() / filename
        cv2.imwrite(str(save_path), f_image.astype(np.uint8))

    def convert_image_to_ascii_image_complex_color(self, image_path, mag_threshold):
        gray, image, color, d_width, d_height, rows, cols = self.prepare_images(image_path,True)

        magnitude_downscaled, orientation_downscaled, magnitude = self.compute_gradients(gray, rows, cols)

        nonzero_magnitudes = magnitude_downscaled[magnitude_downscaled > 0]
        threshold = np.percentile(nonzero_magnitudes, 90)

        f_image = np.zeros([rows, cols, 3], dtype=np.uint8)

        value = 30
        if str(mag_threshold).lower() == "auto":
            value = threshold
        else:
            try:
                int_value = int(mag_threshold)
                if 0 <= int_value <= 255:
                    value = int_value
                else:
                    print("Error: magnitude must be between 0 and 255")
                    return
            except ValueError:
                print("Error: invalid magnitude parameter (must be 'auto' or an integer between 0 and 255)")
                return

        for i in range(d_height):
            for j in range(d_width):
                block_mag = magnitude[i * 8 : (i + 1) * 8, j * 8 : (j + 1) * 8]

                if np.mean(block_mag) < value:
                    pixel_gray = gray[i * 8, j * 8]
                    pixel_color = color[i * 8, j * 8]
                    index_gray = self.get_ascii_index_for_pixel(pixel_gray)
                    ascii_text_image = self.get_ascii_image_by_index(index_gray, True).copy()
                    ascii_text_image_color = self.set_color_of_ascii_image(ascii_text_image, pixel_color)
                    f_image[i * 8 : (i + 1) * 8, j * 8 : (j + 1) * 8] = (ascii_text_image_color)
                    continue

                angle = orientation_downscaled[i * 8, j * 8]
                pixel_color = color[i * 8, j * 8]
                index = self.get_ascii_orientation_index_for_angle(angle)
                ascii_orientation_image = self.get_ascii_orientation_image_by_index(index, True).copy()
                ascii_orientation_image_color = self.set_color_of_ascii_image(ascii_orientation_image, pixel_color)
                f_image[i * 8 : (i + 1) * 8, j * 8 : (j + 1) * 8] = (ascii_orientation_image_color)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"image_{timestamp}.png"
        save_path = Path.cwd() / filename
        cv2.imwrite(str(save_path), f_image.astype(np.uint8))

    def convert_image_to_ascii_text_complex_color(self, image_path, mag_threshold):
        gray, image, color, d_width, d_height, rows, cols = self.prepare_images(image_path,True)

        magnitude_downscaled, orientation_downscaled, magnitude = self.compute_gradients(gray, rows, cols)

        nonzero_magnitudes = magnitude_downscaled[magnitude_downscaled > 0]
        threshold = np.percentile(nonzero_magnitudes, 90)

        result = "\n"

        value = 30
        if str(mag_threshold).lower() == "auto":
            value = threshold
        else:
            try:
                int_value = int(mag_threshold)
                if 0 <= int_value <= 255:
                    value = int_value
                else:
                    print("Error: magnitude must be between 0 and 255")
                    return
            except ValueError:
                print(
                    "Error: invalid magnitude parameter (must be 'auto' or an integer between 0 and 255)"
                )
                return

        for i in range(d_height):
            for j in range(d_width):
                block_mag = magnitude[i * 8 : (i + 1) * 8, j * 8 : (j + 1) * 8]

                if np.mean(block_mag) < value:
                    pixel_gray = gray[i * 8, j * 8]
                    pixel_color = color[i * 8, j * 8]
                    index_gray = self.get_ascii_index_for_pixel(pixel_gray)
                    b = pixel_color[0]
                    g = pixel_color[1]
                    r = pixel_color[2]
                    result += (
                        "\033[38;2;"
                        + str(r)
                        + ";"
                        + str(g)
                        + ";"
                        + str(b)
                        + ";m"
                        + self.ascii_list[index_gray] * 2
                        + "\033[0m"
                    )
                    continue

                angle = orientation_downscaled[i * 8, j * 8]
                pixel_color = color[i * 8, j * 8]
                index = self.get_ascii_orientation_index_for_angle(angle)
                b = pixel_color[0]
                g = pixel_color[1]
                r = pixel_color[2]
                result += (
                    "\033[38;2;"
                    + str(r)
                    + ";"
                    + str(g)
                    + ";"
                    + str(b)
                    + ";m"
                    + self.ascii_orientation_list[index] * 2
                    + "\033[0m"
                )
            result += "\n"

        return result

    def show_help(self):
        print("Usage:")
        print("  p2ascii help                              Show this help message")
        print()
        print("  # SIMPLE CONVERSION WITHOUT EDGE DETECTION")
        print("  p2ascii sc2image <img>                    Convert image to ASCII image")
        print("  p2ascii sc2text <img>                     Convert image to ASCII text")
        print("  p2ascii sc2cimage <img>                   Convert image to ASCII image with color")
        print("  p2ascii sc2ctext <img>                    Convert image to ASCII text with color")
        print()
        print("  # COMPLEX CONVERSION WITH EDGE DETECTION")
        print("  p2ascii cc2image <img> <thresh>           Convert image to ASCII image")
        print("  p2ascii cc2text <img> <thresh>            Convert image to ASCII text")
        print("  p2ascii cc2cimage <img> <thresh>          Convert image to ASCII image with color")
        print("  p2ascii cc2ctext <img> <thresh>           Convert image to ASCII text with color")
        print()
        print("Note on <thresh>: ")
        print("  <thresh> defines the edge detection threshold:")
        print("    - Higher values result in fewer edges (stricter)")
        print("    - Lower values result in more edges (more permissive)")
        print("    - 'auto' sets the threshold automatically using a 90th percentile of the nonzero gradient magnitudes:")

    def run(self):
        if len(sys.argv) < 2:
            print("Missing command. Try 'p2ascii help'")
            return

        command = sys.argv[1]

        if command == "help":
            self.show_help()
            return

        if len(sys.argv) < 3:
            print("No image specified")
            return

        match command:

            case "sc2image":
                self.convert_image_to_ascii_image_simple(sys.argv[2])

            case "sc2cimage":
                self.convert_image_to_ascii_image_simple_color(sys.argv[2])

            case "sc2text":
                print(self.convert_image_to_ascii_text_simple(sys.argv[2]))
            
            case "sc2ctext":
                print(self.convert_image_to_ascii_text_simple_color(sys.argv[2]))

            case "cc2image":
                if len(sys.argv) < 4:
                    print("Error: need argument for mag threshold")
                    return
                self.convert_image_to_ascii_image_complex(sys.argv[2], sys.argv[3])

            case "cc2text":
                if len(sys.argv) < 4:
                    print("Error: need argument for mag threshold")
                    return
                print(self.convert_image_to_ascii_text_complex(sys.argv[2], sys.argv[3]))

            case "cc2cimage":
                if len(sys.argv) < 4:
                    print("Error: need argument for mag threshold")
                    return
                self.convert_image_to_ascii_image_complex_color(sys.argv[2], sys.argv[3])

            case "cc2ctext":
                if len(sys.argv) < 4:
                    print("Error: need argument for mag threshold")
                    return
                print(self.convert_image_to_ascii_text_complex_color(sys.argv[2], sys.argv[3]))

            case _:
                print(f"Unknown command: {command}")
                print("Try 'p2ascii help'")


if __name__ == "__main__":
    program = P2Ascii()
    program.run()