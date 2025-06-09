# 🖼️ p2ascii

**p2ascii** is a Python-based tool that converts images into ASCII art. It supports:

- **Simple mode**: brightness-based mapping of pixels to ASCII characters.
- **Complex mode**: edge-aware rendering using **Sobel gradients** to compute orientation and magnitude.

It can output:
- ASCII **text**, optionally with ANSI color codes.
- ASCII **images**, where characters are rendered as image blocks.

---

## 📌 Features

- Sobel-based edge detection for orientation-aware rendering.
- Optional color support for both text and image modes.
- Configurable edge sensitivity (`<thresh>`).
- Terminal or image file output.
- Simple and complex conversion modes.

---

## ⚙️ Requirements

- Python 3.7+
- [`opencv-python`](https://pypi.org/project/opencv-python/)
- [`numpy`](https://pypi.org/project/numpy/)

## ✅ Usage

### 🔹 Simple Conversion (No Edge Detection)

  - sc2image <img>	Convert image to ASCII image
  - sc2text <img>	Convert image to ASCII text (stdout)
  - sc2cimage <img>	Colored ASCII image
  - sc2ctext <img>	Colored ASCII text (stdout)
### 🔸 Complex Conversion (With Edge Detection)

  - cc2image <img> <thresh>	ASCII image using edge orientation
  - cc2text <img> <thresh>	ASCII text with edge symbols (stdout)
  - cc2cimage <img> <thresh>	Colored ASCII image with edge awareness
  - cc2ctext <img> <thresh>	Colored ASCII text with edge symbols
