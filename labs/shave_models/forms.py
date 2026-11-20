"""A screen-space model for locating four Windows Forms control types."""
import numpy as np
from PIL import Image


ROLES = ["button", "textbox", "checkbox", "radio"]
GRID = (32, 48)


def pixels(image):
    gray = image.convert("L").resize((192, 128), Image.BILINEAR)
    return ((np.asarray(gray, dtype=np.float32) - 190) / 64)[None]


def points(scores):
    positions = np.asarray(scores).reshape(4, -1).argmax(1)
    y, x = np.unravel_index(positions, GRID)
    return np.stack(((x + 0.5) / GRID[1], (y + 0.5) / GRID[0]), axis=1)


def hit_test(scores, boxes):
    xy = points(scores)
    return ((xy >= boxes[:, :2]) & (xy < boxes[:, :2] + boxes[:, 2:])).all(1)
