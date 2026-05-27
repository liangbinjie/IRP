import cv2
import numpy as np

def centrar(img, size=28, padding=4):

    # Ensure binary
    _, img = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)

    # Find white pixels directly
    coords = cv2.findNonZero(img)

    if coords is None:
        return np.zeros((size, size), dtype=np.uint8)

    # Bounding box around digit
    x, y, w, h = cv2.boundingRect(coords)

    digit = img[y:y+h, x:x+w]

    # Compute scaling
    target = size - 2 * padding

    if w > h:
        new_w = target
        new_h = max(1, int(h * target / w))
    else:
        new_h = target
        new_w = max(1, int(w * target / h))

    # Resize digit
    digit = cv2.resize(
        digit,
        (new_w, new_h),
        interpolation=cv2.INTER_NEAREST
    )

    # Black canvas
    canvas = np.zeros((size, size), dtype=np.uint8)

    # Center coordinates
    x_offset = (size - new_w) // 2
    y_offset = (size - new_h) // 2

    # Paste digit
    canvas[
        y_offset:y_offset+new_h,
        x_offset:x_offset+new_w
    ] = digit

    return canvas