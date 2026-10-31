"""Feature-independent detection windows. Coordinates are half-open (x0,y0,x1,y1)."""
SIZES = (8, 16, 24, 32, 48, 64)
WINDOWS = tuple((s, s * ratio) for s in SIZES for ratio in (1, 2))


def axis_positions(length, window, stride):
    if min(length, window, stride) <= 0:
        raise ValueError('Dimensions and strides must be positive')
    if window > length:
        return ()
    positions = list(range(0, length - window + 1, stride))
    if positions[-1] != length - window:
        positions.append(length - window)
    return positions


def sliding_windows(image_width, image_height, width, height, stride_x, stride_y):
    """Yield boxes without reading pixels; include right/bottom aligned windows."""
    for y in axis_positions(image_height, height, stride_y):
        for x in axis_positions(image_width, width, stride_x):
            yield (x, y, x + width, y + height)


def best_window(box, image_width, image_height, width, height, stride_x, stride_y):
    """Exact best grid IoU in O(number of x positions + number of y positions).

    Intersection is separable and union decreases as intersection increases for
    fixed box/window areas, so independently maximizing each axis is exact.
    """
    xs = axis_positions(image_width, width, stride_x)
    ys = axis_positions(image_height, height, stride_y)
    if not xs or not ys:
        return 0.0, None
    x0, y0, x1, y1 = box
    overlap = lambda p, size, a, b: max(0, min(p + size, b) - max(p, a))
    x = max(xs, key=lambda p: overlap(p, width, x0, x1))
    y = max(ys, key=lambda p: overlap(p, height, y0, y1))
    intersection = overlap(x, width, x0, x1) * overlap(y, height, y0, y1)
    union = (x1-x0)*(y1-y0) + width*height - intersection
    return intersection / union, (x, y, x+width, y+height)
