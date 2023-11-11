from PIL import Image, ImageDraw, ImageFont
import io
import numpy as np
import math
from collections.abc import Iterable


def plot_to_image(p, *args, **kwargs):
    """
    Convert a plotnine plot to a PIL Image and return it
    """
    img_buf = io.BytesIO()
    p.save(img_buf, format='png', verbose=False, *args, **kwargs)
    im = Image.open(img_buf).copy()
    img_buf.close()
    return im


def draw_simple_text(width, height, text, font_size=24, color='black', background_color='white',
                font_path='/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf',
                direction='ltr', anchor='mm'):
    font = ImageFont.truetype(font_path, font_size)
    im = Image.new("RGB", (width, height), background_color)
    draw = ImageDraw.Draw(im)
    draw.text((width/2, height/2), text, fill=color, font=font, direction=direction, anchor=anchor)
    return im


def compose_grid(imgs: list|dict, rows: int = None, cols: int = None, line_width=1, line_color=(255, 255, 255),
                 row_names=None, col_names=None, 
                 axis_font_size=24, axis_font_color='black', axis_background_color='white',
                 font_path='/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf'):
    """
    imgs: list or dict of Pillow images
        If dict, keys should be (row, column) tuples
        If list, can be a list of images or a list of lists of images
        If imgs if either a dict or list of lists, rows and cols can be None.
    rows: number of rows
    cols: number of columns
    line_width: width of lines between images
    line_color: color of lines between images
    row_names: list of row names (should equal number of rows)
    col_names: list of column names (should equal number of columns)
    row_name_size: font size of row names
    col_name_size: font size of column names
    """
    if isinstance(imgs, dict):
        empty_img = Image.fromarray(np.zeros(next(imgs.values()).size, dtype=bool))
        if rows is None:
            rows = max([k[0] for k in imgs.keys()]) + 1
        if cols is None:
            cols = max([k[1] for k in imgs.keys()]) + 1
        imgs = [imgs[(i, j)] if (i, j) in imgs else empty_img for i in range(rows) for j in range(cols)]
    elif isinstance(imgs[0], Iterable): # list of lists
        empty_img = Image.fromarray(np.zeros(imgs[0][0].size, dtype=bool))
        rows = len(imgs)
        cols = max([len(row) for row in imgs])
        flat_imgs = []
        for row in imgs:
            flat_imgs.extend(row)
            flat_imgs.extend([empty_img] * (cols - len(row)))
        imgs = flat_imgs
    else:
        if rows is None:
            rows = math.ceil(len(imgs) // cols)
        elif cols is None:
            cols = math.ceil(len(imgs) // rows)
            
        empty_img = Image.fromarray(np.zeros(imgs[0].size, dtype=bool))
        if len(imgs) > rows * cols:
            imgs = imgs[:rows*cols]
        elif len(imgs) < rows * cols:
            imgs = imgs + [Image.fromarray(np.zeros(imgs[0].size, dtype=bool))] * (rows*cols - len(imgs))

    w, h = imgs[0].size
    width = cols*w + line_width * (cols - 1)
    height = rows*h + line_width * (rows - 1)
    row_name_width = 0
    col_name_height = 0
    if row_names is not None:
        row_name_width = line_width + int(axis_font_size * 1.4)
        width += row_name_width
    if col_names is not None:
        col_name_height = line_width + int(axis_font_size * 1.4)
        height += col_name_height
        
    grid = Image.new('RGB', size=(width, height), color=line_color)

    if row_names is not None:
        for i, name in enumerate(row_names):
            img= draw_simple_text(row_name_width - line_width, h, name, font_path=font_path, direction='ttb',
                                color=axis_font_color, background_color=axis_background_color)
            grid.paste(img, box=(0, col_name_height + i * (line_width + h)))
    if col_names is not None:
        for i, name in enumerate(col_names):
            img= draw_simple_text(w, col_name_height - line_width, name, font_path=font_path,
                                color=axis_font_color, background_color=axis_background_color)
            grid.paste(img, box=(row_name_width + i%cols*(line_width+w), 0))
    
    for i, img in enumerate(imgs):
        x = row_name_width + i%cols*(line_width+w)
        y = col_name_height + i//cols*(line_width+h)
        grid.paste(img, box=(x, y))
    return grid
