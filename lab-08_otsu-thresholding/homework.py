import marimo

__generated_with = "0.23.1"
app = marimo.App(width="medium")

with app.setup:
    import marimo as mo

    import cv2
    import numpy as np
    from matplotlib import pyplot as plt

    plt.rcParams['figure.figsize'] = [15, 10]

    GrayImage = np.ndarray[tuple[int, int], np.dtype[np.uint8]]


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    # HOMEWORK 8

    In this homework you are going to implement your first machine learning
    algorithm to automatically binarize document images. The goal of
    document binarization is to seprate the characters (letters) from
    everything else. This is the crucial part for automatic document
    understanding and information extraction from the document. In order to
    do so, you will use the Otsu thresholding algorithm.

    At the end of this notebook, there are a couple of questions for you to
    answer.
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    Let’s load the document image we will be working on in this homework.
    """)
    return


@app.cell
def _():
    img_bgr = cv2.imread('./data/document.jpg')
    img: GrayImage = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)

    plt.imshow(img, cmap='gray')
    return (img,)


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    First, let’s have a look at the histogram.
    """)
    return


@app.cell
def _(img: GrayImage):
    hist, edges = np.histogram(img, 256)

    plt.bar(edges[:-1], hist)
    (plt.xlabel('Colour'), plt.ylabel('Count'))
    plt.grid(True)

    plt.show()
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ### Otsu Thresholding

    Let’s now implement the Otsu thresholding algorithm. Remember that the
    algorithm consists of an optimization process that finds the thresholds
    that minimizes the intra-class variance or, equivalently, maximizes the
    inter-class variance.

    In this homework, you are going to demonstrate the working principle of
    the Otsu algorithm. Therefore, you won’t have to worry about an
    efficient implementation, we are going to use the brute force approach
    here.
    """)
    return


@app.cell
def _(img: GrayImage):
    rows, cols = img.shape[:2]

    num_pixels = rows * cols
    best_wcv = 1000000.0
    opt_th: int = 0

    for th in range(0, 256):
        foreground = img[img >= th]
        background = img[img < th]

        if len(foreground) == 0 or len(background) == 0:
            continue

        omega_f = np.sum(foreground) / num_pixels
        omega_b = np.sum(background) / num_pixels

        sigma_f = np.var(foreground)
        sigma_b = np.var(background)

        wcv = omega_f * sigma_f + omega_b * sigma_b

        if wcv < best_wcv:
            best_wcv = wcv
            opt_th = th

    print('Optimal threshold', opt_th)
    return (opt_th,)


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    Finally, let’s compare the original image and its thresholded
    representation.
    """)
    return


@app.cell
def _(img: GrayImage, opt_th: int):
    plt.subplot(121), plt.imshow(img, cmap='gray')
    plt.subplot(122), plt.imshow(img > opt_th, cmap='gray')
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ### Questions

    - Looking at the computed histogram, could it be considered bimodal?

    Pixels with gray intensity around 120 refer to the pixels of the text and
    one at 200 – the newspaper paper:) I *would* consider the histogram
    bimodal, but my example with a book (below) show a more obvious bimodal
    histogram:
    """)
    return


@app.cell
def _():
    img2_bgr = cv2.imread('./data/book.jpg')
    img2: GrayImage = cv2.cvtColor(img2_bgr, cv2.COLOR_BGR2GRAY)

    plt.subplot(121), plt.imshow(img2, cmap='gray')
    plt.subplot(122), plt.imshow(img2 > 146, cmap='gray') # Optimal threshold
    return (img2,)


@app.cell
def _(img2: GrayImage):
    hist2, edges2 = np.histogram(img2, 256)

    plt.bar(edges2[:-1], hist2)
    (plt.xlabel('Colour'), plt.ylabel('Count'))
    plt.grid(True)

    plt.show()
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    - Looking at the computed histogram, what binarization threshold would you
      chose? Why?

    I can’t say precisely, but the value I would pick will be in range
    between 155 and 175. It should be in the valley between the peaks we can see on the bimodal histogram.

    But if I was asked to
    make a guess by hand, I would look for the lightest text
    on the image and will set it’s gray intensity as a threshold. That way I
    would pick 170 (instead of computed 167) as a threshold so that the first lines of “Die Themen…” section are more
    readable. But technically my decision would cause more noise to appear
    """)
    return


@app.cell
def _(img: GrayImage, opt_th: int):
    plt.subplot(121), plt.imshow(img > 170, cmap='gray')
    plt.subplot(122), plt.imshow(img > opt_th, cmap='gray')
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    - Looking at the resulting (thresholded) image, is the text binarization
      (detection) good?

    Yes, Otsu’s method did a good job
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## Additional task 1

    I'll add a couple of examples
    """)
    return


@app.cell
def _():
    img3: GrayImage = cv2.cvtColor(cv2.imread('./data/coins.jpeg'), cv2.COLOR_BGR2GRAY)

    plt.imshow(img3, cmap='gray')
    return (img3,)


@app.cell
def _(img3: GrayImage):
    hist3, edges3 = np.histogram(img3, 256)

    plt.bar(edges3[:-1], hist3)
    (plt.xlabel('Colour'), plt.ylabel('Count'))
    plt.grid(True)

    plt.show()
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    This one doesn't look like a bimodal histogram
    """)
    return


@app.cell
def _():
    img4: GrayImage = cv2.cvtColor(cv2.imread('./data/gold.jpg'), cv2.COLOR_BGR2GRAY)

    plt.imshow(img4, cmap='gray')
    return (img4,)


@app.cell
def _(img4: GrayImage):
    hist4, edges4 = np.histogram(img4, 256)

    plt.bar(edges4[:-1], hist4)
    (plt.xlabel('Colour'), plt.ylabel('Count'))
    plt.grid(True)

    plt.show()
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    The histogram of this one looks more like a bimodal one.
    """)
    return


@app.cell
def _(img4: GrayImage):
    plt.subplot(121), plt.imshow(img4, cmap='gray')
    plt.subplot(122), plt.imshow(img4 > 82, cmap='gray') # Optimal threshold
    return


if __name__ == "__main__":
    app.run()
