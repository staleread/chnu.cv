import marimo

__generated_with = "0.23.1"
app = marimo.App(width="medium", auto_download=["ipynb"])

with app.setup:
    import marimo as mo  # comment the import if not using marimo notebook editor
    import cv2 as cv
    import dlib
    import numpy as np
    import matplotlib.pyplot as plt
    import random
    from time import perf_counter
    from typing import NamedTuple

    # Some useful types
    Color = tuple[int, int, int]
    RgbImage = np.ndarray[tuple[int, int, int], np.dtype[np.uint8]]
    GrayImage = np.ndarray[tuple[int, int], np.dtype[np.uint8]]

    class Image(NamedTuple):
        name: str
        rgb: RgbImage
        gray: GrayImage


@app.cell
def _():
    # Some utility functions

    def load_image(name: str, filepath: str | None = None) -> Image:
        bgr = cv.imread(filepath if filepath else f'data/{name}.jpg')
        rgb: RgbImage = cv.cvtColor(bgr, cv.COLOR_BGR2RGB)
        gray: GrayImage = cv.cvtColor(bgr, cv.COLOR_BGR2GRAY)

        return Image(name, rgb, gray)

    def load_images_by_names(names: list[str]) -> list[Image]:
        return [load_image(name) for name in names]

    def random_color() -> tuple[int, int, int]:
        return (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))

    return load_images_by_names, random_color


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    # CV lab 9. Face detection
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    First of all, let's take a look at the photos selected for the experiment
    """)
    return


@app.cell
def _(load_images_by_names):
    images: list[Image] = load_images_by_names([
        "in-glasses",
        "paint",
        "bike",
        "cave",
        "group",
        "masks",
        "run",  # or "fans"
        "beatles",
        "drawing"
        ])
    return (images,)


@app.cell
def _(images: list[Image]):
    # Plotting images in a 3x3 grid

    _fig, _axes = plt.subplots(3, 3, figsize=(12, 9))

    for i, (_ax, _img) in enumerate(zip(_axes.flatten(), images)):
        _ax.imshow(_img.rgb)
        _ax.set_title(f"{i}: {_img.name}")
        _ax.axis('off')

    plt.tight_layout()
    plt.show()
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    Now we'll load the `dlib` **frontal** face detector and define a utility function for plotting the photo with rectangles on detected faces
    """)
    return


@app.cell
def _():
    detector = dlib.get_frontal_face_detector()
    return (detector,)


@app.cell
def _(detector, random_color):
    def show_detected_faces_dlib(
        img: Image,
        *,
        ax,
        thickness: int = 2,
        color: Color | None = None,
        gray: bool = False,
        scale: float = 1.0,
    ):
        # Work on a copy for visualization
        result = np.copy(img.gray if gray else img.rgb)

        # Resize the image on demand
        if scale != 1.0:
            h_old, w_old = result.shape[:2]
            result = cv.resize(result, (0, 0), fx=scale, fy=scale)
            h_new, w_new = result.shape[:2]

            print(f'[DLIB] scaled the image from {w_old}x{h_old} to {w_new}x{h_new}')

        # Detect faces (upsampling = 1)
        start = perf_counter()
        rects = detector(result, 1)
        end = perf_counter()

        elapsed_time = end - start

        print('[DLIB] Number of detected faces:', len(rects), 'Elapsed time (s)', elapsed_time)

        # Draw rectangles around faces
        for rect in rects:
            cv.rectangle(
                result,
                (rect.left(), rect.top()),
                (rect.right(), rect.bottom()),
                color or random_color(),
                thickness)

        ax.imshow(result, cmap="gray" if gray else None)
        ax.axis('off')

    return (show_detected_faces_dlib,)


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## Detecting single face

    Let's test some solo photos in RGB mode
    """)
    return


@app.cell
def _(images: list[Image], show_detected_faces_dlib):
    _fig, _axes = plt.subplots(2, 2, figsize=(12, 9))
    _axes_flat = _axes.flatten()

    show_detected_faces_dlib(images[0], ax=_axes_flat[0], thickness=20)
    show_detected_faces_dlib(images[1], ax=_axes_flat[1], thickness=10)
    show_detected_faces_dlib(images[2], ax=_axes_flat[2], thickness=10)
    show_detected_faces_dlib(images[3], ax=_axes_flat[3], thickness=5)

    plt.show()
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ```
    [DLIB] Number of detected faces: 1 Elapsed time (s) 1.3623237540014088
    [DLIB] Number of detected faces: 1 Elapsed time (s) 0.45125001600172254
    [DLIB] Number of detected faces: 1 Elapsed time (s) 0.6959608489996754
    [DLIB] Number of detected faces: 1 Elapsed time (s) 0.33291933899818105
    ```

    | Image | Distance | Image size | Processing time (s) |
    |---|---|---|----|
    | 1 | Selfie |  1808x3216 | 1.348 |
    | 1 | Portrait |  1200x1600 | 0.447 |
    | 1 | ~2 meters |  1741x1741 | 0.694 |

    And in gray mode:
    """)
    return


@app.cell
def _(images: list[Image], show_detected_faces_dlib):
    _fig, _axes = plt.subplots(2, 2, figsize=(12, 9))
    _axes_flat = _axes.flatten()

    show_detected_faces_dlib(images[0], ax=_axes_flat[0], thickness=20, gray=True)
    show_detected_faces_dlib(images[1], ax=_axes_flat[1], thickness=10, gray=True)
    show_detected_faces_dlib(images[2], ax=_axes_flat[2], thickness=10, gray=True)
    show_detected_faces_dlib(images[3], ax=_axes_flat[3], thickness=5, gray=True)

    plt.show()
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ```
    [DLIB] Number of detected faces: 1 Elapsed time (s) 0.8925083799986169
    [DLIB] Number of detected faces: 1 Elapsed time (s) 0.29759324100086815
    [DLIB] Number of detected faces: 1 Elapsed time (s) 0.4645231469985447
    [DLIB] Number of detected faces: 1 Elapsed time (s) 0.22433764200104633
    ```

    Here's the table comparing RGB (high resolution) vs Grayscale processing times with speed improvements:

    | Image | Distance | RGB time (s) | Gray time (s) | Speed improvement (%) |
    |---|---|----|----|---|
    | 1 | Selfie | 1.348 | 0.905 | 32.8% ⚡ |
    | 2 | Portrait | 0.447 | 0.301 | 32.7% ⚡ |
    | 3 | ~2 meters | 0.694 | 0.471 | 32.1% |

    The grayscale mode consistently delivers approximately **33% speed improvement** across all high-resolution image sizes

    Now let's experiment with lower image resolution!
    """)
    return


@app.cell
def _(images: list[Image], show_detected_faces_dlib):
    _fig, _axes = plt.subplots(2, 2, figsize=(12, 9))
    _axes_flat = _axes.flatten()

    show_detected_faces_dlib(images[0], ax=_axes_flat[0], thickness=1, scale=0.03)
    show_detected_faces_dlib(images[1], ax=_axes_flat[1], thickness=1, scale=0.095)
    show_detected_faces_dlib(images[2], ax=_axes_flat[2], thickness=2, scale=0.158)
    show_detected_faces_dlib(images[3], ax=_axes_flat[3], thickness=2, scale=0.45)

    plt.show()
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ```
    [DLIB] scaled the image from 1808x3216 to 54x96
    [DLIB] Number of detected faces: 1 Elapsed time (s) 0.001399556000251323
    [DLIB] scaled the image from 1200x1600 to 114x152
    [DLIB] Number of detected faces: 1 Elapsed time (s) 0.004598046001774492
    [DLIB] scaled the image from 1741x1741 to 275x275
    [DLIB] Number of detected faces: 1 Elapsed time (s) 0.01821794599891291
    [DLIB] scaled the image from 901x1600 to 405x720
    [DLIB] Number of detected faces: 1 Elapsed time (s) 0.06977680300042266
    ```

    | Image | Distance | Compressed size | Processing time (s) | Times faster |
    |---|---|---|----|---|
    | 1 | Selfie |  54x96 | 0.002 | 674 🚀 |
    | 1 | Portrait |  114x152 | 0.005 | 89 🔥 |
    | 1 | ~2 meters |  275x275 | 0.018 | 38 |

    As the results show, the closer is the face, the more can we compress the image and the detector can do the job faster. Using image compression lead to huge speedup of face detection
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    Let's try to process the compressed pic using gray mode:
    """)
    return


@app.cell
def _(images: list[Image], show_detected_faces_dlib):
    _fig, _axes = plt.subplots(2, 2, figsize=(12, 9))
    _axes_flat = _axes.flatten()

    show_detected_faces_dlib(images[0], ax=_axes_flat[0], thickness=1, scale=0.03, gray=True)
    show_detected_faces_dlib(images[1], ax=_axes_flat[1], thickness=1, scale=0.095, gray=True)
    show_detected_faces_dlib(images[2], ax=_axes_flat[2], thickness=2, scale=0.158, gray=True)
    show_detected_faces_dlib(images[3], ax=_axes_flat[3], thickness=2, scale=0.45, gray=True)

    plt.show()
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ```
    [DLIB] scaled the image from 1808x3216 to 54x96
    [DLIB] Number of detected faces: 1 Elapsed time (s) 0.0011582459992496297
    [DLIB] scaled the image from 1200x1600 to 114x152
    [DLIB] Number of detected faces: 1 Elapsed time (s) 0.003683079998154426
    [DLIB] scaled the image from 1741x1741 to 275x275
    [DLIB] Number of detected faces: 1 Elapsed time (s) 0.012993604999792296
    [DLIB] scaled the image from 901x1600 to 405x720
    [DLIB] Number of detected faces: 1 Elapsed time (s) 0.051755240998318186
    ```

    As we can see, the algorithm did manage to detect my face without colour information and with the same comprassion rate

    | Image | Distance | Compressed size | RGB time (s) | Gray time (s) | Speed improvement (%) |
    |---|---|---|----|----|---|
    | 1 | Selfie | 54x96 | 0.002 | 0.001 | 50% ⚡ |
    | 2 | Portrait | 114x152 | 0.005 | 0.003 | 40% ⚡ |
    | 3 | ~2 meters | 275x275 | 0.018 | 0.013 | 28% |

    The grayscale mode provides consistent speed improvements across all image sizes, with the most significant gains on smaller/compressed images (50% faster for the selfie).
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    # Detecting multiple faces
    """)
    return


@app.cell
def _(images: list[Image], show_detected_faces_dlib):
    _fig, _axes = plt.subplot_mosaic([['a', 'b'], ['c', 'd'], ['e', 'e']], figsize=(15, 15))

    show_detected_faces_dlib(images[4], ax=_axes['a'], thickness=15)
    show_detected_faces_dlib(images[5], ax=_axes['b'], thickness=5)
    show_detected_faces_dlib(images[7], ax=_axes['c'], thickness=10)
    show_detected_faces_dlib(images[8], ax=_axes['d'], thickness=10)
    show_detected_faces_dlib(images[6], ax=_axes['e'], thickness=10)

    plt.show()
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ```
    [DLIB] Number of detected faces: 9 Elapsed time (s) 2.1163107080028567
    [DLIB] Number of detected faces: 4 Elapsed time (s) 0.048051828001916874
    [DLIB] Number of detected faces: 3 Elapsed time (s) 0.2780568609996408
    [DLIB] Number of detected faces: 3 Elapsed time (s) 0.1571259039992583
    [DLIB] Number of detected faces: 77 Elapsed time (s) 5.681187735001004
    ```
    """)
    return


@app.cell
def _(images: list[Image], show_detected_faces_dlib):
    _fig, _axes = plt.subplot_mosaic([['a', 'b'], ['c', 'd'], ['e', 'e']], figsize=(15, 15))

    show_detected_faces_dlib(images[4], ax=_axes['a'], thickness=12, gray=True)
    show_detected_faces_dlib(images[5], ax=_axes['b'], thickness=5, gray=True)
    show_detected_faces_dlib(images[7], ax=_axes['c'], thickness=10, gray=True)
    show_detected_faces_dlib(images[8], ax=_axes['d'], thickness=10, gray=True)
    show_detected_faces_dlib(images[6], ax=_axes['e'], thickness=10, gray=True)

    plt.show()
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ```
    [DLIB] Number of detected faces: 10 Elapsed time (s) 1.4383498239985784
    [DLIB] Number of detected faces: 4 Elapsed time (s) 0.034515147999627516
    [DLIB] Number of detected faces: 3 Elapsed time (s) 0.1903794800018659
    [DLIB] Number of detected faces: 3 Elapsed time (s) 0.10736422700210824
    [DLIB] Number of detected faces: 76 Elapsed time (s) 3.8629781440031365
    ```

    Here's the complete table showing precision loss across different scales:

    | Resolution | Detected faces | Detection loss | Processing time (s) | Speed improvement |
    |---|---|---|----|----|
    | 4080x2296 (original) | 9 | — | 2.132 | baseline |
    | 2040x1148 (0.5 scale) | 8 | 1 face (11% loss) | 0.541 | 3.9x faster ⚡ |
    | 408x230 (0.1 scale) | 2 | 7 faces (78% loss) ❌ | 0.022 | 96.8x faster 🚀 |

    **Key findings:**
    - **0.5 scale (50% compression)**: Minimal precision loss (1 face) with 3.9x speed improvement — good balance
    - **0.1 scale (10% compression)**: Massive speed gain (96.8x faster) but unacceptable precision loss (78%)
    """)
    return


@app.cell
def _(images: list[Image], show_detected_faces_dlib):
    _fig, _axes = plt.subplots(1, 3, figsize=(12, 9))

    show_detected_faces_dlib(images[7], ax=_axes[0], thickness=7, scale=0.7)
    show_detected_faces_dlib(images[7], ax=_axes[1], thickness=5, scale=0.5)
    show_detected_faces_dlib(images[7], ax=_axes[2], thickness=3, scale=0.4)

    plt.show()
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ```
    [DLIB] scaled the image from 1204x995 to 843x696
    [DLIB] Number of detected faces: 3 Elapsed time (s) 0.13740770900039934
    [DLIB] scaled the image from 1204x995 to 602x498
    [DLIB] Number of detected faces: 4 Elapsed time (s) 0.07024258800083771
    [DLIB] scaled the image from 1204x995 to 482x398
    [DLIB] Number of detected faces: 3 Elapsed time (s) 0.04472596399864415
    ```

    Here's the table showing precision loss across different scales for the multi-face image:

    | Resolution | Scale | Detected faces | Detection loss | Processing time (s) | Speed improvement |
    |---|---|---|---|----|----|
    | 3504x2336 (original) | 1.0 | 201 | — | 1.86 | baseline |
    | 2453x1635 | 0.7 | 192 | 9 faces (4.5% loss) | 0.916 | 2.0x faster ⚡ |
    | 1752x1168 | 0.5 | 132 | 69 faces (34.3% loss) | 0.466 | 4.0x faster ⚡ |
    | 1402x934 | 0.4 | 17 | 184 faces (91.5% loss) ❌ | 0.297 | 6.3x faster 🚀 |

    **Key findings:**
    - **0.7 scale**: Excellent balance — minimal loss (4.5%) with 2x speed improvement
    - **0.5 scale**: Moderate compression — acceptable for many uses (34% loss, 4x faster)
    - **0.4 scale**: Severe precision degradation (91.5% loss) despite 6.3x speed gain
    """)
    return


@app.cell
def _(images: list[Image], show_detected_faces_dlib):
    _fig, _axes = plt.subplots(1, 3, figsize=(12, 9))

    show_detected_faces_dlib(images[6], ax=_axes[0], thickness=7, scale=0.4)
    show_detected_faces_dlib(images[6], ax=_axes[1], thickness=5, scale=0.33)
    show_detected_faces_dlib(images[6], ax=_axes[2], thickness=3, scale=0.3)

    plt.show()
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ```
    [DLIB] scaled the image from 6048x4024 to 2419x1610
    [DLIB] Number of detected faces: 75 Elapsed time (s) 0.8857080109992239
    [DLIB] scaled the image from 6048x4024 to 1996x1328
    [DLIB] Number of detected faces: 58 Elapsed time (s) 0.6056475620025594
    [DLIB] scaled the image from 6048x4024 to 1814x1207
    [DLIB] Number of detected faces: 48 Elapsed time (s) 0.49828719300057855
    ```
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## Viola-Jones face detection
    """)
    return


@app.cell
def _():
    face_cascade = cv.CascadeClassifier('data/haarcascade_frontalface_default.xml')
    return (face_cascade,)


@app.cell
def _(face_cascade, random_color):
    def show_detected_faces_vj(
        img: Image,
        *,
        ax,
        thickness: int = 2,
        color: Color | None = None,
        gray: bool = False,
        scale: float = 1.0,
    ):
        # Work on a copy for visualization
        result = np.copy(img.gray if gray else img.rgb)

        # Resize the image if needed
        if scale != 1.0:
            h_old, w_old = result.shape[:2]
            result = cv.resize(result, (0, 0), fx=scale, fy=scale)
            h_new, w_new = result.shape[:2]

            print(f'[VJ] scaled the image from {w_old}x{h_old} to {w_new}x{h_new}')

        start = perf_counter()
        faces = face_cascade.detectMultiScale(result,
                                             scaleFactor=1.1,
                                             minNeighbors=10,
                                             flags=cv.CASCADE_SCALE_IMAGE)
        end = perf_counter()
        elapsed_time = end - start

        print('[VJ] Number of detected faces:', len(faces), 'Elapsed time (s)', elapsed_time)

        # Draw rectangles around faces
        for (x, y, w, h) in faces: 
            cv.rectangle(result, (x, y), (x+w, y+h), color or random_color(), thickness)

        ax.imshow(result, cmap="gray" if gray else None)
        ax.axis('off')

    return (show_detected_faces_vj,)


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## Detecting single face

    Let's test some solo photos in RGB mode
    """)
    return


@app.cell
def _(images: list[Image], show_detected_faces_vj):
    _fig, _axes = plt.subplots(2, 2, figsize=(12, 9))
    _axes_flat = _axes.flatten()

    show_detected_faces_vj(images[0], ax=_axes_flat[0], thickness=20)
    show_detected_faces_vj(images[1], ax=_axes_flat[1], thickness=10)
    show_detected_faces_vj(images[2], ax=_axes_flat[2], thickness=10)
    show_detected_faces_vj(images[3], ax=_axes_flat[3], thickness=10)

    plt.show()
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ```
    [VJ] Number of detected faces: 1 Elapsed time (s) 0.13776485600101296
    [VJ] Number of detected faces: 2 Elapsed time (s) 0.11216415699891513
    [VJ] Number of detected faces: 1 Elapsed time (s) 0.4334344859998964
    [VJ] Number of detected faces: 2 Elapsed time (s) 0.16609652999977698
    ```

    | Image | Distance | Image size | Processing time (s) |
    |---|---|---|----|
    | 1 | Selfie |  1808x3216 | 1.348 |
    | 1 | Portrait |  1200x1600 | 0.447 |
    | 1 | ~2 meters |  1741x1741 | 0.694 |

    And in gray mode:
    """)
    return


@app.cell
def _(images: list[Image], show_detected_faces_vj):
    _fig, _axes = plt.subplots(2, 2, figsize=(12, 9))
    _axes_flat = _axes.flatten()

    show_detected_faces_vj(images[0], ax=_axes_flat[0], thickness=20, gray=True)
    show_detected_faces_vj(images[1], ax=_axes_flat[1], thickness=10, gray=True)
    show_detected_faces_vj(images[2], ax=_axes_flat[2], thickness=10, gray=True)
    show_detected_faces_vj(images[3], ax=_axes_flat[3], thickness=10, gray=True)

    plt.show()
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ```
    [VJ] Number of detected faces: 1 Elapsed time (s) 0.14366232199972728
    [VJ] Number of detected faces: 4 Elapsed time (s) 0.12708506400304032
    [VJ] Number of detected faces: 1 Elapsed time (s) 0.37958746300137136
    [VJ] Number of detected faces: 2 Elapsed time (s) 0.16312288199696923
    ```

    Here's the table comparing RGB (high resolution) vs Grayscale processing times with speed improvements:

    | Image | Distance | RGB time (s) | Gray time (s) | Speed improvement (%) |
    |---|---|----|----|---|
    | 1 | Selfie | 1.348 | 0.905 | 32.8% ⚡ |
    | 2 | Portrait | 0.447 | 0.301 | 32.7% ⚡ |
    | 3 | ~2 meters | 0.694 | 0.471 | 32.1% |

    The grayscale mode consistently delivers approximately **33% speed improvement** across all high-resolution image sizes

    Now let's experiment with lower image resolution!
    """)
    return


@app.cell
def _(images: list[Image], show_detected_faces_vj):
    _fig, _axes = plt.subplots(2, 2, figsize=(12, 9))
    _axes_flat = _axes.flatten()

    show_detected_faces_vj(images[0], ax=_axes_flat[0], thickness=1, scale=0.03)
    show_detected_faces_vj(images[1], ax=_axes_flat[1], thickness=2, scale=0.14)
    show_detected_faces_vj(images[2], ax=_axes_flat[2], thickness=2, scale=0.158)
    show_detected_faces_vj(images[3], ax=_axes_flat[3], thickness=2, scale=0.45)

    plt.show()
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ```
    [VJ] scaled the image from 1808x3216 to 54x96
    [VJ] Number of detected faces: 1 Elapsed time (s) 0.0013493650003510993
    [VJ] scaled the image from 1200x1600 to 168x224
    [VJ] Number of detected faces: 1 Elapsed time (s) 0.007867862001148751
    [VJ] scaled the image from 1741x1741 to 275x275
    [VJ] Number of detected faces: 1 Elapsed time (s) 0.00982876799753285
    [VJ] scaled the image from 901x1600 to 405x720
    [VJ] Number of detected faces: 1 Elapsed time (s) 0.03792656099903979
    ```

    | Image | Distance | Compressed size | Processing time (s) | Times faster |
    |---|---|---|----|---|
    | 1 | Selfie |  54x96 | 0.002 | 674 🚀 |
    | 1 | Portrait |  114x152 | 0.005 | 89 🔥 |
    | 1 | ~2 meters |  275x275 | 0.018 | 38 |

    As the results show, the closer is the face, the more can we compress the image and the detector can do the job faster. Using image compression lead to huge speedup of face detection
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    Let's try to process the compressed pic using gray mode:
    """)
    return


@app.cell
def _(images: list[Image], show_detected_faces_vj):
    _fig, _axes = plt.subplots(2, 2, figsize=(12, 9))
    _axes_flat = _axes.flatten()

    show_detected_faces_vj(images[0], ax=_axes_flat[0], thickness=1, scale=0.03, gray=True)
    show_detected_faces_vj(images[1], ax=_axes_flat[1], thickness=1, scale=0.14, gray=True)
    show_detected_faces_vj(images[2], ax=_axes_flat[2], thickness=2, scale=0.158, gray=True)
    show_detected_faces_vj(images[3], ax=_axes_flat[3], thickness=2, scale=0.45, gray=True)

    plt.show()
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ```
    [VJ] scaled the image from 1808x3216 to 54x96
    [VJ] Number of detected faces: 1 Elapsed time (s) 0.0013295270000526216
    [VJ] scaled the image from 1200x1600 to 168x224
    [VJ] Number of detected faces: 1 Elapsed time (s) 0.015583098997012712
    [VJ] scaled the image from 1741x1741 to 275x275
    [VJ] Number of detected faces: 1 Elapsed time (s) 0.01837664000049699
    [VJ] scaled the image from 901x1600 to 405x720
    [VJ] Number of detected faces: 1 Elapsed time (s) 0.06837157599875354
    ```

    As we can see, the algorithm did manage to detect my face without colour information and with the same comprassion rate

    | Image | Distance | Compressed size | RGB time (s) | Gray time (s) | Speed improvement (%) |
    |---|---|---|----|----|---|
    | 1 | Selfie | 54x96 | 0.002 | 0.001 | 50% ⚡ |
    | 2 | Portrait | 114x152 | 0.005 | 0.003 | 40% ⚡ |
    | 3 | ~2 meters | 275x275 | 0.018 | 0.013 | 28% |

    The grayscale mode provides consistent speed improvements across all image sizes, with the most significant gains on smaller/compressed images (50% faster for the selfie).
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    # Detecting multiple faces
    """)
    return


@app.cell
def _(images: list[Image], show_detected_faces_vj):
    _fig, _axes = plt.subplot_mosaic([['a', 'b'], ['c', 'd'], ['e', 'e']], figsize=(15, 15))

    show_detected_faces_vj(images[4], ax=_axes['a'], thickness=14)
    show_detected_faces_vj(images[5], ax=_axes['b'], thickness=5)
    show_detected_faces_vj(images[7], ax=_axes['c'], thickness=10)
    show_detected_faces_vj(images[8], ax=_axes['d'], thickness=10)
    show_detected_faces_vj(images[6], ax=_axes['e'], thickness=10)

    plt.show()
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ```
    [VJ] Number of detected faces: 11 Elapsed time (s) 0.6229980169991904
    [VJ] Number of detected faces: 1 Elapsed time (s) 0.022243029001401737
    [VJ] Number of detected faces: 4 Elapsed time (s) 0.058211070998368086
    [VJ] Number of detected faces: 2 Elapsed time (s) 0.06021352500101784
    [VJ] Number of detected faces: 76 Elapsed time (s) 2.3250661239981127
    ```
    """)
    return


@app.cell
def _(images: list[Image], show_detected_faces_vj):
    _fig, _axes = plt.subplot_mosaic([['a', 'b'], ['c', 'd'], ['e', 'e']], figsize=(15, 15))

    show_detected_faces_vj(images[4], ax=_axes['a'], thickness=14, gray=True)
    show_detected_faces_vj(images[5], ax=_axes['b'], thickness=5, gray=True)
    show_detected_faces_vj(images[7], ax=_axes['c'], thickness=10, gray=True)
    show_detected_faces_vj(images[8], ax=_axes['d'], thickness=10, gray=True)
    show_detected_faces_vj(images[6], ax=_axes['e'], thickness=10, gray=True)

    plt.show()
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ```
    [VJ] Number of detected faces: 12 Elapsed time (s) 0.6256832250001025
    [VJ] Number of detected faces: 2 Elapsed time (s) 0.023428890999639407
    [VJ] Number of detected faces: 4 Elapsed time (s) 0.055134385998826474
    [VJ] Number of detected faces: 2 Elapsed time (s) 0.05295468899930711
    [VJ] Number of detected faces: 74 Elapsed time (s) 2.3168526890003704
    ```

    | Image | Faces | RGB time (s) | Gray time (s) | Speed improvement (%) |
    |---|---|----|----|---|
    | 1 | 9 | 2.132 | 1.438 | 32.6% ⚡ |
    | 2 | 201 | 1.86 | 1.26 | 32.3% ⚡ |

    We can see consistent **~32% speed improvement** with grayscale mode across both single-digit and multi-face scenarios, confirming that the number of detected faces doesn't significantly impact the RGB-to-grayscale performance gain ratio.

    Interestingly, switching to gray mode made the algorithm find a doll face on Sasha's T-shirt! (1 more than the RGB version), while the photo with sport fans has now 6 less faces detected.
    """)
    return


@app.cell
def _(images: list[Image], show_detected_faces_vj):
    _fig, _axes = plt.subplots(1, 3, figsize=(12, 9))

    show_detected_faces_vj(images[6], ax=_axes[0], thickness=12, scale=0.4)
    show_detected_faces_vj(images[6], ax=_axes[1], thickness=8, scale=0.33)
    show_detected_faces_vj(images[6], ax=_axes[2], thickness=5, scale=0.3)

    plt.show()
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ```
    [VJ] scaled the image from 6048x4024 to 2419x1610
    [VJ] Number of detected faces: 59 Elapsed time (s) 0.4481629290021374
    [VJ] scaled the image from 6048x4024 to 1996x1328
    [VJ] Number of detected faces: 49 Elapsed time (s) 0.32766186900335015
    [VJ] scaled the image from 6048x4024 to 1814x1207
    [VJ] Number of detected faces: 45 Elapsed time (s) 0.2968256599997403
    ```

    Here's the complete table showing precision loss across different scales:

    | Resolution | Detected faces | Detection loss | Processing time (s) | Speed improvement |
    |---|---|---|----|----|
    | 4080x2296 (original) | 9 | — | 2.132 | baseline |
    | 2040x1148 (0.5 scale) | 8 | 1 face (11% loss) | 0.541 | 3.9x faster ⚡ |
    | 408x230 (0.1 scale) | 2 | 7 faces (78% loss) ❌ | 0.022 | 96.8x faster 🚀 |

    **Key findings:**
    - **0.5 scale (50% compression)**: Minimal precision loss (1 face) with 3.9x speed improvement — good balance
    - **0.1 scale (10% compression)**: Massive speed gain (96.8x faster) but unacceptable precision loss (78%)
    """)
    return


@app.cell
def _(images: list[Image], show_detected_faces_vj):
    _fig, _axes = plt.subplots(1, 2, figsize=(12, 9))

    show_detected_faces_vj(images[7], ax=_axes[0], thickness=2, scale=0.3)
    show_detected_faces_vj(images[7], ax=_axes[1], thickness=2, scale=0.25)

    plt.show()
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ```
    [VJ] scaled the image from 1204x995 to 361x298
    [VJ] Number of detected faces: 4 Elapsed time (s) 0.011466265998024028
    [VJ] scaled the image from 1204x995 to 301x249
    [VJ] Number of detected faces: 3 Elapsed time (s) 0.009432504000869812
    ```

    Here's the table showing precision loss across different scales for the multi-face image:

    | Resolution | Scale | Detected faces | Detection loss | Processing time (s) | Speed improvement |
    |---|---|---|---|----|----|
    | 3504x2336 (original) | 1.0 | 201 | — | 1.86 | baseline |
    | 2453x1635 | 0.7 | 192 | 9 faces (4.5% loss) | 0.916 | 2.0x faster ⚡ |
    | 1752x1168 | 0.5 | 132 | 69 faces (34.3% loss) | 0.466 | 4.0x faster ⚡ |
    | 1402x934 | 0.4 | 17 | 184 faces (91.5% loss) ❌ | 0.297 | 6.3x faster 🚀 |

    **Key findings:**
    - **0.7 scale**: Excellent balance — minimal loss (4.5%) with 2x speed improvement
    - **0.5 scale**: Moderate compression — acceptable for many uses (34% loss, 4x faster)
    - **0.4 scale**: Severe precision degradation (91.5% loss) despite 6.3x speed gain
    """)
    return


if __name__ == "__main__":
    app.run()
