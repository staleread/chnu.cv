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
    from time import perf_counter_ns
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
        "lighter",
        "cave",
        "group",
        "masks-theatre",
        "fans",  # or "run"
        "beatles",
        "drawing",
        "statue",
        "coin",
        ])
    return (images,)


@app.cell
def _(images: list[Image]):
    # Plotting images in a 4x3 grid

    _fig, _axes = plt.subplots(3, 4, figsize=(12, 9))

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
        start_ns = perf_counter_ns()
        rects = detector(result, 1)
        end_ns = perf_counter_ns()

        elapsed_time = round((end_ns - start_ns) / 1e9, 3)
    
        print('[DLIB] Number of detected faces:', len(rects), 'Elapsed time (s)', elapsed_time)

        # Draw rectangles
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
    _fig, _axes = plt.subplots(1, 3, figsize=(12, 9))

    show_detected_faces_dlib(images[0], ax=_axes[0], thickness=20)
    show_detected_faces_dlib(images[1], ax=_axes[1], thickness=10)
    show_detected_faces_dlib(images[2], ax=_axes[2], thickness=10)

    plt.show()
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
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
    _fig, _axes = plt.subplots(1, 3, figsize=(12, 9))

    show_detected_faces_dlib(images[0], ax=_axes[0], thickness=20, gray=True)
    show_detected_faces_dlib(images[1], ax=_axes[1], thickness=10, gray=True)
    show_detected_faces_dlib(images[2], ax=_axes[2], thickness=10, gray=True)

    plt.show()
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
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
    _fig, _axes = plt.subplots(1, 3, figsize=(12, 9))
    _scale = 0.05

    show_detected_faces_dlib(images[0], ax=_axes[0], thickness=1, scale=0.03)
    show_detected_faces_dlib(images[1], ax=_axes[1], thickness=1, scale=0.095)
    show_detected_faces_dlib(images[2], ax=_axes[2], thickness=2, scale=0.158)

    plt.show()
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
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
    _fig, _axes = plt.subplots(1, 3, figsize=(12, 9))
    _scale = 0.05

    show_detected_faces_dlib(images[0], ax=_axes[0], thickness=1, scale=0.03, gray=True)
    show_detected_faces_dlib(images[1], ax=_axes[1], thickness=1, scale=0.095, gray=True)
    show_detected_faces_dlib(images[2], ax=_axes[2], thickness=2, scale=0.158, gray=True)

    plt.show()
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
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
    _fig, _axes = plt.subplots(1, 2, figsize=(12, 9))

    show_detected_faces_dlib(images[5], ax=_axes[0], thickness=12)
    show_detected_faces_dlib(images[7], ax=_axes[1], thickness=10)

    plt.show()
    return


@app.cell
def _(images: list[Image], show_detected_faces_dlib):
    _fig, _axes = plt.subplots(1, 2, figsize=(12, 9))

    show_detected_faces_dlib(images[5], ax=_axes[0], thickness=12, gray=True)
    show_detected_faces_dlib(images[7], ax=_axes[1], thickness=10, gray=True)

    plt.show()
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    | Image | Faces | RGB time (s) | Gray time (s) | Speed improvement (%) |
    |---|---|----|----|---|
    | 1 | 9 | 2.132 | 1.438 | 32.6% ⚡ |
    | 2 | 201 | 1.86 | 1.26 | 32.3% ⚡ |

    We can see consistent **~32% speed improvement** with grayscale mode across both single-digit and multi-face scenarios, confirming that the number of detected faces doesn't significantly impact the RGB-to-grayscale performance gain ratio.

    Interestingly, switching to gray mode made the algorithm find a doll face on Sasha's T-shirt! (1 more than the RGB version), while the photo with sport fans has now 6 less faces detected.
    """)
    return


@app.cell
def _(images: list[Image], show_detected_faces_dlib):
    _fig, _axes = plt.subplots(1, 2, figsize=(12, 9))

    show_detected_faces_dlib(images[5], ax=_axes[0], thickness=12, scale=0.5)
    show_detected_faces_dlib(images[5], ax=_axes[1], thickness=2, scale=0.1)

    plt.show()
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
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

    show_detected_faces_dlib(images[7], ax=_axes[0], thickness=12, scale=0.7)
    show_detected_faces_dlib(images[7], ax=_axes[1], thickness=12, scale=0.5)
    show_detected_faces_dlib(images[7], ax=_axes[2], thickness=2, scale=0.4)

    plt.show()
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
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
