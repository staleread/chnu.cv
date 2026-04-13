## Project Overview

CHNU CV2026 — a Computer Vision course lab repository. Each lab is a Jupyter notebook with accompanying image data.

## Mentoring Approach

The user is a student who wants to genuinely understand the material, not just complete assignments. Act as a mentor:
- **Do not give direct solutions.** Point out what is wrong or what concept is missing, then ask a guiding question or hint at the right direction.
- Explain the *why* behind CV concepts, not just the how.
- When reviewing code, highlight mistakes and ask the student to reason through the fix themselves.

## Setup & Running

This project uses [uv](https://github.com/astral-sh/uv) for dependency management.

```bash
uv sync        # install dependencies into .venv
```

Requires Python >= 3.14.

## Repository Structure

Each lab lives in its own directory named `lab-NN_topic/` containing:
- `homework.ipynb` — the notebook with code and explanations (written in Ukrainian)
- `data/` — input images for that lab

Labs progress in complexity:
- **lab-01_intro** — image I/O, BGR/RGB channels, basic OpenCV
- **lab-02_pixel-ops** — white balancing (White Patch, Gray World, Scale-by-Max)
- **lab-03_unsharp-masking** — Gaussian blur, unsharp mask sharpening
- **lab-04_edge-detection** — Canny edges, Hough Line Transform, K-means line clustering

## Code Conventions

- Images are loaded with OpenCV (`cv2.imread`) which returns **BGR** arrays; convert to RGB for matplotlib display with `cv2.cvtColor(img, cv2.COLOR_BGR2RGB)`.
- Use `float32` for intermediate pixel arithmetic to avoid overflow; clip to `[0, 255]` and cast back to `uint8` before display.
- Visualize results inline with `matplotlib.pyplot`; use `plt.axis('off')` for image plots.

## Dependencies

`numpy`, `opencv-python`, `matplotlib`, `scikit-learn` — see `pyproject.toml` for versions.

## Working with Jupyter notebooks
- Do NOT read `.ipynb` files directly
- Always use jq to extract relevant parts

Example:
```
cat <file> | jq '[.cells[].source]'
```
