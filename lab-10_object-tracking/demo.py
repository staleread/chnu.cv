import marimo

__generated_with = "0.23.1"
app = marimo.App(width="medium", auto_download=["ipynb"])

with app.setup:
    import marimo as mo  # comment this if not using Marimo notebook editor
    import cv2
    import os
    import json
    import tempfile
    import imageio
    import matplotlib.pyplot as plt


@app.cell
def _():
    # Profile management
    # The lab requires the analysis of different trackers on different
    # scenarios like object scaling, background or lightning changes.
    # In order to be make the setup persistent a JSON file is used as
    # a storage. The state management of marimo makes the "profile" creation
    # easy with possibly many setups for a single video by using a unique label

    config_path = "config.json"

    def load_profiles():
        import os
        import json

        if not os.path.exists(config_path):
            return {}
        try:
            with open(config_path, "r") as f:
                return json.load(f)
        except Exception:
            pass

    get_profiles, set_profiles = mo.state(load_profiles())
    get_active_profile, set_active_profile = mo.state(None)
    get_config_override, set_config_override = mo.state({})
    return (
        config_path,
        get_active_profile,
        get_config_override,
        get_profiles,
        set_active_profile,
        set_config_override,
        set_profiles,
    )


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ### Profile Management

    Save and load your tracking configurations:
    - **Select Profile**: Choose an existing configuration to instantly restore all settings.
    - **Profile Label**: Add a custom tag to your current setup before saving.
    - **Save Profile**: Click to store your current video, range, and ROI settings.

    *Note: Manually changing any setting will switch the profile back to "Custom".*
    """)
    return


@app.cell
def _(
    get_active_profile,
    get_profiles,
    set_active_profile,
    set_config_override,
):
    def on_profile_select(v):
        if v and v != "Custom":
            set_active_profile(v)
            set_config_override(get_profiles()[v])
        else:
            set_active_profile(None)

    _options = ["Custom"] + list(get_profiles().keys())
    _current_val = get_active_profile() or "Custom"

    if _current_val not in _options:
        _current_val = "Custom"

    profile_dropdown = mo.ui.dropdown(
        options=_options,
        value=_current_val,
        label="Select Profile",
        on_change=on_profile_select,
    )

    profile_label = mo.ui.text(label="Profile Label")

    mo.hstack([profile_dropdown, profile_label], justify="start")
    return (profile_label,)


@app.function
def get_tracker(tracker_type: str):
    if tracker_type == "MIL":
        return cv2.TrackerMIL.create()
    if tracker_type == "TLD":
        return cv2.legacy.TrackerTLD.create()
    return None


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    # CV Lab 10 — Object Tracking Demo

    This notebook demonstrates object tracking using OpenCV trackers.

    1. **Select a video** and **tracker**.
    2. **Configure the region of interest** using sliders.
    3. **Run tracking** to see the result.
    """)
    return


@app.cell
def _(get_config_override, set_active_profile):
    # Get video file paths
    _video_files = [
        f"data/{file}" for file in os.listdir("data") if file.endswith((".mp4", ".gif"))
    ]

    _default_video = get_config_override().get("video")
    if _default_video not in _video_files:
        _default_video = _video_files[0] if _video_files else None

    video_select = mo.ui.dropdown(
        options=_video_files,
        value=_default_video,
        label="Select Video",
        on_change=lambda _: set_active_profile(None),
    )

    video_select
    return (video_select,)


@app.cell
def _(video_select):
    mo.stop(video_select.value is None)

    _cap = cv2.VideoCapture(video_select.value)
    total_frames: int = int(_cap.get(cv2.CAP_PROP_FRAME_COUNT))
    _cap.release()
    return (total_frames,)


@app.cell
def _(get_config_override, set_active_profile, total_frames: int):
    tracker_select = mo.ui.dropdown(
        options=["MIL", "TLD"],
        value=get_config_override().get("tracker", "MIL"),
        label="Select Tracker",
        on_change=lambda _: set_active_profile(None),
    )

    _default_range = get_config_override().get(
        "frame_range", [0, min(150, total_frames - 1)]
    )
    # Validate range
    if not (
        isinstance(_default_range, list)
        and len(_default_range) == 2
        and _default_range[0] < total_frames
    ):
        _default_range = [0, min(150, total_frames - 1)]

    frame_range = mo.ui.range_slider(
        start=0,
        stop=max(0, total_frames - 1),
        step=1,
        value=_default_range,
        label="Frame Range",
        full_width=True,
        on_change=lambda _: set_active_profile(None),
    )

    mo.vstack([tracker_select, frame_range])
    return frame_range, tracker_select


@app.cell
def _(frame_range, video_select):
    mo.stop(video_select.value is None)

    _cap = cv2.VideoCapture(video_select.value)
    _cap.set(cv2.CAP_PROP_POS_FRAMES, frame_range.value[0])
    _ret, first_frame = _cap.read()
    _cap.release()

    if not _ret:
        mo.md("Failed to load video.")
        mo.stop(True)
    return (first_frame,)


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    Configure the region of interest (ROI) using sliders
    """)
    return


@app.cell
def _(first_frame, get_config_override, set_active_profile):
    h, w = first_frame.shape[:2]
    _roi = get_config_override().get("roi", {})

    def on_roi_change(_):
        set_active_profile(None)

    # Sliders for ROI selection
    x_slider = mo.ui.slider(
        start=0,
        stop=w - 1,
        step=1,
        value=_roi.get("x", w // 4),
        label="X",
        full_width=True,
        on_change=on_roi_change,
    )
    y_slider = mo.ui.slider(
        start=0,
        stop=h - 1,
        step=1,
        value=_roi.get("y", h // 4),
        label="Y",
        full_width=True,
        on_change=on_roi_change,
    )
    w_slider = mo.ui.slider(
        start=1,
        stop=w,
        step=1,
        value=_roi.get("w", w // 2),
        label="Width",
        full_width=True,
        on_change=on_roi_change,
    )
    h_slider = mo.ui.slider(
        start=1,
        stop=h,
        step=1,
        value=_roi.get("h", h // 2),
        label="Height",
        full_width=True,
        on_change=on_roi_change,
    )

    mo.vstack([x_slider, y_slider, w_slider, h_slider])
    return h_slider, w_slider, x_slider, y_slider


@app.cell
def _(h_slider, w_slider, x_slider, y_slider):
    # Get slider values
    x, y, bw, bh = x_slider.value, y_slider.value, w_slider.value, h_slider.value
    return bh, bw, x, y


@app.cell
def _(bh, bw, first_frame, x, y):
    # Preview ROI selection

    _preview = first_frame.copy()

    cv2.rectangle(_preview, (x, y), (x + bw, y + bh), (0, 255, 0), 2)

    plt.figure(figsize=(8, 5))
    plt.imshow(cv2.cvtColor(_preview, cv2.COLOR_BGR2RGB))
    plt.axis("off")
    plt.title("ROI Preview")
    return


@app.cell
def _(get_active_profile):
    # UI button for starting tracking
    run_button = mo.ui.run_button(label="Start Tracking")
    save_button = mo.ui.run_button(
        label="Save Profile", disabled=(get_active_profile() is not None)
    )

    mo.hstack([run_button, save_button])
    return run_button, save_button


@app.cell
def _(
    bh,
    bw,
    config_path,
    frame_range,
    get_profiles,
    profile_label,
    save_button,
    set_active_profile,
    set_profiles,
    tracker_select,
    video_select,
    x,
    y,
):
    if save_button.value:
        # Construct profile name
        _base = os.path.basename(video_select.value).replace(".", "_")
        _label = profile_label.value.strip()
        _name = f"{_base}_{_label}" if _label else _base

        _new_profile = {
            "video": video_select.value,
            "tracker": tracker_select.value,
            "frame_range": frame_range.value,
            "roi": {"x": x, "y": y, "w": bw, "h": bh},
        }

        _profiles = get_profiles()
        _profiles[_name] = _new_profile

        with open(config_path, "w") as _f:
            json.dump(_profiles, _f, indent=4)

        set_profiles(_profiles)
        set_active_profile(_name)
    return


@app.cell
def _(
    bh,
    bw,
    first_frame,
    frame_range,
    run_button,
    tracker_select,
    video_select,
    x,
    y,
):
    mo.stop(not run_button.value)

    _tracker = get_tracker(tracker_select.value)
    _tracker.init(first_frame, (x, y, bw, bh))

    _cap = cv2.VideoCapture(video_select.value)
    _start, _end = frame_range.value
    _cap.set(cv2.CAP_PROP_POS_FRAMES, _start)
    _max_frames = _end - _start + 1

    _frames = []

    # Process each frame in the selected range
    for _ in mo.status.progress_bar(range(int(_max_frames)), title="Tracking..."):
        _success, _frame = _cap.read()
        if not _success:
            break

        _success, _trk_bbox = _tracker.update(_frame)

        if _success:
            # Draw the object rectange with label
            (_bx, _by, _btw, _bth) = [int(_v) for _v in _trk_bbox]
            cv2.rectangle(_frame, (_bx, _by), (_bx + _btw, _by + _bth), (0, 255, 0), 2)
            cv2.putText(
                _frame,
                tracker_select.value,
                (_bx, _by - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 255, 0),
                2,
            )
        else:
            cv2.putText(
                _frame,
                "Tracking failure",
                (50, 50),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.75,
                (0, 0, 255),
                2,
            )

        # Ensure dimensions are even for FFMPEG (libx264 yuv420p)
        _h, _w = _frame.shape[:2]
        if _h % 2 != 0 or _w % 2 != 0:
            _frame = _frame[: _h - (_h % 2), : _w - (_w % 2)]

        _frames.append(cv2.cvtColor(_frame, cv2.COLOR_BGR2RGB))

    _cap.release()

    tracking_video = None

    if _frames:
        # Save frames with rectanges to a temporary file so we can show it with marimo
        with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as _tmp:
            imageio.mimsave(
                _tmp.name,
                _frames,
                fps=30,
                format="FFMPEG",
                codec="libx264",
                macro_block_size=None,
            )
            tracking_video = mo.video(_tmp.name)
    return (tracking_video,)


@app.cell
def _(tracking_video):
    mo.stop(tracking_video is None)
    mo.vstack([mo.md("### Tracking Result"), tracking_video])
    return


if __name__ == "__main__":
    app.run()
