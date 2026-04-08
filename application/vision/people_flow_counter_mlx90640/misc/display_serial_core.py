import io
import struct
import time
import uuid

import matplotlib
matplotlib.use("TkAgg")

import matplotlib.patches as patches
import matplotlib.pyplot as plt

import imageio
import numpy as np
from scipy.ndimage import gaussian_filter, zoom

matplotlib.use("TkAgg")  # explicit backend – avoids focus-stealing on some OSes

# ===================== PARAMETERS =====================
# Sensor resolution
IMG_W = 32
IMG_H = 24
IMG_DEPTH = 1

# Display
CMAP = "inferno"
VMIN = 25
VMAX = 32
FIG_SIZE = (8, 6)
WINDOW_TITLE = "People Flow Counter"

# Crossing line (x position in sensor pixels)
CROSSING_LINE_X = 20

# Overlay styling
BBOX_COLOR = "lime"
BBOX_LINEWIDTH = 1.5
CENTROID_RADIUS = 0.5
ARROW_COLOR = "white"
ARROW_MIN_DIST = 1.0

# Default smoothing
DEFAULT_SIGMA = 0.5

# Upscale factor applied to image data before display (1 = native sensor resolution)
# Smoothing (sigma) is applied before upscaling, so both are visible.
DISPLAY_SCALE = 4

# Zoom interpolation order: 0=nearest, 1=bilinear, 2=quadratic, 3=cubic, 4/5=higher
DISPLAY_INTERP_ORDER = 3

# ===================== CONSTANTS =====================
types = ["UINT8", "FLOAT"]
type_sizes = [1, 4]

HEADER_MARKER = b"image:image,"


def add_args(parser):
    parser.add_argument(
        "--save",
        action="store_true",
        help="Record frames into a .csv file for gathering data.",
    )
    parser.add_argument(
        "--animate",
        action="store_true",
        help="Record frames into a .mp4 file for demo videos.",
    )
    parser.add_argument(
        "--sigma",
        type=float,
        default=DEFAULT_SIGMA,
        help=f"Gaussian smoothing sigma (0 = disabled). Default: {DEFAULT_SIGMA}",
    )


def figure_to_img(fig):
    io_buf = io.BytesIO()
    fig.savefig(io_buf, format="raw")
    io_buf.seek(0)
    w, h = fig.canvas.get_width_height()
    fig_img = np.frombuffer(io_buf.getvalue(), dtype=np.uint8).reshape(
        int(h), int(w), -1
    )
    io_buf.close()
    return fig_img


def data_to_value(data, type_idx):
    if types[type_idx] == "UINT8":
        return struct.unpack("B" * (len(data) // type_sizes[type_idx]), data)
    if types[type_idx] == "FLOAT":
        return struct.unpack("f" * (len(data) // type_sizes[type_idx]), data)
    raise ValueError(f"Unsupported type index: {type_idx}")


def to_image(data, w, h, d, type_idx):
    v = data_to_value(data, type_idx)
    img = np.array(v).reshape(h, w, d)
    return img


def is_printable_ascii(s):
    return all(32 <= ord(c) < 127 for c in s)


def validate_header(line):
    line_info = line.split(",")
    if len(line_info) < 5:
        return None
    try:
        w = int(line_info[1])
        h = int(line_info[2])
        d = int(line_info[3])
        type_idx = int(line_info[4])
    except (ValueError, IndexError):
        return None
    if w <= 0 or w > 256 or h <= 0 or h > 256:
        return None
    if d <= 0 or d > 4:
        return None
    if type_idx < 0 or type_idx >= len(types):
        return None
    misc_info = ",".join(line_info[5:])
    if not is_printable_ascii(misc_info):
        return None
    return w, h, d, type_idx, misc_info


def scan_for_header(ser):
    marker_len = len(HEADER_MARKER)
    for _ in range(50):
        ring = bytearray(marker_len)
        found = False
        for _ in range(100000):
            ch = ser.read(1)
            if not ch:
                return None
            ring.append(ch[0])
            if len(ring) > marker_len:
                ring = ring[-marker_len:]
            if ring == HEADER_MARKER:
                found = True
                break
        if not found:
            return None
        rest = b""
        line_too_long = False
        while True:
            ch = ser.read(1)
            if not ch:
                return None
            if ch == b"\n":
                break
            rest += ch
            if len(rest) > 200:
                line_too_long = True
                break
        if line_too_long:
            continue
        try:
            line = "image:image," + rest.decode("utf-8").strip()
        except UnicodeDecodeError:
            continue
        result = validate_header(line)
        if result is None:
            continue
        print(line)
        return result
    return None


def wait_for_image(ser):
    last_heartbeat = time.monotonic()
    resync_count = 0
    while True:
        result = scan_for_header(ser)
        if result is None:
            now = time.monotonic()
            if now - last_heartbeat >= 5.0:
                print("  (waiting for image data...)")
                last_heartbeat = now
            try:
                plt.gcf().canvas.flush_events()
            except Exception as exc:
                print(f"WARNING: flush_events failed: {exc}")
            continue

        w, h, d, type_idx, misc_info = result
        expected_size = w * h * d * type_sizes[type_idx]
        img_data = ser.read(expected_size)
        if len(img_data) < expected_size:
            resync_count += 1
            print(
                "WARNING: incomplete image data "
                f"({len(img_data)}/{expected_size} bytes), resyncing... "
                f"(#{resync_count})"
            )
            continue
        img = to_image(img_data, w, h, d, type_idx)
        return img, misc_info


def wait_for_bboxes(ser):
    for _ in range(50):
        raw_line = ser.readline()
        if not raw_line:
            continue
        try:
            line = raw_line.decode("utf-8").strip()
        except UnicodeDecodeError:
            continue
        if not line:
            continue
        print(line)
        if line.startswith("image:"):
            print(
                "WARNING: desync in wait_for_bboxes (got image header), skipping frame"
            )
            return None
        if not line.startswith("bboxes"):
            continue

        line_info = line.split(":")
        try:
            num_bboxes = int(line_info[1])
        except (IndexError, ValueError):
            return None

        bboxes = []
        for i in range(num_bboxes):
            raw = ser.readline()
            if not raw:
                print(f"WARNING: timeout waiting for bbox {i + 1}/{num_bboxes}")
                return None
            try:
                bbox_line = raw.decode("utf-8").strip()
            except UnicodeDecodeError:
                return None

            print(bbox_line)
            if bbox_line.startswith("image:"):
                print(f"WARNING: desync mid-bbox read ({i + 1}/{num_bboxes})")
                return None
            if "," not in bbox_line:
                return None

            bbox_info = bbox_line.split(",")
            if len(bbox_info) < 5:
                return None

            try:
                bbox = np.zeros(shape=(5,))
                bbox[0] = float(bbox_info[0])
                bbox[1] = float(bbox_info[1])
                bbox[2] = float(bbox_info[2])
                bbox[3] = float(bbox_info[3])
                bbox[4] = float(bbox_info[4])
            except ValueError:
                return None

            bboxes.append(bbox)
        return bboxes
    return None


def wait_for_centroids(ser):
    for _ in range(50):
        raw_line = ser.readline()
        if not raw_line:
            continue
        try:
            line = raw_line.decode("utf-8").strip()
        except UnicodeDecodeError:
            continue
        if not line:
            continue

        print(line)
        if line.startswith("image:"):
            print(
                "WARNING: desync in wait_for_centroids (got image header), skipping frame"
            )
            return None
        if not line.startswith("centroids"):
            continue

        line_info = line.split(":")
        try:
            num_centroids = int(line_info[1])
        except (IndexError, ValueError):
            return None

        centroids = []
        for i in range(num_centroids):
            raw = ser.readline()
            if not raw:
                print(f"WARNING: timeout waiting for centroid {i + 1}/{num_centroids}")
                return None
            try:
                centroid_line = raw.decode("utf-8").strip()
            except UnicodeDecodeError:
                return None

            print(centroid_line)
            if centroid_line.startswith("image:"):
                print(f"WARNING: desync mid-centroid read ({i + 1}/{num_centroids})")
                return None
            if "," not in centroid_line:
                return None

            centroid_info_prev = None
            if "-" in centroid_line:
                track_info = centroid_line.split("-")
                centroid_info = track_info[0].split(",")
                centroid_info_prev = track_info[1].split(",")
            else:
                centroid_info = centroid_line.split(",")

            if len(centroid_info) < 3:
                return None

            try:
                res = []
                x = float(centroid_info[0])
                y = float(centroid_info[1])
                count = int(centroid_info[2])
                res.extend([x, y, count])

                if centroid_info_prev is not None and len(centroid_info_prev) >= 4:
                    x_prev = float(centroid_info_prev[0])
                    y_prev = float(centroid_info_prev[1])
                    count_prev = int(centroid_info_prev[2])
                    dist_squared = float(centroid_info_prev[3])
                    res.extend([x_prev, y_prev, count_prev, dist_squared])
            except ValueError:
                return None

            centroids.append(res)
        return centroids
    return None


def display_serial(ser, args):
    sigma = getattr(args, "sigma", DEFAULT_SIGMA)

    plt.ion()
    fig, ax = plt.subplots(1, 1, figsize=FIG_SIZE)
    fig.canvas.manager.set_window_title(WINDOW_TITLE)

    # Persistent artists — created once, updated each frame
    dummy = np.zeros((IMG_H * DISPLAY_SCALE, IMG_W * DISPLAY_SCALE))
    im = ax.imshow(
        dummy,
        cmap=CMAP,
        vmin=VMIN,
        vmax=VMAX,
        interpolation="nearest",  # no matplotlib smoothing — sigma + zoom control this
        extent=(0, IMG_W, IMG_H, 0),
        aspect="auto",
    )
    ax.axvline(
        CROSSING_LINE_X,
        color="cyan",
        linewidth=1.5,
        linestyle="--",
        alpha=0.7,
    )
    title_text = ax.set_title("Waiting for data...", fontsize=11)
    fps_text = ax.text(
        0.01,
        0.97,
        "",
        transform=ax.transAxes,
        fontsize=9,
        color="white",
        fontweight="bold",
        verticalalignment="top",
        bbox=dict(boxstyle="round,pad=0.3", facecolor="black", alpha=0.6),
    )
    fig.tight_layout()
    fig.canvas.draw()
    fig.canvas.flush_events()

    # State for overlays — we keep a list so we can remove them each frame
    overlay_artists = []

    writer = None
    if args.animate:
        writer = imageio.get_writer("animation.mp4", fps=1)

    timings = []
    frames_ok = 0
    frames_dropped = 0
    sample_count = 0
    sample_name = str(uuid.uuid4())
    save_file = None
    if args.save:
        save_file = open(f"data/{sample_name}.csv", "a", encoding="utf-8")

    try:
        while True:
            t_start = time.time()

            img, img_misc = wait_for_image(ser)

            bboxes = wait_for_bboxes(ser)
            if bboxes is None:
                frames_dropped += 1
                continue

            centroids = wait_for_centroids(ser)
            if centroids is None:
                frames_dropped += 1
                continue

            frames_ok += 1

            # ---- remove previous overlay patches ----
            for artist in overlay_artists:
                artist.remove()
            overlay_artists.clear()

            # ---- update image data (no full redraw!) ----
            # Pipeline: squeeze → smooth (sigma) → upscale (zoom) → display
            display_img = img.squeeze().astype(np.float32)
            if sigma > 0:
                display_img = gaussian_filter(display_img, sigma=sigma)
            if DISPLAY_SCALE > 1:
                display_img = zoom(
                    display_img, DISPLAY_SCALE, order=DISPLAY_INTERP_ORDER
                )
            im.set_data(display_img)

            # ---- title ----
            title_text.set_text(img_misc)

            # ---- FPS / stats overlay ----
            if timings:
                avg_dt = np.mean(timings[-30:])
                fps_str = (
                    f"FPS: {1.0 / avg_dt:.1f}  |  "
                    f"Latency: {avg_dt * 1000:.0f} ms  |  "
                    f"OK: {frames_ok}  Drop: {frames_dropped}"
                )
            else:
                fps_str = f"OK: {frames_ok}  Drop: {frames_dropped}"
            fps_text.set_text(fps_str)

            # ---- bounding boxes ----
            for bbox in bboxes:
                rect = patches.Rectangle(
                    (bbox[0], bbox[1]),
                    bbox[2],
                    bbox[3],
                    edgecolor=BBOX_COLOR,
                    facecolor="none",
                    linewidth=BBOX_LINEWIDTH,
                )
                ax.add_patch(rect)
                overlay_artists.append(rect)

                txt = ax.annotate(
                    f"{bbox[4]:.2f}",
                    (bbox[0], bbox[1]),
                    ha="left",
                    va="bottom",
                    color=BBOX_COLOR,
                    fontsize=10,
                    fontweight="bold",
                )
                overlay_artists.append(txt)

            # ---- centroids + motion arrows ----
            for centroid in centroids:
                x, y = centroid[0], centroid[1]
                circle = patches.Circle(
                    (x, y),
                    radius=CENTROID_RADIUS,
                    facecolor="white",
                    edgecolor="black",
                    linewidth=1.2,
                    zorder=5,
                )
                ax.add_patch(circle)
                overlay_artists.append(circle)

                if len(centroid) > 3:
                    x_prev, y_prev = centroid[3], centroid[4]
                    dist_sq = centroid[6]
                    if np.sqrt(dist_sq) >= ARROW_MIN_DIST:
                        arrow = patches.FancyArrowPatch(
                            (x_prev, y_prev),
                            (x, y),
                            arrowstyle="-|>",
                            mutation_scale=12,
                            color=ARROW_COLOR,
                            linewidth=1.5,
                            zorder=4,
                        )
                        ax.add_patch(arrow)
                        overlay_artists.append(arrow)

            # ---- efficient redraw (blit-style) ----
            fig.canvas.draw_idle()
            fig.canvas.flush_events()

            # ---- recording ----
            if args.animate and writer is not None:
                fig_img = figure_to_img(fig)
                writer.append_data(fig_img)

            if args.save and save_file is not None:
                save_string = (
                    f"{sample_count},{','.join(map(str, img.flatten().tolist()))}\n"
                )
                save_file.write(save_string)

            sample_count += 1
            dt = time.time() - t_start
            timings.append(dt)

            print(
                f"({img.shape[0]}, {img.shape[1]}, {img.shape[2]})  "
                f"[ok:{frames_ok} dropped:{frames_dropped}  {1.0 / dt:.1f} fps]"
            )

    except KeyboardInterrupt:
        print(f"\nStopped. Frames OK: {frames_ok}, Dropped: {frames_dropped}")

    finally:
        if save_file is not None:
            save_file.close()

        if args.animate and writer is not None and timings:
            delay = np.mean(timings)
            print(f"Saving animation with FPS {1 / delay:.3f}s")
            writer.close()
            gif = imageio.mimread("animation.mp4", memtest=False)
            imageio.mimwrite("animation.mp4", gif, fps=1 / delay)
        elif args.animate and writer is not None:
            writer.close()
