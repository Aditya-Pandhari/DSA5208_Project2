"""
Shared figure styling for the DSA5208 Project 2 report.

Design decisions (following a design-system-agnostic data-viz method):
  * Sequential magnitude  -> a single blue hue, light to dark. Rainfall is a
    magnitude, never a category, so it never gets a rainbow ramp.
  * Polarity (anomalies)  -> a two-hue diverging pair, red (deficit) <-> blue
    (surplus), with a NEUTRAL GREY midpoint so "normal" reads as "nothing".
  * Categorical identity  -> a fixed slot order, never cycled. The first three
    slots are validated for all-pairs colour-vision-deficiency separation
    (scatter/map forms); slot 4 is used only for adjacent forms (lines/bars).
  * All palettes were machine-validated (lightness band, chroma floor, CVD
    delta-E, normal-vision floor, surface contrast) rather than eyeballed.
  * Two slots sit below 3:1 contrast on the light surface, so the "relief rule"
    applies: every figure ships with its underlying CSV table and uses direct
    labels rather than relying on colour alone.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

# --- ink & chrome ---------------------------------------------------------
SURFACE = "#fcfcfb"
PLANE = "#f9f9f7"
INK = "#0b0b0b"
INK_2 = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
AXIS = "#c3c2b7"

# --- categorical slots (fixed order) --------------------------------------
CAT = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300"]
BLUE, ORANGE, AQUA, YELLOW, MAGENTA, GREEN = CAT
RED = "#d03b3b"

# --- sequential blue ramp (100 -> 700) ------------------------------------
BLUE_RAMP = [
    "#cde2fb", "#b7d3f6", "#9ec5f4", "#86b6ef", "#6da7ec", "#5598e7",
    "#3987e5", "#2a78d6", "#256abf", "#1c5cab", "#184f95", "#104281", "#0d366b",
]
CMAP_RAIN = LinearSegmentedColormap.from_list("rain_seq", ["#fcfcfb"] + BLUE_RAMP, N=256)

# --- diverging: red (deficit) <-> neutral grey <-> blue (surplus) ---------
CMAP_ANOM = LinearSegmentedColormap.from_list(
    "anom_div",
    ["#7d1d1d", "#a82c2c", "#d03b3b", "#e28080", "#f0c5c5",
     "#f0efec",
     "#c2daf6", "#86b6ef", "#3987e5", "#256abf", "#104281"],
    N=256,
)

# --- monsoon season palette (4 ordered seasons) ---------------------------
SEASON_ORDER = ["NE Monsoon (Dec-Mar)", "Inter-monsoon (Apr-May)",
                "SW Monsoon (Jun-Sep)", "Inter-monsoon (Oct-Nov)"]
SEASON_COLOR = {
    SEASON_ORDER[0]: BLUE,
    SEASON_ORDER[1]: ORANGE,
    SEASON_ORDER[2]: AQUA,
    SEASON_ORDER[3]: YELLOW,
}


def use_style() -> None:
    """Install the report-wide matplotlib defaults."""
    mpl.rcParams.update({
        "figure.facecolor": SURFACE,
        "axes.facecolor": SURFACE,
        "savefig.facecolor": SURFACE,
        "savefig.dpi": 300,
        "figure.dpi": 120,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.28,

        "font.family": "sans-serif",
        "font.sans-serif": ["Helvetica Neue", "Helvetica", "Arial", "DejaVu Sans"],
        "font.weight": "normal",
        "font.size": 9.5,
        "text.color": INK,

        # recessive chrome: only the axes we need, hairline grid
        "axes.edgecolor": AXIS,
        "axes.linewidth": 0.8,
        "axes.labelcolor": INK_2,
        "axes.labelsize": 9.5,
        "axes.labelpad": 7,
        "axes.titlesize": 11,
        "axes.titleweight": "bold",
        "axes.titlecolor": INK,
        "axes.titlelocation": "left",
        "axes.titlepad": 10,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "axes.axisbelow": True,

        "grid.color": GRID,
        "grid.linewidth": 0.7,
        "grid.alpha": 1.0,

        "xtick.color": MUTED,
        "ytick.color": MUTED,
        "xtick.labelcolor": INK_2,
        "ytick.labelcolor": INK_2,
        "xtick.labelsize": 8.5,
        "ytick.labelsize": 8.5,
        "xtick.major.size": 0,
        "ytick.major.size": 0,
        "xtick.major.pad": 5,
        "ytick.major.pad": 5,

        "legend.frameon": False,
        "legend.fontsize": 8.5,
        "legend.labelcolor": INK_2,
        "legend.handlelength": 1.5,
        "legend.handletextpad": 0.6,
        "legend.columnspacing": 1.4,

        "lines.linewidth": 2.0,
        "lines.solid_capstyle": "round",
        "lines.markersize": 4.5,

        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    })


def layout(fig, header: float = 1.55, footer: float = 0.60,
           left: float = 0.95, right: float = 0.32) -> None:
    """Reserve a header band and a footer band, measured in INCHES.

    Figure-fraction margins break as soon as the figure size changes; inches do
    not. Every figure calls this once, after its axes exist, so that the title
    block and the source note always have room that the axes cannot invade.
    """
    w, h = fig.get_figwidth(), fig.get_figheight()
    fig.subplots_adjust(top=1 - header / h, bottom=footer / h,
                        left=left / w, right=1 - right / w)


def titleblock(fig, title: str, subtitle: str = "", *,
               title_size: float = 15.0) -> None:
    """Left-aligned title + deck, positioned in the header band reserved by
    layout(). Offsets are in inches so they hold at any figure size."""
    w, h = fig.get_figwidth(), fig.get_figheight()
    x = 0.42 / w
    fig.text(x, 1 - 0.40 / h, title, ha="left", va="baseline",
             fontsize=title_size, fontweight="bold", color=INK)
    if subtitle:
        fig.text(x, 1 - 0.56 / h, subtitle, ha="left", va="top", fontsize=9.3,
                 color=INK_2, linespacing=1.5)


def sourcenote(fig, text: str) -> None:
    """Provenance line, muted, pinned to the foot of the figure."""
    w, h = fig.get_figwidth(), fig.get_figheight()
    fig.text(0.42 / w, 0.10 / h, text, ha="left", va="bottom", fontsize=7.6,
             color=MUTED, linespacing=1.55)


def despine(ax, left: bool = False, bottom: bool = False) -> None:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_visible(not left)
    ax.spines["bottom"].set_visible(not bottom)


_PDF = None        # open PdfPages, set by open_pdf()


def open_pdf(path: Path, title: str = "", subject: str = "") -> None:
    """Start collecting every subsequent save() into one multi-page PDF.

    Individual PNGs are still written per figure; the vector copies are
    gathered into a single document rather than scattered across one file
    per figure.
    """
    global _PDF
    from matplotlib.backends.backend_pdf import PdfPages

    path.parent.mkdir(parents=True, exist_ok=True)
    _PDF = PdfPages(str(path))
    meta = _PDF.infodict()
    if title:
        meta["Title"] = title
    if subject:
        meta["Subject"] = subject


def close_pdf() -> Path | None:
    """Finalise the multi-page PDF and report where it landed."""
    global _PDF
    if _PDF is None:
        return None
    path = Path(_PDF._file.fh.name) if hasattr(_PDF, "_file") else None
    _PDF.close()
    _PDF = None
    return path


def save(fig, path: Path) -> None:
    """Write the figure as a PNG, and append it to the open PDF if there is one."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path)
    if _PDF is not None:
        _PDF.savefig(fig, bbox_inches="tight", pad_inches=0.28)
    plt.close(fig)
    print(f"  [figure] {path.name}")
