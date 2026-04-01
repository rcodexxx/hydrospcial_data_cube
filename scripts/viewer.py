# scripts/viewer.py
"""
Hydrospatial Data Cube Interactive Viewer

Usage:
    panel serve scripts/viewer.py --show

Features:
    - Layer selection and overlay
    - Click on map to query all layer values at that point
    - Adjustable colormap range
    - Region statistics
"""
import numpy as np
import panel as pn
import holoviews as hv
import xarray as xr
from holoviews import opts, streams
from pathlib import Path

hv.extension("bokeh")
pn.extension(sizing_mode="stretch_width")

# ── Load Data ────────────────────────────────────────────────
NC_PATH = Path(__file__).parent.parent / "outputs/hydrospatial_datacube.nc"
ds = xr.open_dataset(NC_PATH)

# build layer info
LAYER_INFO = {}
CATEGORICAL = {"sediment_class"}

for var in ds.data_vars:
    data = ds[var].values
    if np.all(np.isnan(data)):
        continue
    long_name = ds[var].attrs.get("long_name", var)
    units = ds[var].attrs.get("units", "")
    label = f"{long_name} ({units})" if units else long_name
    LAYER_INFO[var] = {
        "label": label,
        "units": units,
        "long_name": long_name,
        "vmin": float(np.nanpercentile(data, 2)),
        "vmax": float(np.nanpercentile(data, 98)),
    }

LAYER_NAMES = list(LAYER_INFO.keys())
COLORMAPS = ["turbo", "viridis", "inferno", "RdBu_r", "YlOrRd",
             "Blues_r", "Greys_r", "RdYlGn_r", "coolwarm"]

# sediment labels
SEDIMENT_LABELS = [
    "Coarse sand", "Fine sand", "Very fine sand", "Silty sand",
    "Sandy silt", "Silt", "Sandy-silt-clay", "Silty clay",
    "Clayey silt", "Framework-supported mud", "Fluid mud",
]


# ── Widgets ──────────────────────────────────────────────────
layer_select = pn.widgets.Select(
    name="Layer",
    options={LAYER_INFO[k]["label"]: k for k in LAYER_NAMES},
    value=LAYER_NAMES[0],
    width=280,
)

cmap_select = pn.widgets.Select(
    name="Colormap",
    options=COLORMAPS,
    value="turbo",
    width=280,
)

range_slider = pn.widgets.RangeSlider(
    name="Color Range",
    start=-1000, end=1000,
    value=(-100, 100),
    step=0.1,
    width=280,
)

opacity_slider = pn.widgets.FloatSlider(
    name="Opacity",
    start=0.0, end=1.0, value=1.0, step=0.05,
    width=280,
)

# point query display
query_pane = pn.pane.Markdown(
    "### Point Query\nClick on the map to query all layers.",
    width=300,
    styles={"font-size": "12px"},
)

# region stats display
stats_pane = pn.pane.Markdown(
    "### Region Statistics\nUse Box Select tool on the map.",
    width=300,
    styles={"font-size": "12px"},
)


def update_range(event):
    var = layer_select.value
    info = LAYER_INFO[var]
    range_slider.start = float(np.nanmin(ds[var].values))
    range_slider.end = float(np.nanmax(ds[var].values))
    range_slider.value = (info["vmin"], info["vmax"])
    range_slider.step = (info["vmax"] - info["vmin"]) / 200

layer_select.param.watch(update_range, "value")
# initialize
update_range(None)


# ── Map Plot ─────────────────────────────────────────────────
def make_image(var, cmap, color_range, opacity):
    data = ds[var].values.copy()
    info = LAYER_INFO[var]

    if var in CATEGORICAL:
        data = np.where(np.isnan(data), -1, data)
        img = hv.Image(
            (ds.x.values, ds.y.values, data),
            kdims=["x", "y"], vdims=[var],
        ).opts(
            cmap="tab10", clim=(-0.5, 10.5),
            alpha=opacity,
            colorbar=True,
            tools=["hover", "box_select"],
            width=700, height=600,
            xlabel="Easting (m)", ylabel="Northing (m)",
            title=info["label"],
        )
    else:
        img = hv.Image(
            (ds.x.values, ds.y.values, data),
            kdims=["x", "y"], vdims=[var],
        ).opts(
            cmap=cmap,
            clim=tuple(color_range),
            alpha=opacity,
            colorbar=True,
            tools=["hover", "box_select"],
            width=700, height=600,
            xlabel="Easting (m)", ylabel="Northing (m)",
            title=info["label"],
        )
    return img


# tap stream for point query
tap_stream = streams.Tap(x=0, y=0)

def on_tap(x, y):
    if x == 0 and y == 0:
        return

    # find nearest grid cell
    xi = np.argmin(np.abs(ds.x.values - x))
    yi = np.argmin(np.abs(ds.y.values - y))

    lines = [f"### Point Query",
             f"**Location:** ({x:.1f}, {y:.1f})",
             f"**Grid cell:** ({xi}, {yi})", ""]

    for var in LAYER_NAMES:
        info = LAYER_INFO[var]
        val = float(ds[var].values[yi, xi])

        if var == "sediment_class" and not np.isnan(val):
            idx = int(val)
            if 0 <= idx < len(SEDIMENT_LABELS):
                lines.append(f"**{info['long_name']}:** {SEDIMENT_LABELS[idx]}")
            else:
                lines.append(f"**{info['long_name']}:** class {idx}")
        elif np.isnan(val):
            lines.append(f"**{info['long_name']}:** no data")
        else:
            if info["units"]:
                lines.append(f"**{info['long_name']}:** {val:.4g} {info['units']}")
            else:
                lines.append(f"**{info['long_name']}:** {val:.4g}")

    query_pane.object = "\n\n".join(lines)


tap_watcher = tap_stream.param.watch(
    lambda event: on_tap(tap_stream.x, tap_stream.y),
    ["x", "y"],
)


# box select stream for region stats
box_stream = streams.BoundsXY(bounds=(0, 0, 0, 0))

def on_box(bounds):
    x0, y0, x1, y1 = bounds
    if x0 == x1 or y0 == y1:
        return

    # select region
    region = ds.sel(
        x=slice(min(x0, x1), max(x0, x1)),
        y=slice(max(y0, y1), min(y0, y1)),
    )

    lines = [f"### Region Statistics",
             f"**Bounds:** ({x0:.0f}, {y0:.0f}) — ({x1:.0f}, {y1:.0f})",
             f"**Size:** {abs(x1-x0):.0f} × {abs(y1-y0):.0f} m", ""]

    for var in LAYER_NAMES:
        info = LAYER_INFO[var]
        vals = region[var].values.ravel()
        valid = vals[np.isfinite(vals)]

        if len(valid) == 0:
            lines.append(f"**{info['long_name']}:** no data in region")
            continue

        if var == "sediment_class":
            classes, counts = np.unique(valid.astype(int), return_counts=True)
            dominant = classes[np.argmax(counts)]
            if 0 <= dominant < len(SEDIMENT_LABELS):
                lines.append(f"**{info['long_name']}:** {SEDIMENT_LABELS[dominant]} "
                             f"({100 * counts.max() / counts.sum():.0f}%)")
        else:
            u = info["units"]
            lines.append(
                f"**{info['long_name']}:** "
                f"{valid.min():.3g} ~ {valid.max():.3g} {u}, "
                f"mean={valid.mean():.3g} {u}"
            )

    stats_pane.object = "\n\n".join(lines)


box_watcher = box_stream.param.watch(
    lambda event: on_box(box_stream.bounds),
    ["bounds"],
)


# ── Dynamic Map ──────────────────────────────────────────────
@pn.depends(layer_select, cmap_select, range_slider, opacity_slider)
def map_view(var, cmap, color_range, opacity):
    img = make_image(var, cmap, color_range, opacity)

    # attach streams
    tap_stream.source = img
    box_stream.source = img

    return img


# ── Layout ───────────────────────────────────────────────────
controls = pn.Column(
    pn.pane.Markdown("## Data Cube Viewer", styles={"font-size": "16px"}),
    pn.pane.Markdown(f"**File:** {NC_PATH.name}"),
    pn.pane.Markdown(f"**Layers:** {len(LAYER_NAMES)}"),
    pn.layout.Divider(),
    layer_select,
    cmap_select,
    range_slider,
    opacity_slider,
    pn.layout.Divider(),
    query_pane,
    pn.layout.Divider(),
    stats_pane,
    width=320,
)

main_layout = pn.Row(
    controls,
    pn.panel(map_view, sizing_mode="stretch_both"),
    sizing_mode="stretch_both",
)

# serve
main_layout.servable(title="Hydrospatial Data Cube Viewer")