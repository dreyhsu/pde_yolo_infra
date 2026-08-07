"""
Gantt timeline comparing model predictions against ground truth.

One subplot per model plus a ground-truth row at the bottom. Within each subplot
the y axis is categorical over class names, so overlapping classes each get their
own lane and stay readable. Bar colour identifies the class; how washed-out the
bar is encodes the model's mean confidence over that run -- dimmer means less
sure.

Normally driven by eval_report.py. Run directly to re-render a chart without
recomputing metrics:

    python generate_eval_gantt.py --project projects/hph_hw --video test1 --roi tray --conf 0.4
"""

import argparse
import os

import plotly.graph_objects as go
from plotly.subplots import make_subplots

from eval_common import build_color_map, frames_to_timecode, sanitize

GT_ROW_NAME = "GT"
ROW_PX_PER_CLASS = 26
BAR_WIDTH = 0.7


def dim_hex(hex_color, conf, conf_min, mode="blend", a_min=0.25, a_max=1.0):
    """
    Modulate a #rrggbb colour by confidence. Returns an 'rgb(...)' or 'rgba(...)'
    string, both of which plotly accepts for marker_color.

    Confidence is normalised over [conf_min, 1.0] rather than [0, 1]: at a
    threshold of 0.5 nothing below 0.5 is on the chart at all, so normalising from
    zero would render every bar at the same middling tone and throw away the whole
    point of the encoding.

    'blend' (the default) mixes toward the white plot background instead of using
    real transparency -- visually equivalent here, but it survives PNG export and
    does not double-darken where bars touch.
    """
    span = max(1e-6, 1.0 - conf_min)
    ratio = min(max((conf - conf_min) / span, 0.0), 1.0)
    alpha = a_min + (a_max - a_min) * ratio
    r, g, b = (int(hex_color[i:i + 2], 16) for i in (1, 3, 5))
    if mode == "alpha":
        return f"rgba({r},{g},{b},{alpha:.3f})"
    r, g, b = (round(c + (255 - c) * (1 - alpha)) for c in (r, g, b))
    return f"rgb({r},{g},{b})"


def _add_bar(fig, row, cls, start, end, color, hover, text=None):
    fig.add_trace(
        go.Bar(
            x=[end - start + 1],
            y=[cls],
            base=[start],
            orientation="h",
            width=BAR_WIDTH,
            marker=dict(color=color),
            showlegend=False,
            hovertemplate=hover + "<extra></extra>",
            text=text,
            textposition="inside",
            insidetextfont=dict(size=10, color="black"),
            cliponaxis=False,
        ),
        row=row, col=1,
    )


def build_gantt(gt, roi_name, per_model_segments, classes, conf_thr,
                title="", dim_mode="blend", color_map=None):
    """
    per_model_segments: {model_name: {class_name: [(start, end, mean_conf), ...]}}
    Ground-truth segments are read straight off the gt dict for `roi_name`.
    """
    color_map = color_map or build_color_map(classes)
    models = list(per_model_segments.keys())
    rows = models + [GT_ROW_NAME]
    fps = float(gt.get("fps") or 30.0)
    total_frames = int(gt["total_frames"])

    fig = make_subplots(
        rows=len(rows), cols=1, shared_xaxes=True, vertical_spacing=0.03,
        row_titles=rows,
    )

    for r, model in enumerate(models, start=1):
        for cls, segments in per_model_segments[model].items():
            for start, end, conf in segments:
                _add_bar(
                    fig, r, cls, start, end,
                    dim_hex(color_map[cls], conf, conf_thr, dim_mode),
                    f"<b>{model}</b><br>{cls}<br>frames {start}-{end} "
                    f"({end - start + 1})<br>conf {conf:.2f}",
                )

    gt_row = len(rows)
    for seg in gt["segments"]:
        if seg["roi"] != roi_name or seg["class"] not in color_map:
            continue
        start, end = int(seg["start_frame"]), int(seg["end_frame"])
        fig.add_trace(
            go.Bar(
                x=[end - start + 1], y=[seg["class"]], base=[start],
                orientation="h", width=BAR_WIDTH,
                marker=dict(color=color_map[seg["class"]], line=dict(width=1, color="black")),
                showlegend=False,
                text=[f"x{seg['count']}" if seg.get("count", 1) > 1 else ""],
                textposition="inside",
                insidetextanchor="middle",
                textangle=0,
                constraintext="none",
                insidetextfont=dict(size=11, color="white"),
                hovertemplate=(
                    f"<b>GT</b><br>{seg['class']} x{seg.get('count', 1)}<br>"
                    f"frames {start}-{end} ({end - start + 1})<extra></extra>"
                ),
                cliponaxis=False,
            ),
            row=gt_row, col=1,
        )

    # Legend entries: zero-width bars at full saturation, one per class.
    for cls in classes:
        fig.add_trace(
            go.Bar(x=[0], y=[cls], base=[0], orientation="h",
                   marker=dict(color=color_map[cls]), name=cls,
                   showlegend=True, legendgroup=cls, hoverinfo="skip"),
            row=1, col=1,
        )

    order = list(reversed(classes))
    fig.update_yaxes(
        categoryorder="array", categoryarray=order,
        showgrid=True, gridcolor="#eeeeee", zeroline=False,
        linecolor="black", mirror=True, tickfont=dict(size=10),
    )

    n_ticks = 10
    step = max(1, total_frames // n_ticks)
    tickvals = list(range(0, total_frames + 1, step))
    fig.update_xaxes(
        range=[0, total_frames], showgrid=False, zeroline=False,
        linecolor="black", mirror=True,
        tickvals=tickvals,
        ticktext=[f"{t}<br>{frames_to_timecode(t, fps)}" for t in tickvals],
    )
    fig.update_xaxes(title_text="Frame / time", row=len(rows), col=1)

    fig.update_layout(
        title=dict(text=title, x=0.5, xanchor="center", font=dict(size=15)),
        barmode="overlay",
        plot_bgcolor="white",
        width=1400,
        height=ROW_PX_PER_CLASS * len(classes) * len(rows) + 180,
        margin=dict(l=110, r=90, t=110, b=70),
        legend=dict(orientation="h", yanchor="bottom", y=1.015,
                    xanchor="center", x=0.5, font=dict(size=10)),
        bargap=0.1,
    )
    fig.update_traces(marker_line_width=0, selector=dict(type="bar", showlegend=False))
    for annotation in fig.layout.annotations:
        annotation.font.size = 11
    return fig


def write_figure(fig, stem):
    """HTML always; PNG only if kaleido is installed."""
    html_path = stem + ".html"
    fig.write_html(html_path, include_plotlyjs="cdn")
    print(f"   📈 {html_path}")
    try:
        fig.write_image(stem + ".png", scale=2)
        print(f"   🖼  {stem}.png")
    except Exception as exc:
        print(f"   ⚠ PNG export skipped ({type(exc).__name__}: {exc}).")
        print("     Install with: pip install -U kaleido  (plotly 5.x needs kaleido==0.2.1)")
    return html_path


def main():
    """Standalone path: rebuild a chart from the detection cache and ground truth."""
    from eval_common import (
        get_roi, load_classes, load_det_cache, load_gt, load_project,
        pred_frame_counts, rle_segments, smooth_segments,
    )

    parser = argparse.ArgumentParser(description="Render an evaluation Gantt timeline.")
    parser.add_argument("--project", required=True)
    parser.add_argument("--video", required=True)
    parser.add_argument("--roi", required=True)
    parser.add_argument("--conf", type=float, required=True)
    parser.add_argument("--models", help="Comma-separated model names (default: all)")
    parser.add_argument("--dim-mode", choices=["blend", "alpha"], default="blend")
    parser.add_argument("--out", help="Output path stem (default: alongside the project cache)")
    args = parser.parse_args()

    cfg = load_project(args.project)
    classes = load_classes(args.project, cfg)
    gt = load_gt(args.project, args.video)
    if gt is None:
        print(f"❌ no ground truth for {args.video}")
        return
    roi = get_roi(gt, args.roi)

    models = cfg["models"]
    if args.models:
        wanted = {s.strip() for s in args.models.split(",")}
        models = [m for m in models if m["name"] in wanted]

    per_model = {}
    for m in models:
        try:
            df, meta = load_det_cache(args.project, m["name"], args.video)
        except FileNotFoundError as exc:
            print(f"   ⚠ {exc}")
            continue
        counts, confs, _ = pred_frame_counts(
            df, meta, roi, classes, args.conf, int(gt["total_frames"]),
            cfg.get("class_alias", {}),
        )
        by_class = {}
        for k, cls in enumerate(classes):
            segs = smooth_segments(
                rle_segments(counts[:, k] > 0),
                cfg["merge_gap_frames"], cfg["min_segment_frames"],
            )
            by_class[cls] = [
                (s, e, float(confs[s:e + 1, k][confs[s:e + 1, k] > 0].mean())
                 if (confs[s:e + 1, k] > 0).any() else 0.0)
                for s, e in segs
            ]
        per_model[m["name"]] = by_class

    if not per_model:
        print("❌ no detection cache found for any model")
        return

    fig = build_gantt(
        gt=gt, roi_name=args.roi, per_model_segments=per_model, classes=classes,
        conf_thr=args.conf,
        title=f"{cfg['project']} -- {args.video} / {args.roi} @ conf {args.conf:.2f}",
        dim_mode=args.dim_mode,
    )
    stem = args.out or os.path.join(
        args.project, "reports",
        f"gantt_{sanitize(args.video)}_{sanitize(args.roi)}_conf{args.conf:.2f}",
    )
    os.makedirs(os.path.dirname(stem), exist_ok=True)
    write_figure(fig, stem)


if __name__ == "__main__":
    main()
