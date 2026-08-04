import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np
import random

# --- Configuration & Constants ---
FPS = 30
TOTAL_FRAMES = 1300
METHODS = ["yolo11s", "GT"]
CLASSES = [
    "teminal", "IPI_card", "cable_tie", "tether", 
    "mount_puck", "usb_cable", "mount", 
    "sticker1", "sticker2", "gap_idle"
]

# Color palette inspired by the reference image
COLOR_MAP = {
    "teminal": "#ff7f0e",     # Orange
    "IPI_card": "#d62728",    # Red
    "cable_tie": "#8c564b",   # Brown
    "tether": "#e377c2",      # Pink
    "mount_puck": "#bcbd22",  # Olive
    "usb_cable": "#17becf",   # Cyan
    "mount": "#9467bd",       # Purple
    "sticker1": "#2ca02c",    # Green
    "sticker2": "#20026b",    # Pink (repeat if needed)
    "gap_idle": "#4682B4"     # Steel Blue (similar to 'No' in image)
}

def generate_mock_data():
    gt_array = np.full(TOTAL_FRAMES, "gap_idle", dtype=object)
    curr = 0

    # Phase 1
    p1_steps = ["teminal", "IPI_card", "cable_tie", "tether", "mount_puck", "usb_cable", "mount"]
    for cls in p1_steps:
        dur = 3 * FPS if cls == "teminal" else random.randint(2 * FPS, 3 * FPS)
        end = min(curr + dur, TOTAL_FRAMES)
        gt_array[curr:end] = cls
        curr = end
        gap = random.randint(25, 35)
        curr = min(curr + gap, TOTAL_FRAMES)

    # The Big Gap
    curr = min(curr + 300, TOTAL_FRAMES)

    # Phase 2
    for cls in ["sticker1", "sticker2"]:
        if curr >= TOTAL_FRAMES: break
        dur = 60 
        end = min(curr + dur, TOTAL_FRAMES)
        gt_array[curr:end] = cls
        curr = end
        gap = 45
        curr = min(curr + gap, TOTAL_FRAMES)

    # Model Array
    model_array = np.full(TOTAL_FRAMES, "gap_idle", dtype=object)
    segments = []
    start_idx = 0
    for i in range(1, TOTAL_FRAMES):
        if gt_array[i] != gt_array[start_idx]:
            segments.append((gt_array[start_idx], start_idx, i))
            start_idx = i
    segments.append((gt_array[start_idx], start_idx, TOTAL_FRAMES))

    for cls, start, end in segments:
        if cls == "gap_idle": continue
        m_start = start + random.randint(2, 15)
        m_end = end + random.randint(-12, 8)
        if m_start < m_end:
            model_array[max(0, m_start):min(TOTAL_FRAMES, m_end)] = cls

    # Flickering
    for _ in range(50):
        f_start = random.randint(0, TOTAL_FRAMES - 5)
        f_dur = random.randint(1, 4)
        if model_array[f_start] != "gap_idle":
            model_array[f_start:f_start+f_dur] = "gap_idle"

    def array_to_df(arr, method_name):
        records = []
        s_idx = 0
        for i in range(1, len(arr)):
            if arr[i] != arr[s_idx]:
                records.append({"Method": method_name, "Class": arr[s_idx], "Start": s_idx, "Duration": i - s_idx})
                s_idx = i
        records.append({"Method": method_name, "Class": arr[s_idx], "Start": s_idx, "Duration": len(arr) - s_idx})
        return records

    return array_to_df(model_array, "yolo11s"), array_to_df(gt_array, "GT")

# --- Plot Generation ---
model_data, gt_data = generate_mock_data()

# Create subplots: one row per method
fig = make_subplots(
    rows=2, cols=1, 
    shared_xaxes=True, 
    vertical_spacing=0.05
)

fig.update_yaxes(title_text="yolo11s", row=1, col=1)
fig.update_yaxes(title_text="GT", row=2, col=1)

# Function to add segments to a specific row
def add_method_to_plot(data, row_idx):
    for _, row in pd.DataFrame(data).iterrows():
        fig.add_trace(
            go.Bar(
                x=[row["Duration"]],
                y=[row["Method"]],
                base=[row["Start"]],
                orientation='h',
                marker_color=COLOR_MAP[row["Class"]],
                showlegend=False,
                name=row["Class"],
                hoverinfo='name',
                width=1.0  # Full height of the row
            ),
            row=row_idx, col=1
        )

add_method_to_plot(model_data, 1)
add_method_to_plot(gt_data, 2)

# --- Add legend manually (one item per unique class) ---
for cls in CLASSES:
    fig.add_trace(
        go.Bar(
            x=[0], y=[METHODS[0]], marker_color=COLOR_MAP[cls],
            name=cls, showlegend=True, legendgroup=cls
        ),
        row=1, col=1
    )

# --- Styling & Layout ---
fig.update_layout(
    title=dict(
        text="Action Segmentation for Packaging Task",
        x=0.5, y=0.98, xanchor='center', font=dict(size=16)
    ),
    xaxis2_title="Frame",
    plot_bgcolor="white",
    height=400,
    width=1300,
    margin=dict(l=150, r=40, t=100, b=100),
    barmode='stack',
    legend=dict(
        orientation="h",
        yanchor="bottom",
        y=1.02,
        xanchor="center",
        x=0.5,
        font=dict(size=10)
    ),
    # Add the figure caption as an annotation
    annotations=[
        dict(
            text="",
            xref="paper", yref="paper",
            x=0.5, y=-0.35,
            showarrow=False,
            font=dict(size=14, family="Times New Roman")
        )
    ]
)

# Clean up axes
fig.update_xaxes(showgrid=False, zeroline=False, range=[0, TOTAL_FRAMES], linecolor="black", mirror=True)
fig.update_yaxes(showgrid=False, zeroline=False, showticklabels=False, linecolor="black", mirror=True)

# Remove bar outlines
fig.update_traces(marker_line_width=0)

fig.show()
