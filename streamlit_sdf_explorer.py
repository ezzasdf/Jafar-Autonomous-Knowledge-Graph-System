import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import streamlit as st
import plotly.graph_objects as go

from tools.implicit_surface_tool import (
    _marching_cubes, _sphere, _box, _cylinder, _torus,
    _sdf_union, _sdf_intersect, _sdf_subtract, _sdf_smooth_union,
)

st.set_page_config(page_title="SDF Explorer", layout="wide")
st.title("SDF Shape Explorer")

SHAPE_DEFS = {
    "sphere": {"fn": _sphere, "params": {"r": (0.1, 3.0, 1.0)}},
    "box": {"fn": _box, "params": {"size_x": (0.1, 3.0, 1.0), "size_y": (0.1, 3.0, 1.0), "size_z": (0.1, 3.0, 1.0)}},
    "cylinder": {"fn": _cylinder, "params": {"r": (0.1, 3.0, 1.0), "h": (0.1, 5.0, 2.0)}},
    "torus": {"fn": _torus, "params": {"R": (0.5, 4.0, 2.0), "r": (0.1, 2.0, 0.5)}},
}

OP_DEFS = {
    "none": "Single Shape",
    "union": "Union (A ∪ B)",
    "intersect": "Intersect (A ∩ B)",
    "subtract": "Subtract (A \\ B)",
    "smooth_union": "Smooth Union",
}

with st.sidebar:
    st.header("Shape A")
    shape_a = st.selectbox("Primitive", list(SHAPE_DEFS.keys()), key="a")
    params_a = {}
    for name, (lo, hi, default) in SHAPE_DEFS[shape_a]["params"].items():
        if name == "size_x":
            params_a["size"] = np.array([
                st.slider(f"A: size_x", lo, hi, default, key="ax"),
                st.slider(f"A: size_y", lo, hi, default, key="ay"),
                st.slider(f"A: size_z", lo, hi, default, key="az"),
            ])
        elif name == "size_y" or name == "size_z":
            continue
        else:
            params_a[name] = st.slider(f"A: {name}", lo, hi, default, key=f"a_{name}")

    op = st.selectbox("Operation", list(OP_DEFS.keys()), format_func=lambda k: OP_DEFS[k])
    op_k = 0.5
    if op == "smooth_union":
        op_k = st.slider("Smooth factor k", 0.0, 2.0, 0.5)

    with st.expander("Shape B", True):
        shape_b = st.selectbox("Primitive", list(SHAPE_DEFS.keys()), key="b")
        params_b = {}
        for name, (lo, hi, default) in SHAPE_DEFS[shape_b]["params"].items():
            if name == "size_x":
                params_b["size"] = np.array([
                    st.slider(f"B: size_x", lo, hi, default, key="bx"),
                    st.slider(f"B: size_y", lo, hi, default, key="by"),
                    st.slider(f"B: size_z", lo, hi, default, key="bz"),
                ])
            elif name == "size_y" or name == "size_z":
                continue
            else:
                params_b[name] = st.slider(f"B: {name}", lo, hi, default, key=f"b_{name}")

    st.divider()
    st.subheader("Variable Selection")
    all_vars = {}
    for k, v in params_a.items():
        all_vars[f"A.{k}"] = v
    for k, v in params_b.items():
        all_vars[f"B.{k}"] = v

    var_labels = list(all_vars.keys())
    if len(var_labels) == 1:
        sel = st.selectbox("Select variable", var_labels)
        selected_vars = [sel] if sel else []
    elif len(var_labels) > 1:
        selected_vars = st.multiselect("Select variables", var_labels, default=[])
    else:
        selected_vars = []

    res = st.slider("Grid resolution", 8, 64, 24, key="res")

    render_opts = st.multiselect("Render options", ["wireframe", "surface", "points"], default=["surface"])

def make_volume(shape, params, res, grid_info):
    p = grid_info["p"]
    fn = SHAPE_DEFS[shape]["fn"]
    if shape == "box":
        return fn(p, params["size"])
    elif shape == "cylinder":
        return fn(p, params["r"], params["h"])
    elif shape == "torus":
        return fn(p, params["R"], params["r"])
    else:
        return fn(p, params.get("r", 1.0))

@st.fragment
def draw():
    grid = np.linspace(-2.5, 2.5, res)
    x, y, z = np.meshgrid(grid, grid, grid, indexing="ij")
    grid_info = {"p": np.stack([x, y, z], axis=-1), "x": x, "y": y, "z": z}

    v_a = make_volume(shape_a, params_a, res, grid_info)
    level = 0.0

    if op == "none":
        vol = v_a
    else:
        v_b = make_volume(shape_b, params_b, res, grid_info)
        op_fn = {
            "union": _sdf_union,
            "intersect": _sdf_intersect,
            "subtract": _sdf_subtract,
            "smooth_union": lambda a, b: _sdf_smooth_union(a, b, op_k),
        }[op]
        vol = op_fn(v_a, v_b)

    verts, faces = _marching_cubes(vol, level)

    fig = go.Figure()
    fig.update_layout(
        scene=dict(
            xaxis=dict(range=[-2.5, 2.5]),
            yaxis=dict(range=[-2.5, 2.5]),
            zaxis=dict(range=[-2.5, 2.5]),
            aspectmode="cube",
        ),
        margin=dict(l=0, r=0, t=0, b=0),
        height=700,
    )

    if verts.shape[0] > 0 and faces.shape[0] > 0:
        if "surface" in render_opts:
            fig.add_trace(go.Mesh3d(
                x=verts[:, 0], y=verts[:, 1], z=verts[:, 2],
                i=faces[:, 0], j=faces[:, 1], k=faces[:, 2],
                color="lightblue", opacity=0.85, lighting=dict(diffuse=0.8, ambient=0.3),
                name="surface",
            ))
        if "wireframe" in render_opts:
            tri_verts = verts[faces.flatten()]
            xe, ye, ze = [], [], []
            for f_idx in range(faces.shape[0]):
                for j in range(3):
                    a, b = faces[f_idx, j], faces[f_idx, (j + 1) % 3]
                    xe += [verts[a, 0], verts[b, 0], None]
                    ye += [verts[a, 1], verts[b, 1], None]
                    ze += [verts[a, 2], verts[b, 2], None]
            fig.add_trace(go.Scatter3d(
                x=xe, y=ye, z=ze,
                mode="lines", line=dict(color="gray", width=1),
                name="wireframe",
                showlegend=False,
            ))
        if "points" in render_opts:
            fig.add_trace(go.Scatter3d(
                x=verts[:, 0], y=verts[:, 1], z=verts[:, 2],
                mode="markers", marker=dict(size=2, color="white", opacity=0.5),
                name="vertices",
            ))

        st.plotly_chart(fig, use_container_width=True)

        with st.expander("Mesh stats", icon="📊"):
            c1, c2, c3 = st.columns(3)
            c1.metric("Vertices", verts.shape[0])
            c2.metric("Triangles", faces.shape[0])
            c3.metric("Resolution", f"{res}³")

        if selected_vars:
            st.subheader("Variable Comparison")
            vcols = st.columns(min(len(selected_vars), 4))
            for ci, vname in enumerate(selected_vars):
                val = all_vars[vname]
                vcols[ci % 4].metric(vname, f"{val:.3f}" if isinstance(val, float) else str(val))
    else:
        st.info("No surface generated — the isosurface may be empty at this level.")

draw()
