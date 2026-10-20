"""Tkinter GUI for pySHAPE.

Lets a user:
  * load compatible geometry (STL natively; OBJ/PLY/OFF/glTF/3MF/DAE via the
    optional ``trimesh`` dependency, unified into a triangle mesh),
  * compute the SHAPE result parameters and tick which ones to export,
  * export the ticked parameters to CSV next to the input file, with a
    configurable filename suffix and optional progressive numbering to
    avoid overwriting previous exports,
  * preview the geometry in a slowly auto-rotating 3D view with an XYZ axes
    triad scaled to the mesh's bounding box.

Run with ``python -m pyshape.gui``.
"""
from __future__ import annotations

import sys
import tkinter as tk
from pathlib import Path
from tkinter import filedialog, messagebox, ttk
from typing import Dict, Optional

import numpy as np

from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from matplotlib.figure import Figure
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

from .analysis import UNIT_SCALES, prepare_mesh, scale_nodes
from .gui_core import (
    LoadedGeometry,
    Metric,
    SUPPORTED_EXTENSIONS,
    axis_triad,
    compute_metrics,
    load_geometry,
    make_export_path,
    write_csv,
)

ROTATION_DEG_PER_TICK = 0.6
ROTATION_INTERVAL_MS = 40


class ShapeGUI(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("pySHAPE Viewer")
        self.geometry("1200x720")

        self.geo: Optional[LoadedGeometry] = None
        self.nodes: Optional[np.ndarray] = None
        self.faces: Optional[np.ndarray] = None
        self.metrics: list[Metric] = []
        self.metric_vars: Dict[str, tk.BooleanVar] = {}

        self._rotation_job: Optional[str] = None
        self._azim = -60.0
        self._rotating_paused = False

        self._build_ui()
        self.protocol("WM_DELETE_WINDOW", self._on_close)

    # ------------------------------------------------------------------
    # Layout
    # ------------------------------------------------------------------
    def _build_ui(self):
        self.columnconfigure(0, weight=0)
        self.columnconfigure(1, weight=1)
        self.rowconfigure(0, weight=1)

        left = ttk.Frame(self, padding=8)
        left.grid(row=0, column=0, sticky="nsw")
        right = ttk.Frame(self, padding=0)
        right.grid(row=0, column=1, sticky="nsew")
        right.rowconfigure(0, weight=1)
        right.columnconfigure(0, weight=1)

        self._build_file_section(left)
        self._build_results_section(left)
        self._build_export_section(left)
        self._build_preview(right)

    def _build_file_section(self, parent):
        row = 0
        ttk.Label(parent, text="Geometry file").grid(row=row, column=0, sticky="w")
        row += 1
        self.path_var = tk.StringVar()
        ttk.Entry(parent, textvariable=self.path_var, width=40).grid(
            row=row, column=0, columnspan=2, sticky="we", pady=(0, 4)
        )
        ttk.Button(parent, text="Browse…", command=self.browse_file).grid(row=row, column=2, padx=(6, 0))
        row += 1

        ttk.Label(parent, text="Units").grid(row=row, column=0, sticky="w", pady=(6, 0))
        self.units_var = tk.StringVar(value="auto")
        ttk.Combobox(
            parent, textvariable=self.units_var, state="readonly", width=10,
            values=["auto", *UNIT_SCALES.keys()],
        ).grid(row=row, column=1, sticky="w", pady=(6, 0))
        row += 1

        ttk.Button(parent, text="Load", command=self.load_mesh).grid(
            row=row, column=0, sticky="we", pady=(6, 8)
        )
        self.file_info_var = tk.StringVar(value="No geometry loaded.")
        row += 1
        ttk.Label(parent, textvariable=self.file_info_var, wraplength=320, justify="left").grid(
            row=row, column=0, columnspan=3, sticky="w"
        )
        row += 1
        ttk.Separator(parent, orient="horizontal").grid(
            row=row, column=0, columnspan=3, sticky="we", pady=(8, 8)
        )
        self._file_row_next = row + 1

    def _build_results_section(self, parent):
        row = self._file_row_next
        ttk.Button(parent, text="Calculate results", command=self.calculate_metrics).grid(
            row=row, column=0, columnspan=2, sticky="we"
        )
        row += 1
        ttk.Label(parent, text="Results (tick to export)").grid(row=row, column=0, sticky="w", pady=(8, 0))
        row += 1
        btns = ttk.Frame(parent)
        btns.grid(row=row, column=0, columnspan=3, sticky="w")
        ttk.Button(btns, text="Select all", command=lambda: self._set_all_metrics(True)).pack(side="left")
        ttk.Button(btns, text="Select none", command=lambda: self._set_all_metrics(False)).pack(
            side="left", padx=(6, 0)
        )
        row += 1

        container = ttk.Frame(parent)
        container.grid(row=row, column=0, columnspan=3, sticky="nsew", pady=(4, 0))
        parent.rowconfigure(row, weight=1)
        canvas = tk.Canvas(container, width=330, height=260, highlightthickness=0)
        scrollbar = ttk.Scrollbar(container, orient="vertical", command=canvas.yview)
        self.results_frame = ttk.Frame(canvas)
        self.results_frame.bind(
            "<Configure>", lambda e: canvas.configure(scrollregion=canvas.bbox("all"))
        )
        canvas.create_window((0, 0), window=self.results_frame, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)
        canvas.grid(row=0, column=0, sticky="nsew")
        scrollbar.grid(row=0, column=1, sticky="ns")
        container.rowconfigure(0, weight=1)
        container.columnconfigure(0, weight=1)
        self._results_canvas = canvas
        row += 1
        ttk.Separator(parent, orient="horizontal").grid(
            row=row, column=0, columnspan=3, sticky="we", pady=(8, 8)
        )
        self._results_row_next = row + 1

    def _build_export_section(self, parent):
        row = self._results_row_next
        ttk.Label(parent, text="Export").grid(row=row, column=0, sticky="w")
        row += 1
        ttk.Label(parent, text="Filename suffix").grid(row=row, column=0, sticky="w")
        self.suffix_var = tk.StringVar(value="_SHAPE")
        self.suffix_var.trace_add("write", lambda *a: self._update_export_preview())
        ttk.Entry(parent, textvariable=self.suffix_var, width=16).grid(row=row, column=1, sticky="w")
        row += 1

        self.avoid_overwrite_var = tk.BooleanVar(value=True)
        self.avoid_overwrite_var.trace_add("write", lambda *a: self._update_export_preview())
        ttk.Checkbutton(
            parent,
            text="Avoid overwrite (append _1, _2, ... if file exists)",
            variable=self.avoid_overwrite_var,
        ).grid(row=row, column=0, columnspan=3, sticky="w", pady=(2, 0))
        row += 1

        self.export_preview_var = tk.StringVar(value="")
        ttk.Label(parent, textvariable=self.export_preview_var, foreground="#555").grid(
            row=row, column=0, columnspan=3, sticky="w", pady=(4, 4)
        )
        row += 1
        ttk.Button(parent, text="Export CSV…", command=self.export_csv).grid(
            row=row, column=0, columnspan=2, sticky="we"
        )

    def _build_preview(self, parent):
        top = ttk.Frame(parent)
        top.grid(row=0, column=0, sticky="we")
        self.rotate_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(
            top, text="Slow auto-rotate", variable=self.rotate_var,
            command=self._on_rotate_toggle,
        ).pack(side="left")

        self.fig = Figure(figsize=(6, 5), dpi=100)
        self.ax = self.fig.add_subplot(111, projection="3d")
        self.ax.set_box_aspect([1, 1, 1])
        self.canvas = FigureCanvasTkAgg(self.fig, master=parent)
        self.canvas.draw()
        self.canvas.get_tk_widget().grid(row=1, column=0, sticky="nsew")
        parent.rowconfigure(1, weight=1)
        toolbar = NavigationToolbar2Tk(self.canvas, parent, pack_toolbar=False)
        toolbar.update()
        toolbar.grid(row=2, column=0, sticky="we")

        widget = self.canvas.get_tk_widget()
        widget.bind("<Button-1>", lambda e: self._pause_rotation())
        widget.bind("<ButtonRelease-1>", lambda e: self._resume_rotation())

    # ------------------------------------------------------------------
    # File loading
    # ------------------------------------------------------------------
    def browse_file(self):
        filetypes = [
            ("Compatible geometry", " ".join(f"*{ext}" for ext in SUPPORTED_EXTENSIONS)),
            ("All files", "*.*"),
        ]
        filename = filedialog.askopenfilename(
            title="Open geometry", initialdir=str(Path.cwd()), filetypes=filetypes
        )
        if filename:
            self.path_var.set(filename)

    def load_mesh(self):
        path = self.path_var.get().strip()
        if not path:
            messagebox.showwarning("No file", "Please specify a geometry file.")
            return
        try:
            geo = load_geometry(path)
            nodes, _ = scale_nodes(geo.nodes, self.units_var.get())
            nodes, faces = prepare_mesh(nodes, geo.faces)
        except Exception as exc:
            messagebox.showerror("Load error", str(exc))
            return

        self.geo = geo
        self.nodes, self.faces = nodes, faces
        self.metrics = []
        self._clear_results()

        info = (
            f"{geo.source_path.name}\n"
            f"format: {geo.source_format.upper()}"
            + (" (converted to a triangle mesh)" if geo.converted_from_native else "")
            + f"\nvertices: {nodes.shape[0]}  faces: {faces.shape[0]}"
        )
        self.file_info_var.set(info)
        self._update_export_preview()
        self._render_mesh()
        self._start_rotation()

    # ------------------------------------------------------------------
    # Results / metrics
    # ------------------------------------------------------------------
    def _clear_results(self):
        for child in self.results_frame.winfo_children():
            child.destroy()
        self.metric_vars = {}

    def calculate_metrics(self):
        if self.nodes is None or self.faces is None:
            messagebox.showwarning("No mesh", "Load a geometry file first.")
            return
        try:
            self.metrics = compute_metrics(self.nodes, self.faces)
        except Exception as exc:
            messagebox.showerror("Compute error", str(exc))
            return
        self._populate_results()

    def _populate_results(self):
        self._clear_results()
        current_group = None
        r = 0
        for metric in self.metrics:
            if metric.group != current_group:
                current_group = metric.group
                ttk.Label(
                    self.results_frame, text=current_group, font=("TkDefaultFont", 9, "bold")
                ).grid(row=r, column=0, sticky="w", pady=(6 if r else 0, 2))
                r += 1
            var = tk.BooleanVar(value=metric.default_selected and metric.value is not None)
            self.metric_vars[metric.key] = var
            cb = ttk.Checkbutton(
                self.results_frame,
                text=f"{metric.label} = {metric.text}",
                variable=var,
                state=("normal" if metric.value is not None else "disabled"),
            )
            cb.grid(row=r, column=0, sticky="w")
            r += 1

    def _set_all_metrics(self, value: bool):
        for metric in self.metrics:
            if metric.value is None:
                continue
            self.metric_vars[metric.key].set(value)

    # ------------------------------------------------------------------
    # Export
    # ------------------------------------------------------------------
    def _update_export_preview(self):
        if self.geo is None:
            self.export_preview_var.set("")
            return
        try:
            out = make_export_path(
                self.geo.source_path, self.suffix_var.get(), self.avoid_overwrite_var.get()
            )
            self.export_preview_var.set(f"Will write: {out.name}")
        except Exception:
            self.export_preview_var.set("")

    def export_csv(self):
        if self.geo is None:
            messagebox.showwarning("No geometry", "Load a geometry file first.")
            return
        if not self.metrics:
            messagebox.showwarning("No results", "Click 'Calculate results' first.")
            return
        selected = [
            (m.label, m.value)
            for m in self.metrics
            if m.value is not None and self.metric_vars.get(m.key, tk.BooleanVar(value=False)).get()
        ]
        if not selected:
            messagebox.showwarning("Nothing ticked", "Tick at least one result parameter to export.")
            return

        default_path = make_export_path(
            self.geo.source_path, self.suffix_var.get(), self.avoid_overwrite_var.get()
        )
        out_path = filedialog.asksaveasfilename(
            title="Export CSV",
            initialdir=str(default_path.parent),
            initialfile=default_path.name,
            defaultextension=".csv",
            filetypes=[("CSV files", "*.csv"), ("All files", "*.*")],
        )
        if not out_path:
            return
        try:
            write_csv(out_path, selected, source_name=self.geo.source_path.name)
        except Exception as exc:
            messagebox.showerror("Export error", str(exc))
            return
        messagebox.showinfo("Exported", f"Wrote {len(selected)} parameter(s) to:\n{out_path}")

    # ------------------------------------------------------------------
    # 3D preview + rotation
    # ------------------------------------------------------------------
    def _render_mesh(self):
        self.ax.clear()
        self.ax.set_title("Geometry preview")
        if self.nodes is None or self.faces is None:
            self.canvas.draw()
            return
        verts = [self.nodes[idx] for idx in self.faces]
        coll = Poly3DCollection(
            verts, facecolors="lightsteelblue", edgecolors="k", linewidths=0.4, alpha=0.9
        )
        self.ax.add_collection3d(coll)

        mins = self.nodes.min(axis=0)
        maxs = self.nodes.max(axis=0)
        center = (mins + maxs) / 2.0
        size = float((maxs - mins).max())
        if size <= 0:
            size = 1.0
        lims = np.array([center - size / 2, center + size / 2]).T
        self.ax.set_xlim(*lims[0])
        self.ax.set_ylim(*lims[1])
        self.ax.set_zlim(*lims[2])
        self.ax.set_box_aspect([1, 1, 1])

        self._draw_axis_triad()
        self.ax.view_init(elev=20, azim=self._azim)
        self.canvas.draw()

    def _draw_axis_triad(self):
        center, length = axis_triad(self.nodes)
        colors = {"X": "r", "Y": "g", "Z": "b"}
        for axis_name, direction in zip("XYZ", np.eye(3)):
            end = center + direction * length
            self.ax.plot(
                [center[0], end[0]], [center[1], end[1]], [center[2], end[2]],
                color=colors[axis_name], linewidth=2,
            )
            self.ax.text(*end, axis_name, color=colors[axis_name], fontsize=10, fontweight="bold")

    def _on_rotate_toggle(self):
        if self.rotate_var.get():
            self._start_rotation()
        else:
            self._stop_rotation()

    def _start_rotation(self):
        if self._rotation_job is not None or not self.rotate_var.get():
            return
        self._rotating_paused = False
        self._tick_rotation()

    def _tick_rotation(self):
        if self.nodes is not None and not self._rotating_paused:
            self._azim = (self._azim + ROTATION_DEG_PER_TICK) % 360.0
            self.ax.view_init(elev=20, azim=self._azim)
            self.canvas.draw_idle()
        self._rotation_job = self.after(ROTATION_INTERVAL_MS, self._tick_rotation)

    def _stop_rotation(self):
        if self._rotation_job is not None:
            self.after_cancel(self._rotation_job)
            self._rotation_job = None

    def _pause_rotation(self):
        self._rotating_paused = True

    def _resume_rotation(self):
        self._rotating_paused = False

    def _on_close(self):
        self._stop_rotation()
        self.destroy()


def main():
    app = ShapeGUI()
    app.mainloop()


if __name__ == "__main__":
    sys.exit(main())
