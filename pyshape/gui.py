"""Tkinter GUI for pySHAPE.

Lets a user:
  * load compatible geometry (STL natively; OBJ/PLY/OFF/glTF/3MF/DAE via the
    optional ``trimesh`` dependency, unified into a triangle mesh),
  * compute the SHAPE result parameters and tick which ones to export,
  * export the ticked parameters to CSV next to the input file, with a
    configurable filename suffix and optional progressive numbering to
    avoid overwriting previous exports,
  * preview the geometry in a slowly auto-rotating 3D view with an XYZ axes
    triad scaled to the mesh's bounding box,
  * read what each result and control means: hovering shows a one-line
    description and the "?" buttons open the definition, formula and
    references (texts from :mod:`pyshape.descriptors`).

Run with ``python -m pyshape.gui``.
"""
from __future__ import annotations

import re
import sys
import tkinter as tk
import webbrowser
from pathlib import Path
from tkinter import filedialog, messagebox, ttk
from typing import Callable, Dict, Optional, Union

import numpy as np

from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from matplotlib.figure import Figure
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

from .analysis import (
    OUTPUT_UNITS,
    UNIT_SCALES,
    apply_output_units,
    resolve_output_units,
    scale_nodes,
    unit_label,
)
from . import descriptors
from .bounding import AXES_METHODS
from .preprocess import CAVITY_MODES, prepare_solid
from .gui_core import (
    face_normals,
    preview_face_colors,
    view_direction,
    LoadedGeometry,
    Metric,
    SUPPORTED_EXTENSIONS,
    axis_triad,
    compute_metrics,
    geo_metadata_rows,
    load_geometry,
    make_export_path,
    reader_text,
    units_summary,
    write_csv,
)

ROTATION_DEG_PER_TICK = 0.6
EDGE_FACE_LIMIT = 4000
ROTATION_INTERVAL_MS = 40

UNITS_HINT = (
    "Input: units of the coordinates in the file. Output: unit of lengths in the "
    "results; areas, volumes and inertia follow (e.g. mm², mm³, mm⁵). As-is keeps "
    "the file's units and disables the output choice."
)
SAME_AS_INPUT = "same as input"
TOOLTIP_DELAY_MS = 500


class Tooltip:
    """One-line help shown after a short delay while the pointer is over a widget."""

    def __init__(self, widget: tk.Widget, text: Union[str, Callable[[], str]],
                 delay: int = TOOLTIP_DELAY_MS):
        self.widget, self.text, self.delay = widget, text, delay
        self._job: Optional[str] = None
        self._window: Optional[tk.Toplevel] = None
        widget.bind("<Enter>", self._schedule, add="+")
        widget.bind("<Leave>", self._hide, add="+")
        widget.bind("<ButtonPress>", self._hide, add="+")

    def _schedule(self, _event=None):
        self._hide()
        self._job = self.widget.after(self.delay, self._show)

    def _show(self):
        self._job = None
        text = self.text() if callable(self.text) else self.text
        if not text or not self.widget.winfo_exists():
            return
        x = self.widget.winfo_pointerx() + 12
        y = self.widget.winfo_pointery() + 16
        self._window = window = tk.Toplevel(self.widget)
        window.wm_overrideredirect(True)
        window.wm_geometry(f"+{x}+{y}")
        tk.Label(window, text=text, justify="left", wraplength=360, background="#ffffe0",
                 relief="solid", borderwidth=1, padx=4, pady=2).pack()

    def _hide(self, _event=None):
        if self._job is not None:
            self.widget.after_cancel(self._job)
            self._job = None
        if self._window is not None:
            self._window.destroy()
            self._window = None


def _short(kind: str, key: str) -> str:
    """Tooltip text of a metric key, control name or axes method ("" if none)."""
    entry = {"metrics": descriptors.metric_help, "controls": descriptors.control_help,
             "axes": descriptors.axes_help}[kind](key)
    return str(entry["short"]) if entry else ""


class ShapeGUI(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("pySHAPE Viewer")
        self.geometry("1200x720")

        self.geo: Optional[LoadedGeometry] = None
        self.nodes: Optional[np.ndarray] = None
        self.faces: Optional[np.ndarray] = None
        # Scaled input mesh (before preparation) and the cavity-wall flags of the prepared one.
        self.raw_nodes: Optional[np.ndarray] = None
        self.raw_faces: Optional[np.ndarray] = None
        self.cavity_mask: Optional[np.ndarray] = None
        self.loaded_units = "as-is"
        self.loaded_scale = 1.0
        self.output_effective = "m"
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
        right.rowconfigure(0, weight=0)
        right.columnconfigure(0, weight=1)

        self._build_file_section(left)
        self._build_results_section(left)
        self._build_export_section(left)
        self._build_preview(right)

    def _build_file_section(self, parent):
        row = 0
        self._help_label(parent, "Geometry file", "file").grid(row=row, column=0, sticky="w")
        row += 1
        self.path_var = tk.StringVar()
        entry = ttk.Entry(parent, textvariable=self.path_var, width=40)
        entry.grid(row=row, column=0, columnspan=2, sticky="we", pady=(0, 4))
        browse = ttk.Button(parent, text="Browse…", command=self.browse_file)
        browse.grid(row=row, column=2, padx=(6, 0))
        self._tooltip(entry, "file")
        self._tooltip(browse, "file")
        row += 1

        self._help_label(parent, "Input units (file)", "input_units").grid(
            row=row, column=0, sticky="w", pady=(6, 0))
        self.units_var = tk.StringVar(value="as-is")
        units_combo = ttk.Combobox(
            parent, textvariable=self.units_var, state="readonly", width=12,
            values=list(UNIT_SCALES.keys()),
        )
        units_combo.grid(row=row, column=1, sticky="w", pady=(6, 0))
        self._tooltip(units_combo, "input_units")
        row += 1
        self._help_label(parent, "Output units", "output_units").grid(
            row=row, column=0, sticky="w", pady=(4, 0))
        self.output_units_var = tk.StringVar(value=SAME_AS_INPUT)
        self.output_units_combo = ttk.Combobox(
            parent, textvariable=self.output_units_var, state="readonly", width=12,
            values=[SAME_AS_INPUT, *OUTPUT_UNITS.keys()],
        )
        self.output_units_combo.grid(row=row, column=1, sticky="w", pady=(4, 0))
        self._tooltip(self.output_units_combo, "output_units")
        self.units_var.trace_add("write", lambda *a: self._sync_output_state())
        self.output_units_var.trace_add("write", lambda *a: self._refresh_units_summary())
        self._sync_output_state()
        row += 1
        ttk.Label(parent, text=UNITS_HINT, foreground="#555", wraplength=320, justify="left").grid(
            row=row, column=0, columnspan=3, sticky="w", pady=(2, 0)
        )
        row += 1

        self._help_label(parent, "Principal dimensions (S, I, L)", "axes").grid(
            row=row, column=0, sticky="w", pady=(6, 0))
        self.axes_var = tk.StringVar(value="pca")
        axes_combo = ttk.Combobox(
            parent, textvariable=self.axes_var, state="readonly", width=17,
            values=list(AXES_METHODS),
        )
        axes_combo.grid(row=row, column=1, sticky="w", pady=(6, 0))
        # The tooltip also describes the method currently chosen.
        Tooltip(axes_combo, lambda: f"{_short('controls', 'axes')}\n"
                f"{self.axes_var.get()}: {_short('axes', self.axes_var.get())}")
        row += 1

        self._help_label(parent, "Nested shells", "cavities").grid(
            row=row, column=0, sticky="w", pady=(6, 0))
        self.cavities_var = tk.StringVar(value="fill")
        cavities_combo = ttk.Combobox(
            parent, textvariable=self.cavities_var, state="readonly", width=17,
            values=list(CAVITY_MODES),
        )
        cavities_combo.grid(row=row, column=1, sticky="w", pady=(6, 0))
        cavities_combo.bind("<<ComboboxSelected>>", lambda _event: self._prepare_loaded())
        self._tooltip(cavities_combo, "cavities")
        row += 1

        load = ttk.Button(parent, text="Load", command=self.load_mesh)
        load.grid(row=row, column=0, sticky="we", pady=(6, 8))
        self._tooltip(load, "load")
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
        calculate = ttk.Button(parent, text="Calculate results", command=self.calculate_metrics)
        calculate.grid(row=row, column=0, columnspan=2, sticky="we")
        self._tooltip(calculate, "calculate")
        row += 1
        self._help_label(parent, "Results (tick to export, ? for help)", "results").grid(
            row=row, column=0, columnspan=3, sticky="w", pady=(8, 0))
        row += 1
        self.units_summary_var = tk.StringVar(value="")
        ttk.Label(parent, textvariable=self.units_summary_var, foreground="#555").grid(
            row=row, column=0, columnspan=3, sticky="w"
        )
        row += 1
        btns = ttk.Frame(parent)
        btns.grid(row=row, column=0, columnspan=3, sticky="w")
        select_all = ttk.Button(btns, text="Select all", command=lambda: self._set_all_metrics(True))
        select_all.pack(side="left")
        select_none = ttk.Button(btns, text="Select none",
                                 command=lambda: self._set_all_metrics(False))
        select_none.pack(side="left", padx=(6, 0))
        self._tooltip(select_all, "select_all")
        self._tooltip(select_none, "select_none")
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
        self._help_label(parent, "Filename suffix", "suffix").grid(row=row, column=0, sticky="w")
        self.suffix_var = tk.StringVar(value="_SHAPE")
        self.suffix_var.trace_add("write", lambda *a: self._update_export_preview())
        suffix = ttk.Entry(parent, textvariable=self.suffix_var, width=16)
        suffix.grid(row=row, column=1, sticky="w")
        self._tooltip(suffix, "suffix")
        row += 1

        self.avoid_overwrite_var = tk.BooleanVar(value=True)
        self.avoid_overwrite_var.trace_add("write", lambda *a: self._update_export_preview())
        avoid = ttk.Checkbutton(
            parent,
            text="Avoid overwrite (append _1, _2, ... if file exists)",
            variable=self.avoid_overwrite_var,
        )
        avoid.grid(row=row, column=0, columnspan=3, sticky="w", pady=(2, 0))
        self._tooltip(avoid, "avoid_overwrite")
        row += 1

        self.export_preview_var = tk.StringVar(value="")
        ttk.Label(parent, textvariable=self.export_preview_var, foreground="#555").grid(
            row=row, column=0, columnspan=3, sticky="w", pady=(4, 4)
        )
        row += 1
        export = ttk.Button(parent, text="Export CSV…", command=self.export_csv)
        export.grid(row=row, column=0, columnspan=2, sticky="we")
        self._tooltip(export, "export")

    def _build_preview(self, parent):
        top = ttk.Frame(parent)
        top.grid(row=0, column=0, sticky="we")
        self.rotate_var = tk.BooleanVar(value=True)
        rotate = ttk.Checkbutton(
            top, text="Slow auto-rotate", variable=self.rotate_var,
            command=self._on_rotate_toggle,
        )
        rotate.pack(side="left")
        self._tooltip(rotate, "auto_rotate")
        opacity_label = ttk.Label(top, text="Opacity")
        opacity_label.pack(side="left", padx=(16, 4))
        self.opacity_var = tk.DoubleVar(value=1.0)
        opacity = ttk.Scale(top, from_=0.2, to=1.0, variable=self.opacity_var, length=120,
                            command=lambda _value: self._recolor_mesh(force=True))
        opacity.pack(side="left")
        self._tooltip(opacity, "opacity")
        legend = ttk.Label(top, text="light: outside, dark: inside, orange: cavity walls",
                           foreground="#555555")
        legend.pack(side="left", padx=(12, 0))
        self._tooltip(legend, "surface_sides")

        self.fig = Figure(figsize=(6, 5), dpi=100)
        self.ax = self.fig.add_subplot(111, projection="3d")
        self.ax.set_box_aspect([1, 1, 1])
        self.canvas = FigureCanvasTkAgg(self.fig, master=parent)
        self._mesh_collection = None
        self._mesh_view = None
        self.canvas.mpl_connect("draw_event", self._on_draw)
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
        units = self.units_var.get()
        try:
            geo = load_geometry(path)
            nodes, scale = scale_nodes(geo.nodes, units)
            solid = prepare_solid(nodes, geo.faces, cavities=self.cavities_var.get())
        except Exception as exc:
            messagebox.showerror("Load error", str(exc))
            return

        self.geo = geo
        self.raw_nodes, self.raw_faces = nodes, geo.faces
        self.nodes, self.faces = solid.nodes, solid.faces
        self.cavity_mask = solid.cavity_faces
        nodes, faces = self.nodes, self.faces
        self.loaded_units, self.loaded_scale = units, scale
        self.metrics = []
        self._clear_results()
        self._refresh_units_summary()

        info = (
            f"{geo.source_path.name}\n"
            f"format: {geo.source_format.upper()}"
            + (" (converted to a triangle mesh)" if geo.converted_from_native else "")
            + f"\nreader: {reader_text(geo.reader, geo.reader_version)}"
            + f"\nvertices: {nodes.shape[0]}  faces: {faces.shape[0]}"
            + "".join(f"\nWarning: {text}" for text in geo.warnings)
        )
        self.file_info_var.set(info)
        self._update_export_preview()
        self._render_mesh()
        self._start_rotation()
        if geo.warnings:
            messagebox.showwarning("Read with a fallback", "\n\n".join(geo.warnings))

    def _prepare_loaded(self):
        """Prepare the loaded mesh again after the nested-shell choice changed."""
        if getattr(self, "raw_nodes", None) is None:
            return
        try:
            solid = prepare_solid(self.raw_nodes, self.raw_faces, cavities=self.cavities_var.get())
        except Exception as exc:
            messagebox.showerror("Preparation error", str(exc))
            return
        self.nodes, self.faces, self.cavity_mask = solid.nodes, solid.faces, solid.cavity_faces
        self.metrics = []
        self._clear_results()
        self._render_mesh()

    def _output_choice(self) -> str:
        choice = self.output_units_var.get()
        return "same" if choice == SAME_AS_INPUT else choice

    def _sync_output_state(self):
        """Output units are meaningless for as-is input: disable the selector."""
        state = "disabled" if self.units_var.get() == "as-is" else "readonly"
        self.output_units_combo.configure(state=state)

    def _refresh_units_summary(self):
        if self.nodes is None:
            self.units_summary_var.set("")
            return
        if self.metrics:
            effective = self.output_effective  # units of the results on screen
        else:
            effective = resolve_output_units(self.loaded_units, self._output_choice())
        text = units_summary(self.loaded_units, self.loaded_scale, effective)
        if self.metrics and effective != resolve_output_units(
            self.loaded_units, self._output_choice()
        ):
            text += "  (recalculate to apply new output units)"
        self.units_summary_var.set(text)

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
            self.metrics = compute_metrics(self.raw_nodes, self.raw_faces, axes=self.axes_var.get(),
                                           cavities=self.cavities_var.get())
        except Exception as exc:
            messagebox.showerror("Compute error", str(exc))
            return
        self.output_effective, _ = apply_output_units(
            self.metrics, self.loaded_units, self._output_choice(),
        )
        self._refresh_units_summary()
        self._populate_results()

    def _populate_results(self):
        self._clear_results()
        current_group = None
        r = 0
        for metric in self.metrics:
            if metric.group != current_group:
                current_group = metric.group
                heading = ttk.Label(
                    self.results_frame, text=current_group, font=("TkDefaultFont", 9, "bold")
                )
                heading.grid(row=r, column=0, columnspan=2, sticky="w", pady=(6 if r else 0, 2))
                Tooltip(heading, descriptors.group_help(current_group))
                r += 1
            var = tk.BooleanVar(value=metric.default_selected and metric.value is not None)
            self.metric_vars[metric.key] = var
            cb = ttk.Checkbutton(
                self.results_frame,
                text=f"{metric.label} = {metric.text}"
                + (f" {metric.unit}" if metric.unit and metric.value is not None else ""),
                variable=var,
                state=("normal" if metric.value is not None else "disabled"),
            )
            cb.grid(row=r, column=1, sticky="w")
            self._tooltip(cb, metric.key, kind="metrics")
            entry = descriptors.metric_help(metric.key)
            if entry is not None:
                self._help_button(self.results_frame, metric.label, entry).grid(
                    row=r, column=0, sticky="w", padx=(0, 4))
            r += 1

    # ------------------------------------------------------------------
    # Help (texts from pyshape.descriptors, keyed by metric key or control name)
    # ------------------------------------------------------------------
    def _tooltip(self, widget: tk.Widget, key: str, kind: str = "controls") -> None:
        Tooltip(widget, _short(kind, key))

    def _help_button(self, parent, title: str, entry: dict, name: str = "") -> ttk.Label:
        """A small "?" that opens the long explanation (click, or Enter/space when focused)."""
        button = ttk.Label(parent, text="?", foreground="#17629b", cursor="hand2",
                           takefocus=True, font=("TkDefaultFont", 9, "bold"))
        if name == "axes":
            text = descriptors.axes_control_explanation(AXES_METHODS, title)
        else:
            text = descriptors.explanation(entry, title)
        for sequence in ("<Button-1>", "<Return>", "<space>"):
            button.bind(sequence, lambda e: self._show_help(title, text))
        Tooltip(button, "Click for the definition, formula and references.")
        return button

    def _help_label(self, parent, text: str, name: str) -> ttk.Frame:
        """Label of a control with a tooltip and a "?" button; returned for gridding."""
        frame = ttk.Frame(parent)
        label = ttk.Label(frame, text=text)
        label.pack(side="left")
        self._tooltip(label, name)
        entry = descriptors.control_help(name)
        if entry is not None:
            self._help_button(frame, text, entry, name).pack(side="left", padx=(4, 0))
        return frame

    def _show_help(self, title: str, text: str) -> None:
        """Open (or replace) the help window; links open in the system browser."""
        old = getattr(self, "_help_window", None)
        if old is not None and old.winfo_exists():
            old.destroy()
        window = self._help_window = tk.Toplevel(self)
        window.title(f"Help: {title}")
        window.transient(self)
        body = tk.Text(window, wrap="word", width=72, height=20, padx=10, pady=8,
                       relief="flat", font=("TkDefaultFont", 10))
        scrollbar = ttk.Scrollbar(window, orient="vertical", command=body.yview)
        body.configure(yscrollcommand=scrollbar.set)
        body.grid(row=0, column=0, sticky="nsew")
        scrollbar.grid(row=0, column=1, sticky="ns")
        window.rowconfigure(0, weight=1)
        window.columnconfigure(0, weight=1)
        position = 0
        for n, match in enumerate(re.finditer(r"https?://\S+", text)):
            body.insert("end", text[position:match.start()])
            tag = f"link{n}"
            body.insert("end", match.group(), ("link", tag))
            body.tag_bind(tag, "<Button-1>", lambda e, url=match.group(): webbrowser.open(url))
            position = match.end()
        body.insert("end", text[position:])
        body.tag_add("title", "1.0", "1.0 lineend")
        body.tag_configure("title", font=("TkDefaultFont", 12, "bold"))
        body.tag_configure("link", foreground="#17629b", underline=True)
        body.tag_bind("link", "<Enter>", lambda e: body.configure(cursor="hand2"))
        body.tag_bind("link", "<Leave>", lambda e: body.configure(cursor=""))
        body.configure(state="disabled")
        ttk.Button(window, text="Close", command=window.destroy).grid(
            row=1, column=0, columnspan=2, pady=6)
        window.bind("<Escape>", lambda e: window.destroy())
        window.focus_set()

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
            (m.label, m.value,
             unit_label(m.key, self.loaded_units, ascii=True, output_units=self.output_effective))
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
            write_csv(out_path, selected, source_name=self.geo.source_path.name,
                      metadata=geo_metadata_rows(self.geo))
        except Exception as exc:
            messagebox.showerror("Export error", str(exc))
            return
        messagebox.showinfo("Exported", f"Wrote {len(selected)} parameter(s) to:\n{out_path}")

    # ------------------------------------------------------------------
    # 3D preview + rotation
    # ------------------------------------------------------------------
    def _render_mesh(self):
        self.ax.clear()
        self._mesh_collection = None
        self.ax.set_title("Geometry preview")
        if self.nodes is None or self.faces is None:
            self.canvas.draw()
            return
        verts = [self.nodes[idx] for idx in self.faces]
        # Edges help on coarse meshes but turn dense meshes black; shade instead.
        dense = len(self.faces) > EDGE_FACE_LIMIT
        self._normals = face_normals(self.nodes, self.faces)
        self._mesh_view = None
        self._mesh_collection = Poly3DCollection(
            verts, facecolors=self._face_colors(),
            edgecolors="none" if dense else (0, 0, 0, 0.6), linewidths=0.3,
        )
        self.ax.add_collection3d(self._mesh_collection)

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
        self._mesh_collection.set_facecolor(self._face_colors())
        self.canvas.draw()

    def _face_colors(self):
        """Colours for the current view: outer side light blue, inner side darker blue."""
        self._mesh_view = (float(self.ax.elev), float(self.ax.azim))
        return preview_face_colors(self._normals, view_direction(*self._mesh_view),
                                   self.opacity_var.get(), getattr(self, "cavity_mask", None))

    def _recolor_mesh(self, force: bool = False):
        """Recolour the faces when the view (or the opacity) has changed."""
        collection = getattr(self, "_mesh_collection", None)
        if collection is None or self.nodes is None:
            return
        view = (float(self.ax.elev), float(self.ax.azim))
        if force or view != self._mesh_view:
            collection.set_facecolor(self._face_colors())
            self.canvas.draw_idle()

    def _on_draw(self, _event):
        # Dragging changes the view inside Matplotlib; recolour after such a draw.
        if getattr(self, "_mesh_collection", None) is not None and self._mesh_view != (
                float(self.ax.elev), float(self.ax.azim)):
            self.after_idle(self._recolor_mesh)

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
            self._recolor_mesh()
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
