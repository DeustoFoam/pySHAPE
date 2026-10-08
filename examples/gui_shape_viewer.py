import os
from pathlib import Path
import sys
import tkinter as tk
from tkinter import ttk, filedialog, messagebox

import numpy as np

# Make package importable when running from repo
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from pyshape import (
    load_stl,
    surface_area,
    surface_orientation_tensor,
    sphericity_wadell,
    sphericity_krumbein,
    form_parameters_potticary_et_al,
    form_parameters_kong_and_fonseca,
    form_parameters_zingg,
)


# Optional: drag-and-drop via tkinterdnd2 if installed
try:
    from tkinterdnd2 import DND_FILES, TkinterDnD  # type: ignore
    HAS_DND = True
except Exception:
    DND_FILES = None
    TkinterDnD = tk.Tk  # fallback
    HAS_DND = False


# Matplotlib 3D embed
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from matplotlib.figure import Figure
from mpl_toolkits.mplot3d.art3d import Poly3DCollection


def compute_volume_from_tri_mesh(nodes: np.ndarray, faces: np.ndarray) -> float:
    """Compute signed volume of a closed triangle mesh.
    Uses divergence theorem; positive if outward-facing normals.
    """
    v0 = nodes[faces[:, 0]]
    v1 = nodes[faces[:, 1]]
    v2 = nodes[faces[:, 2]]
    return float(np.einsum('ij,ij->i', v0, np.cross(v1, v2)).sum() / 6.0)


class ShapeGUI(TkinterDnD):
    def __init__(self):
        super().__init__()
        self.title("SHAPE (Python) Viewer")
        self.geometry("1100x700")

        # State
        self.nodes = None
        self.faces = None
        self.current_file = None

        self._build_ui()

    def _build_ui(self):
        # Layout: left controls, right 3D view
        self.columnconfigure(0, weight=0)
        self.columnconfigure(1, weight=1)
        self.rowconfigure(0, weight=1)

        left = ttk.Frame(self, padding=8)
        left.grid(row=0, column=0, sticky="nsw")
        right = ttk.Frame(self, padding=0)
        right.grid(row=0, column=1, sticky="nsew")
        right.rowconfigure(0, weight=1)
        right.columnconfigure(0, weight=1)

        # File controls
        ttk.Label(left, text="STL file").grid(row=0, column=0, sticky="w")
        self.path_var = tk.StringVar()
        path_entry = ttk.Entry(left, textvariable=self.path_var, width=38)
        path_entry.grid(row=1, column=0, columnspan=2, sticky="we", pady=(0,4))
        ttk.Button(left, text="Browse…", command=self.browse_file).grid(row=1, column=2, padx=(6,0))

        # Units scaling
        ttk.Label(left, text="Units").grid(row=2, column=0, sticky="w", pady=(6,0))
        self.units_var = tk.StringVar(value="auto")
        self.units_combo = ttk.Combobox(left, textvariable=self.units_var, state="readonly", width=10,
                                        values=["auto", "as-is", "mm", "cm", "m", "inch"]) 
        self.units_combo.grid(row=2, column=1, sticky="w", pady=(6,0))

        # Drag-and-drop area
        drop_text = "Drag & drop STL here" + (" (DnD enabled)" if HAS_DND else " (install tkinterdnd2 for DnD)")
        self.drop_area = tk.Label(left, text=drop_text, relief=tk.GROOVE, width=46, height=3)
        self.drop_area.grid(row=3, column=0, columnspan=3, sticky="we", pady=4)
        if HAS_DND:
            self.drop_area.drop_target_register(DND_FILES)
            self.drop_area.dnd_bind('<<Drop>>', self.on_drop)

        ttk.Button(left, text="Load", command=self.load_mesh).grid(row=4, column=0, sticky="we", pady=(6,8))

        # Metrics checkboxes
        ttk.Label(left, text="Metrics").grid(row=5, column=0, sticky="w")
        self.var_area = tk.BooleanVar(value=True)
        self.var_volume = tk.BooleanVar(value=True)
        self.var_wadell = tk.BooleanVar(value=True)
        self.var_orient = tk.BooleanVar(value=True)
        self.var_convexity = tk.BooleanVar(value=False)
        self.var_spK = tk.BooleanVar(value=False)
        self.var_params_p = tk.BooleanVar(value=False)  # Potticary
        self.var_params_kf = tk.BooleanVar(value=False) # Kong & Fonseca
        self.var_params_z = tk.BooleanVar(value=False)  # Zingg ratios
        self.var_show_axes = tk.BooleanVar(value=False) # show S/I/L

        ttk.Checkbutton(left, text="Surface area", variable=self.var_area).grid(row=6, column=0, sticky="w")
        ttk.Checkbutton(left, text="Volume (triangle mesh)", variable=self.var_volume).grid(row=7, column=0, sticky="w")
        ttk.Checkbutton(left, text="Wadell sphericity", variable=self.var_wadell).grid(row=8, column=0, sticky="w")
        ttk.Checkbutton(left, text="Orientation (C,F,R)", variable=self.var_orient).grid(row=9, column=0, sticky="w")
        ttk.Separator(left, orient='horizontal').grid(row=10, column=0, columnspan=3, sticky='we', pady=(6,6))
        ttk.Checkbutton(left, text="Convexity (V/V_CH)", variable=self.var_convexity).grid(row=11, column=0, sticky="w")
        ttk.Checkbutton(left, text="Krumbein sphericity (S,I,L)", variable=self.var_spK).grid(row=12, column=0, sticky="w")
        ttk.Checkbutton(left, text="Form params: Potticary", variable=self.var_params_p).grid(row=13, column=0, sticky="w")
        ttk.Checkbutton(left, text="Form params: Kong & Fonseca", variable=self.var_params_kf).grid(row=14, column=0, sticky="w")
        ttk.Checkbutton(left, text="Zingg ratios (S/I, I/L)", variable=self.var_params_z).grid(row=15, column=0, sticky="w")
        ttk.Checkbutton(left, text="Show estimated axes S,I,L (PCA)", variable=self.var_show_axes).grid(row=16, column=0, sticky="w")

        ttk.Button(left, text="Calculate", command=self.calculate_metrics).grid(row=17, column=0, sticky="we", pady=(10,4))

        # Output box
        ttk.Label(left, text="Results").grid(row=18, column=0, sticky="w")
        self.output = tk.Text(left, width=48, height=18)
        self.output.grid(row=19, column=0, columnspan=3, sticky="we")
        ttk.Button(left, text="Copy", command=self.copy_results).grid(row=20, column=0, sticky="w", pady=(4,0))

        # Matplotlib figure for 3D
        self.fig = Figure(figsize=(6, 5), dpi=100)
        self.ax = self.fig.add_subplot(111, projection='3d')
        self.ax.set_box_aspect([1,1,1])
        self.canvas = FigureCanvasTkAgg(self.fig, master=right)
        self.canvas.draw()
        self.canvas.get_tk_widget().grid(row=0, column=0, sticky="nsew")
        toolbar = NavigationToolbar2Tk(self.canvas, right, pack_toolbar=False)
        toolbar.update()
        toolbar.grid(row=1, column=0, sticky="we")

    def on_drop(self, event):
        path = event.data
        # Handle braces if provided by tk on Windows
        path = path.strip('{}')
        self.path_var.set(path)

    def browse_file(self):
        filetypes = [("STL files", "*.stl"), ("All files", "*.*")]
        initial = str(Path.cwd())
        filename = filedialog.askopenfilename(title="Open STL", initialdir=initial, filetypes=filetypes)
        if filename:
            self.path_var.set(filename)

    def load_mesh(self):
        path = self.path_var.get().strip()
        if not path:
            messagebox.showwarning("No file", "Please specify an STL file.")
            return
        p = Path(path)
        if not p.exists():
            messagebox.showerror("Not found", f"File not found:\n{p}")
            return
        try:
            nodes, faces = load_stl(p)
        except Exception as e:
            messagebox.showerror("Load error", str(e))
            return
        self.current_file = str(p)
        # Units scaling (with simple auto heuristic)
        units = self.units_var.get()
        if units == "auto":
            # Heuristic: guess mm if object is large in raw units
            mins = nodes.min(axis=0)
            maxs = nodes.max(axis=0)
            span = float((maxs - mins).max())
            if span > 10000:      # likely micrometers
                scale = 1e-6
            elif span > 10:       # likely millimeters
                scale = 1e-3
            else:
                scale = 1.0
        else:
            scale = {"as-is": 1.0, "mm": 1e-3, "cm": 1e-2, "m": 1.0, "inch": 0.0254}.get(units, 1.0)
        if scale != 1.0:
            nodes = nodes * scale
        self.nodes, self.faces = nodes, faces
        self._render_mesh()

    def _render_mesh(self):
        self.ax.clear()
        self.ax.set_title("Mesh preview (rotate/zoom with mouse)")
        if self.nodes is None or self.faces is None:
            self.canvas.draw()
            return
        verts = [self.nodes[idx] for idx in self.faces]
        coll = Poly3DCollection(verts, facecolors='lightsteelblue', edgecolors='k', linewidths=0.5, alpha=0.9)
        self.ax.add_collection3d(coll)
        # Auto-scale
        mins = self.nodes.min(axis=0)
        maxs = self.nodes.max(axis=0)
        center = (mins + maxs) / 2.0
        size = (maxs - mins).max()
        if size <= 0:
            size = 1.0
        lims = np.array([center - size/2, center + size/2]).T
        self.ax.set_xlim(*lims[0])
        self.ax.set_ylim(*lims[1])
        self.ax.set_zlim(*lims[2])
        self.ax.set_box_aspect([1,1,1])
        self.canvas.draw()

    def calculate_metrics(self):
        if self.nodes is None or self.faces is None:
            messagebox.showwarning("No mesh", "Load an STL first.")
            return
        out = []
        if self.current_file:
            out.append(f"file = {Path(self.current_file).name}")
        try:
            if self.var_area.get():
                A = surface_area(self.nodes, self.faces)
                out.append(f"surface_area = {A:.6f}")
            if self.var_volume.get() or self.var_wadell.get():
                V = abs(compute_volume_from_tri_mesh(self.nodes, self.faces))
                if self.var_volume.get():
                    out.append(f"volume = {V:.6f}")
            if self.var_wadell.get():
                if 'A' not in locals():
                    A = surface_area(self.nodes, self.faces)
                if 'V' not in locals():
                    V = abs(compute_volume_from_tri_mesh(self.nodes, self.faces))
                phi = sphericity_wadell(V, A)
                out.append(f"sphericity_wadell = {phi:.6f}")
            if self.var_orient.get():
                C, F, R, vals, vecs = surface_orientation_tensor(self.nodes, self.faces)
                out.append(f"orientation C={C:.6f}, F={F:.6f}, R={R:.6f}")
                out.append(f"eigenvalues = [{vals[0]:.6f}, {vals[1]:.6f}, {vals[2]:.6f}]")

            # PCA-based axes estimation if any axis-dependent metrics are requested
            need_axes = self.var_spK.get() or self.var_params_p.get() or self.var_params_kf.get() or self.var_params_z.get() or self.var_show_axes.get()
            if need_axes:
                # Estimate S,I,L using PCA extents
                X = self.nodes - self.nodes.mean(axis=0)
                # Principal directions (columns of V)
                Vt = np.linalg.svd(X, full_matrices=False)[2]
                coords = X @ Vt.T
                spans = coords.max(axis=0) - coords.min(axis=0)
                # Sort to get L >= I >= S
                order = np.argsort(spans)
                S_len, I_len, L_len = spans[order[0]], spans[order[1]], spans[order[2]]
                if self.var_show_axes.get():
                    out.append(f"axes_estimated (S,I,L) = [{S_len:.6f}, {I_len:.6f}, {L_len:.6f}]")

                if self.var_spK.get():
                    spK = sphericity_krumbein(S=S_len, I=I_len, L=L_len)
                    out.append(f"sphericity_krumbein = {spK:.6f}")
                if self.var_params_p.get():
                    flP, elP = form_parameters_potticary_et_al(S_len, I_len, L_len)
                    out.append(f"Potticary: flatness={flP:.6f}, elongation={elP:.6f}")
                if self.var_params_kf.get():
                    flK, elK = form_parameters_kong_and_fonseca(S_len, I_len, L_len)
                    out.append(f"Kong&Fonseca: flatness={flK:.6f}, elongation={elK:.6f}")
                if self.var_params_z.get():
                    SIZ, ILZ = form_parameters_zingg(S_len, I_len, L_len)
                    out.append(f"Zingg: S/I={SIZ:.6f}, I/L={ILZ:.6f}")

            if self.var_convexity.get():
                # Prefer SciPy convex hull if available; otherwise provide install hint
                try:
                    from scipy.spatial import ConvexHull  # type: ignore
                    hull = ConvexHull(self.nodes)
                    Vch = float(hull.volume)
                    if 'V' not in locals():
                        V = abs(compute_volume_from_tri_mesh(self.nodes, self.faces))
                    if Vch > 0:
                        out.append(f"convexity = {V / Vch:.6f} (V={V:.6f}, V_CH={Vch:.6f})")
                    else:
                        out.append("convexity: convex hull volume is zero")
                except ModuleNotFoundError:
                    out.append("convexity: SciPy not installed; run 'pip install scipy' to enable")
                except Exception:
                    # Fallback: try trimesh convex hull
                    try:
                        import trimesh
                        mesh = trimesh.Trimesh(vertices=self.nodes, faces=self.faces, process=False)
                        Vch = float(mesh.convex_hull.volume)
                        if 'V' not in locals():
                            V = abs(compute_volume_from_tri_mesh(self.nodes, self.faces))
                        if Vch > 0:
                            out.append(f"convexity = {V / Vch:.6f} (V={V:.6f}, V_CH={Vch:.6f})")
                        else:
                            out.append("convexity: convex hull volume is zero")
                    except Exception as e2:
                        out.append(f"convexity: error computing convex hull ({e2})")
        except Exception as e:
            messagebox.showerror("Compute error", str(e))
            return
        self.output.delete("1.0", tk.END)
        self.output.insert(tk.END, "\n".join(out))

    def copy_results(self):
        text = self.output.get("1.0", tk.END)
        self.clipboard_clear()
        self.clipboard_append(text)
        self.update()  # keep on clipboard after window close


if __name__ == '__main__':
    app = ShapeGUI()
    app.mainloop()
