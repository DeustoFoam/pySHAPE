"use strict";

const $ = (id) => document.getElementById(id);
let analysisFile = null;
let analysisUnits = "as-is";
let analysisAxes = "pca";
let analysisOutput = "same";
let analysisCavities = "fill";
let busy = false;
let triangles = [];
let cavityTriangles = [];
let yaw = -0.6, pitch = 0.4, zoom = 1;
let pointer = null, lastFrame = 0;
const canvas = $("preview");
const ctx = canvas.getContext("2d");

function status(message, error = false) {
  $("status").textContent = message;
  $("status").classList.toggle("error", error);
}

// Same names as pyshape.gui_core.reader_text.
function readerText(result) {
  const version = result.reader_version ? ` ${result.reader_version}` : "";
  return {
    "builtin-stl": "built-in STL parser",
    "trimesh": `trimesh${version}`,
    "trimesh-fallback": `trimesh${version} (fallback: the built-in STL parser rejected the file)`,
  }[result.reader] || result.reader || "unknown";
}

function setBusy(value) {
  busy = value;
  $("analyze").disabled = value;
  $("export").disabled = value || !analysisFile;
}

async function post(path, data) {
  const response = await fetch(path, { method: "POST", body: data });
  if (!response.ok) {
    const body = await response.json();
    throw new Error(typeof body.detail === "string" ? body.detail : JSON.stringify(body.detail));
  }
  return response;
}

function renderResults(metrics) {
  $("results").replaceChildren();
  for (const group of new Set(metrics.map((metric) => metric.group))) {
    const heading = document.createElement("tr");
    heading.className = "group";
    const cell = document.createElement("th");
    cell.colSpan = 4;
    cell.textContent = group;
    cell.dataset.group = group;
    heading.append(cell);
    $("results").append(heading);
    for (const metric of metrics.filter((item) => item.group === group)) {
      const row = document.createElement("tr");
      const selection = document.createElement("td");
      const check = document.createElement("input");
      check.type = "checkbox";
      check.value = metric.key;
      check.setAttribute("aria-label", `Export ${metric.label}`);
      check.disabled = metric.value === null;
      check.checked = !check.disabled && metric.default_selected;
      selection.append(check);
      const label = document.createElement("td");
      const name = document.createElement("button");
      name.type = "button";
      name.className = "metric-name";
      name.dataset.metric = metric.key;  // help is keyed by metric key (pyshape.descriptors)
      name.textContent = metric.label;
      label.append(name);
      const value = document.createElement("td");
      value.textContent = metric.value === null
        ? `n/a (${metric.error || "unavailable"})`
        : Number.isInteger(metric.value) ? String(metric.value) : Number(metric.value).toPrecision(6);
      value.className = metric.value === null ? "value unavailable" : "value";
      const unit = document.createElement("td");
      unit.className = "unit";
      unit.textContent = metric.value === null ? "" : (metric.unit || "");
      row.append(selection, label, value, unit);
      $("results").append(row);
    }
  }
  pyshapeHelp.refresh($("results"));
}

$("analyze-form").addEventListener("submit", async (event) => {
  event.preventDefault();
  if (busy) return;
  const file = $("file").files[0];
  if (!file) return;
  if (file.size > 32 * 1024 * 1024) {
    status("The file exceeds the 32 MiB limit.", true);
    return;
  }
  analysisFile = null;
  triangles = [];
  cavityTriangles = [];
  $("results").replaceChildren();
  $("metadata").textContent = "";
  $("preview-info").textContent = "Drag to rotate. Scroll to zoom.";
  draw();
  setBusy(true);
  status("Calculating...");
  const units = $("units").value;
  const axes = $("axes").value;
  const data = new FormData();
  data.append("file", file);
  const outputUnits = $("output-units").value;
  data.append("units", units);
  data.append("axes", axes);
  const cavities = $("cavities").value;
  data.append("cavities", cavities);
  data.append("output_units", outputUnits);
  data.append("include_preview", "true");
  try {
    const result = await (await post("/api/analyze", data)).json();
    renderResults(result.metrics);
    analysisFile = file;
    analysisUnits = units;
    analysisAxes = axes;
    analysisOutput = outputUnits;
    triangles = result.preview.triangles;
    cavityTriangles = result.preview.cavity_triangles || [];
    analysisCavities = cavities;
    // Cavities are inside the material: make the surface see-through to show them.
    if (cavityTriangles.length && Number($("opacity").value) > 0.6) $("opacity").value = "0.45";
    zoom = 1;
    const unitNames = { m: "m", mm: "mm", um: "µm", cm: "cm", inch: "in" };
    const out = unitNames[result.output_units];
    $("metadata").textContent = `${result.source_name} | Input: ${result.input_units}`
      + (out ? ` | Output: ${result.output_units}` : "")
      + ` | Axes: ${({pca: "PCA", min_volume: "minimum-volume box", inertia_ellipsoid: "inertia-equivalent ellipsoid", fitted_ellipsoid: "fitted ellipsoid"})[result.axes_method] || result.axes_method}. `
      + (out ? `Results in ${out} (area: ${out}², volume: ${out}³, inertia: ${out}⁵).`
             : "Results use the input units of the file.");
    $("preview-info").textContent = `Showing ${result.preview.shown_faces} of ${result.preview.total_faces} faces. `
      + "Large meshes are simplified for display only; calculations always use the full mesh. "
      + "Drag to rotate, scroll to zoom.";
    $("metadata").textContent += ` Read by: ${readerText(result)}.`;
    const failed = result.metrics.filter((metric) => metric.error).length;
    status((failed ? `Analysis complete. ${failed} parameters unavailable; see results.` : "Analysis complete.")
      + result.warnings.map((text) => ` Warning: ${text}`).join(""), result.warnings.length > 0);
    draw();
  } catch (error) {
    status(error.message, true);
  } finally {
    setBusy(false);
  }
});

for (const [id, selected] of [["select-all", true], ["select-none", false]]) {
  $(id).addEventListener("click", () => {
    for (const check of $("results").querySelectorAll("input:not(:disabled)")) {
      check.checked = selected;
    }
  });
}

$("export").addEventListener("click", async () => {
  if (busy || !analysisFile) return;
  const keys = [...$("results").querySelectorAll("input:checked")].map((check) => check.value);
  if (!keys.length) {
    status("Select at least one available result to export.", true);
    return;
  }
  const data = new FormData();
  data.append("file", analysisFile);
  data.append("units", analysisUnits);
  data.append("axes", analysisAxes);
  data.append("cavities", analysisCavities);
  data.append("output_units", analysisOutput);
  data.append("keys", keys.join(","));
  setBusy(true);
  status("Preparing CSV...");
  try {
    const response = await post("/api/export", data);
    const url = URL.createObjectURL(await response.blob());
    const link = document.createElement("a");
    link.href = url;
    const suffix = $("suffix").value.replace(/[<>:"/\\|?*\x00-\x1f]/g, "_");
    link.download = analysisFile.name.replace(/\.[^./\\]+$/, "") + suffix + ".csv";
    document.body.append(link);
    link.click();
    link.remove();
    setTimeout(() => URL.revokeObjectURL(url), 1000);
    status("CSV download ready.");
  } catch (error) {
    status(error.message, true);
  } finally {
    setBusy(false);
  }
});

function project(point) {
  const [x, y, z] = point;
  const a = Math.cos(yaw) * x - Math.sin(yaw) * y;
  const b = Math.sin(yaw) * x + Math.cos(yaw) * y;
  const depth = Math.cos(pitch) * b - Math.sin(pitch) * z;
  const vertical = Math.sin(pitch) * b + Math.cos(pitch) * z;
  const scale = Math.min(canvas.width, canvas.height) * 0.72 * zoom;
  return [canvas.width / 2 + a * scale, canvas.height / 2 - vertical * scale, depth];
}

function draw() {
  const ratio = window.devicePixelRatio || 1;
  const width = Math.round(canvas.clientWidth * ratio);
  const height = Math.round(canvas.clientHeight * ratio);
  if (canvas.width !== width || canvas.height !== height) {
    canvas.width = width;
    canvas.height = height;
  }
  ctx.clearRect(0, 0, canvas.width, canvas.height);
  // Each entry: projected corners, plus a flag for cavity walls.
  const projected = triangles.map((triangle) => triangle.map(project))
    .concat(cavityTriangles.map((triangle) => Object.assign(triangle.map(project), { cavity: true })));
  projected.sort((a, b) => a.reduce((s, p) => s + p[2], 0) - b.reduce((s, p) => s + p[2], 0));
  ctx.lineWidth = 0.4 * ratio;
  const opacity = Number($("opacity").value);
  ctx.globalAlpha = opacity;
  for (const points of projected) {
    const depth = points.reduce((sum, point) => sum + point[2], 0) / 3;
    const light = Math.max(40, Math.min(82, 63 + depth * 18));
    // Seen from outside, an outward-oriented triangle has a positive signed area
    // in canvas coordinates (y pointing down); otherwise its inner side is visible.
    const [p0, p1, p2] = points;
    const outside = (p1[0] - p0[0]) * (p2[1] - p0[1]) - (p1[1] - p0[1]) * (p2[0] - p0[0]) >= 0;
    // Inner side: a darker shade of the same blue (orange is kept for nested shells).
    if (points.cavity) {
      // Cavity walls (nested shells) are orange; darker when seen from behind.
      ctx.fillStyle = outside ? `hsl(28, 75%, ${light}%)` : `hsl(25, 50%, ${light * 0.6}%)`;
      ctx.strokeStyle = "#8a5a2b";
    } else {
      ctx.fillStyle = outside ? `hsl(209, 40%, ${light}%)` : `hsl(212, 28%, ${light * 0.6}%)`;
      ctx.strokeStyle = outside ? "#516b86" : "#2f3e50";
    }
    ctx.beginPath();
    points.forEach((point, index) => index ? ctx.lineTo(point[0], point[1]) : ctx.moveTo(point[0], point[1]));
    ctx.closePath();
    ctx.fill();
    ctx.stroke();
  }
  ctx.globalAlpha = 1;
  if (triangles.length) {
    const origin = project([0, 0, 0]);
    for (const [axis, color, point] of [
      ["X", "#b22525", [0.7, 0, 0]], ["Y", "#237538", [0, 0.7, 0]], ["Z", "#235cb0", [0, 0, 0.7]],
    ]) {
      const end = project(point);
      ctx.strokeStyle = color;
      ctx.fillStyle = color;
      ctx.lineWidth = 2 * ratio;
      ctx.beginPath();
      ctx.moveTo(origin[0], origin[1]);
      ctx.lineTo(end[0], end[1]);
      ctx.stroke();
      ctx.font = `${14 * ratio}px system-ui`;
      ctx.fillText(axis, end[0] + 4 * ratio, end[1]);
    }
  }
}

$("opacity").addEventListener("input", () => draw());

canvas.addEventListener("pointerdown", (event) => {
  pointer = { x: event.clientX, y: event.clientY };
  canvas.setPointerCapture(event.pointerId);
});
canvas.addEventListener("pointermove", (event) => {
  if (!pointer) return;
  yaw -= (event.clientX - pointer.x) * 0.01;
  pitch = Math.max(-1.5, Math.min(1.5, pitch - (event.clientY - pointer.y) * 0.01));
  pointer = { x: event.clientX, y: event.clientY };
  draw();
});
canvas.addEventListener("pointerup", () => { pointer = null; });
canvas.addEventListener("pointercancel", () => { pointer = null; });
canvas.addEventListener("wheel", (event) => {
  event.preventDefault();
  zoom = Math.max(0.2, Math.min(5, zoom * Math.exp(-event.deltaY * 0.001)));
  draw();
}, { passive: false });
window.addEventListener("resize", draw);
$("rotate").addEventListener("change", draw);
function animate(time) {
  if (time - lastFrame >= 40) {
    if (triangles.length && $("rotate").checked && !pointer && !document.hidden) {
      yaw += 0.01;
      draw();
    }
    lastFrame = time;
  }
  requestAnimationFrame(animate);
}
draw();
requestAnimationFrame(animate);

function syncOutputUnits() {
  $("output-units").disabled = $("units").value === "as-is";
}
$("units").addEventListener("change", syncOutputUnits);
syncOutputUnits();
