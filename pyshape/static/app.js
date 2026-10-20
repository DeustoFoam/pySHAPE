"use strict";

const $ = (id) => document.getElementById(id);
let analysisFile = null;
let analysisUnits = "as-is";
let busy = false;
let triangles = [];
let yaw = -0.6, pitch = 0.4, zoom = 1;
let pointer = null, lastFrame = 0;
const canvas = $("preview");
const ctx = canvas.getContext("2d");

function status(message, error = false) {
  $("status").textContent = message;
  $("status").classList.toggle("error", error);
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
    cell.colSpan = 3;
    cell.textContent = group;
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
      label.textContent = metric.label;
      const value = document.createElement("td");
      value.textContent = metric.value === null
        ? `n/a (${metric.error || "unavailable"})`
        : Number(metric.value).toPrecision(6);
      if (metric.value === null) value.className = "unavailable";
      row.append(selection, label, value);
      $("results").append(row);
    }
  }
}

$("analyze-form").addEventListener("submit", async (event) => {
  event.preventDefault();
  if (busy) return;
  const file = $("file").files[0];
  if (!file) return;
  if (file.size > 32 * 1024 * 1024) {
    status("STL upload exceeds the 32 MiB limit.", true);
    return;
  }
  analysisFile = null;
  triangles = [];
  $("results").replaceChildren();
  $("metadata").textContent = "";
  $("preview-info").textContent = "Drag to rotate. Scroll to zoom.";
  draw();
  setBusy(true);
  status("Calculating...");
  const units = $("units").value;
  const data = new FormData();
  data.append("file", file);
  data.append("units", units);
  data.append("include_preview", "true");
  try {
    const result = await (await post("/api/analyze", data)).json();
    renderResults(result.metrics);
    analysisFile = file;
    analysisUnits = units;
    triangles = result.preview.triangles;
    zoom = 1;
    $("metadata").textContent = `${result.source_name} | Input: ${result.input_units} | Scale: ${result.scale_factor}. `
      + (units === "as-is" ? "Results use input length units." : "Results use metres (area: m², volume: m³).")
      + (units === "auto" ? " Units were guessed using the legacy size heuristic." : "");
    $("preview-info").textContent = `Showing ${result.preview.shown_faces} of ${result.preview.total_faces} faces. `
      + "Preview may be sampled; calculations always use the full mesh. Drag to rotate, scroll to zoom.";
    const failed = result.metrics.filter((metric) => metric.error).length;
    status(failed ? `Analysis complete. ${failed} parameters unavailable; see results.` : "Analysis complete.");
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
  data.append("keys", keys.join(","));
  setBusy(true);
  status("Preparing CSV...");
  try {
    const response = await post("/api/export", data);
    const url = URL.createObjectURL(await response.blob());
    const link = document.createElement("a");
    link.href = url;
    const suffix = $("suffix").value.replace(/[<>:"/\\|?*\x00-\x1f]/g, "_");
    link.download = analysisFile.name.replace(/\.stl$/i, "") + suffix + ".csv";
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
  const projected = triangles.map((triangle) => triangle.map(project));
  projected.sort((a, b) => a.reduce((s, p) => s + p[2], 0) - b.reduce((s, p) => s + p[2], 0));
  ctx.lineWidth = 0.4 * ratio;
  ctx.strokeStyle = "#516b86";
  for (const points of projected) {
    const depth = points.reduce((sum, point) => sum + point[2], 0) / 3;
    const light = Math.max(40, Math.min(82, 63 + depth * 18));
    ctx.fillStyle = `hsl(209, 40%, ${light}%)`;
    ctx.beginPath();
    points.forEach((point, index) => index ? ctx.lineTo(point[0], point[1]) : ctx.moveTo(point[0], point[1]));
    ctx.closePath();
    ctx.fill();
    ctx.stroke();
  }
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

canvas.addEventListener("pointerdown", (event) => {
  pointer = { x: event.clientX, y: event.clientY };
  canvas.setPointerCapture(event.pointerId);
});
canvas.addEventListener("pointermove", (event) => {
  if (!pointer) return;
  yaw += (event.clientX - pointer.x) * 0.01;
  pitch = Math.max(-1.5, Math.min(1.5, pitch + (event.clientY - pointer.y) * 0.01));
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
