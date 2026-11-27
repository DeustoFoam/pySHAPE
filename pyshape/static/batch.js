"use strict";

const $ = (id) => document.getElementById(id);
const MAX_BYTES = 32 * 1024 * 1024;
// Same list as pyshape.web.WEB_EXTENSIONS (checked by the tests).
const EXTENSIONS = [".stl", ".obj", ".ply", ".off", ".glb", ".3mf", ".dae"];
const MESH_FILE = new RegExp(`(${EXTENSIONS.map((e) => e.replace(".", "\\.")).join("|")})$`, "i");
let chosen = [];      // { file, path } with path relative to the chosen folder
let outputs = [];     // { path, text } CSVs ready to download
let summaryRows = []; // analysis results, in table order, for the summary CSV
let busy = false;
let stopRequested = false;

function status(message, error = false) {
  $("status").textContent = message;
  $("status").classList.toggle("error", error);
}

function syncOutputUnits() {
  $("output-units").disabled = $("units").value === "as-is";
}

function setBusy(value) {
  busy = value;
  for (const id of ["run", "folder", "files", "recursive", "units", "output-units", "axes", "cavities", "suffix", "summary"]) {
    $(id).disabled = value;
  }
  if (!value) {
    syncOutputUnits();
    $("run").disabled = !meshFiles().length;
  }
  $("stop").hidden = !value;
  $("download-all").disabled = value || !outputs.length;
}

// Paths inside a picked folder start with the folder's own name; drop it.
function fromFolder(file) {
  return { file, path: file.webkitRelativePath.split("/").slice(1).join("/") || file.name };
}

function meshFiles() {
  const recursive = $("recursive").checked;
  return chosen
    .filter(({ path }) => MESH_FILE.test(path))
    .filter(({ path }) => recursive || !path.includes("/"))
    .sort((a, b) => a.path.localeCompare(b.path, undefined, { numeric: true }));
}

function csvPath(path, taken) {
  const suffix = $("suffix").value.replace(/[<>:"/\\|?*\x00-\x1f]/g, "_");
  const stem = path.replace(MESH_FILE, "") + suffix;
  let candidate = `${stem}.csv`;
  for (let n = 1; taken.has(candidate.toLowerCase()); n++) candidate = `${stem}_${n}.csv`;
  taken.add(candidate.toLowerCase());
  return candidate;
}

function cell(text, className) {
  const td = document.createElement("td");
  td.textContent = text;
  if (className) td.className = className;
  return td;
}

function showSelection() {
  const files = meshFiles();
  const skipped = chosen.length - files.length;
  $("files-table").replaceChildren(...files.map(({ path }, i) => {
    const row = document.createElement("tr");
    row.append(cell(String(i + 1)), cell(path), cell("waiting", "pending"), cell(""));
    return row;
  }));
  $("selection").textContent = chosen.length
    ? `${files.length} mesh file${files.length === 1 ? "" : "s"} to analyse`
      + (skipped ? `; ${skipped} other file${skipped === 1 ? "" : "s"} ignored`
        + ($("recursive").checked ? "" : " (subfolders are off)") : "") + "."
    : "No files chosen. Each file may be up to 32 MiB.";
  $("run").disabled = busy || !files.length;
  status(files.length ? "Ready. Press Analyse all." : "Choose a folder or files to begin.");
}

$("folder").addEventListener("change", () => {
  chosen = [...$("folder").files].map(fromFolder);
  $("files").value = "";
  showSelection();
});
$("files").addEventListener("change", () => {
  chosen = [...$("files").files].map((file) => ({ file, path: file.name }));
  $("folder").value = "";
  showSelection();
});
$("recursive").addEventListener("change", showSelection);
$("units").addEventListener("change", syncOutputUnits);
$("stop").addEventListener("click", () => {
  stopRequested = true;
  status("Stopping after the current file...");
});

async function analyse(file) {
  const data = new FormData();
  data.append("file", file);
  data.append("units", $("units").value);
  data.append("axes", $("axes").value);
  data.append("cavities", $("cavities").value);
  data.append("output_units", $("output-units").value);
  data.append("include_csv", "true");
  const response = await fetch("/api/analyze", { method: "POST", body: data });
  const body = await response.json();
  if (!response.ok) {
    throw new Error(typeof body.detail === "string" ? body.detail : JSON.stringify(body.detail));
  }
  return body;
}

function downloadLink(text, name) {
  const link = document.createElement("a");
  link.href = URL.createObjectURL(new Blob([text], { type: "text/csv" }));
  link.download = name.split("/").pop();
  link.textContent = name;
  return link;
}

$("batch-form").addEventListener("submit", async (event) => {
  event.preventDefault();
  if (busy) return;
  const files = meshFiles();
  if (!files.length) return;
  showSelection();
  for (const link of document.querySelectorAll("#files-table a")) URL.revokeObjectURL(link.href);
  outputs = [];
  summaryRows = [];
  stopRequested = false;
  setBusy(true);
  $("progress").hidden = false;
  $("progress").max = files.length;
  $("progress").value = 0;
  $("metadata").textContent = "";
  const taken = new Set();
  let failed = 0;
  const rows = $("files-table").rows;
  for (const [i, { file, path }] of files.entries()) {
    if (stopRequested) break;
    const [, , state, link] = rows[i].cells;
    state.textContent = "calculating...";
    rows[i].scrollIntoView({ block: "nearest" });
    status(`Analysing ${i + 1} of ${files.length}: ${path}`);
    try {
      if (file.size > MAX_BYTES) throw new Error("The file exceeds the 32 MiB limit.");
      const result = await analyse(file);
      const name = csvPath(path, taken);
      outputs.push({ path: name, text: result.csv });
      summaryRows.push({ path, result });
      const unavailable = result.metrics.filter((metric) => metric.error).length;
      state.textContent = (unavailable ? `done (${unavailable} parameters unavailable)` : "done")
        + result.warnings.map((text) => `. Warning: ${text}`).join("");
      state.className = result.warnings.length ? "warning" : "ok";
      link.replaceChildren(downloadLink(result.csv, name));
      if (!$("metadata").textContent) {
        $("metadata").textContent = `Input: ${result.input_units} | Output: ${result.output_units}`
          + ` | Axes: ${({pca: "PCA", min_volume: "minimum-volume box", inertia_ellipsoid: "inertia-equivalent ellipsoid", fitted_ellipsoid: "fitted ellipsoid"})[result.axes_method] || result.axes_method}.`;
      }
    } catch (error) {
      failed++;
      state.textContent = `failed: ${error.message}`;
      state.className = "failed";
    }
    $("progress").value = i + 1;
  }
  if ($("summary").checked && summaryRows.length) {
    outputs.push({ path: "pyshape_summary.csv", text: summaryCsv(summaryRows) });
  }
  const done = $("progress").value;
  const ok = done - failed;
  const stopped = done < files.length ? ` Stopped after ${done} of ${files.length} files.` : "";
  status(`Finished: ${ok} analysed, ${failed} failed.${stopped}`
    + (ok ? " Download the zip, or single CSVs from the table." : ""), failed > 0 && !ok);
  setBusy(false);
});

// Same layout as pyshape.batch.summary_csv_text: one row per particle.
function summaryCsv(rows) {
  const ascii = (unit) => unit.replace("µ", "u").replace("²", "^2").replace("³", "^3").replace("⁵", "^5");
  const quote = (value) => /[",\r\n]/.test(value) ? `"${value.replace(/"/g, '""')}"` : value;
  const columns = rows[0].result.metrics.filter((metric) => metric.default_selected);
  const lines = [["source_file", ...columns.map((m) => m.unit ? `${m.key} [${ascii(m.unit)}]` : m.key),
    "reader", "warnings"]];
  for (const { path, result } of rows) {
    const values = Object.fromEntries(result.metrics.map((m) => [m.key, m.value]));
    lines.push([path, ...columns.map((m) => values[m.key] === null || values[m.key] === undefined ? "" : String(values[m.key])),
      result.reader || "", result.warnings.join(" | ")]);
  }
  return lines.map((line) => line.map(quote).join(",")).join("\r\n") + "\r\n";
}

$("download-all").addEventListener("click", () => {
  if (!outputs.length) return;
  const link = document.createElement("a");
  link.href = URL.createObjectURL(zip(outputs));
  const folder = chosen[0]?.file.webkitRelativePath.split("/")[0];
  link.download = `${folder || "pyshape"}_results.zip`;
  document.body.append(link);
  link.click();
  link.remove();
  setTimeout(() => URL.revokeObjectURL(link.href), 1000);
});

// Minimal uncompressed (stored) zip writer; CSVs are small, so no compression.
const CRC_TABLE = Array.from({ length: 256 }, (_, n) => {
  let c = n;
  for (let k = 0; k < 8; k++) c = c & 1 ? 0xedb88320 ^ (c >>> 1) : c >>> 1;
  return c >>> 0;
});

function crc32(bytes) {
  let c = 0xffffffff;
  for (const byte of bytes) c = CRC_TABLE[(c ^ byte) & 0xff] ^ (c >>> 8);
  return (c ^ 0xffffffff) >>> 0;
}

function zip(entries) {
  const encoder = new TextEncoder();
  const now = new Date();
  const time = (now.getHours() << 11) | (now.getMinutes() << 5) | (now.getSeconds() >> 1);
  const date = ((now.getFullYear() - 1980) << 9) | ((now.getMonth() + 1) << 5) | now.getDate();
  const parts = [], central = [];
  let offset = 0;
  for (const { path, text } of entries) {
    const name = encoder.encode(path);
    const data = encoder.encode(text);
    const crc = crc32(data);
    const header = (signature, size) => {
      const view = new DataView(new ArrayBuffer(size));
      view.setUint32(0, signature, true);
      return view;
    };
    const local = header(0x04034b50, 30);
    // version, flags (bit 11: UTF-8 names), method 0 (stored), time, date, crc, sizes, name length
    [[4, 20], [6, 0x0800], [8, 0], [10, time], [12, date]].forEach(([at, v]) => local.setUint16(at, v, true));
    local.setUint32(14, crc, true);
    local.setUint32(18, data.length, true);
    local.setUint32(22, data.length, true);
    local.setUint16(26, name.length, true);
    const entry = header(0x02014b50, 46);
    [[4, 20], [6, 20], [8, 0x0800], [10, 0], [12, time], [14, date]].forEach(([at, v]) => entry.setUint16(at, v, true));
    entry.setUint32(16, crc, true);
    entry.setUint32(20, data.length, true);
    entry.setUint32(24, data.length, true);
    entry.setUint16(28, name.length, true);
    entry.setUint32(42, offset, true);
    parts.push(local, name, data);
    central.push(entry, name);
    offset += 30 + name.length + data.length;
  }
  const size = central.reduce((sum, part) => sum + part.byteLength, 0);
  const end = new DataView(new ArrayBuffer(22));
  end.setUint32(0, 0x06054b50, true);
  end.setUint16(8, entries.length, true);
  end.setUint16(10, entries.length, true);
  end.setUint32(12, size, true);
  end.setUint32(16, offset, true);
  return new Blob([...parts, ...central, end], { type: "application/zip" });
}

syncOutputUnits();
