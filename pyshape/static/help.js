"use strict";

// In-app help shared by the single-particle and batch pages. The texts come
// from GET /api/descriptors (pyshape.descriptors), keyed by metric key and
// control name. Markup:
//   data-help="name"      a control (label, button, heading): tooltip; labels
//                         and headings also get a "?" button that opens help.
//   data-metric="key"     a result name button: tooltip, click opens help.
//   data-group="name"     a result group heading: tooltip when it has help.
// Pages call pyshapeHelp.refresh() after adding such elements.
const pyshapeHelp = (() => {
  let help = null;
  let dialog = null;

  function shortText(kind, key) {
    return help?.[kind]?.[key]?.short || "";
  }

  function groupText(name) {
    if (!help) return "";
    return help.groups[name] || help.groups[name.split(" (")[0]] || "";
  }

  function paragraph(text, label) {
    const p = document.createElement("p");
    if (label) {
      const strong = document.createElement("strong");
      strong.textContent = `${label}: `;
      p.append(strong);
    }
    p.append(text);
    return p;
  }

  function ensureDialog() {
    if (dialog) return dialog;
    dialog = document.createElement("dialog");
    dialog.id = "help-dialog";
    dialog.setAttribute("aria-labelledby", "help-title");
    dialog.addEventListener("click", (event) => {
      if (event.target === dialog) dialog.close(); // click on the backdrop
    });
    document.body.append(dialog);
    return dialog;
  }

  function entryNodes(entry) {
    const nodes = [paragraph(entry.description)];
    if (entry.formula) {
      const code = document.createElement("code");
      code.textContent = entry.formula;
      const p = paragraph("", "Formula");
      p.append(code);
      nodes.push(p);
    }
    if (entry.depends_on) nodes.push(paragraph(entry.depends_on, "Depends on"));
    return nodes;
  }

  function referenceNodes(keys) {
    const refs = [...new Set(keys)].map((key) => help.references[key]).filter(Boolean);
    if (!refs.length) return [];
    const heading = document.createElement("h3");
    heading.textContent = "References";
    const list = document.createElement("ul");
    for (const ref of refs) {
      const item = document.createElement("li");
      item.append(ref.citation);
      if (ref.url) {
        const link = document.createElement("a");
        link.href = ref.url;
        link.target = "_blank";
        link.rel = "noopener noreferrer";
        link.textContent = ref.url;
        item.append(" ", link);
      }
      list.append(item);
    }
    return [heading, list];
  }

  function open(kind, key, title) {
    const entry = help?.[kind]?.[key];
    if (!entry) return;
    const box = ensureDialog();
    const heading = document.createElement("h2");
    heading.id = "help-title";
    heading.textContent = title;
    const nodes = [heading, ...entryNodes(entry)];
    const refs = [...entry.references];
    if (kind === "controls" && key === "axes") {
      // One section per method offered by the page's axes selector.
      const select = document.querySelector("select[name=axes]");
      for (const option of select ? select.options : []) {
        const method = help.axes_methods[option.value];
        if (!method) continue;
        const sub = document.createElement("h3");
        sub.textContent = option.textContent;
        nodes.push(sub, ...entryNodes(method));
        refs.push(...method.references);
      }
    }
    nodes.push(...referenceNodes(refs));
    const close = document.createElement("button");
    close.type = "button";
    close.textContent = "Close";
    close.addEventListener("click", () => box.close());
    nodes.push(close);
    box.replaceChildren(...nodes);
    if (!box.open) box.showModal();
    close.focus();
  }

  function refresh(root = document) {
    if (!help) return;
    for (const element of root.querySelectorAll("[data-help]")) {
      const name = element.dataset.help;
      const text = shortText("controls", name);
      if (!text) continue;
      element.title = text;
      const target = element.htmlFor && document.getElementById(element.htmlFor);
      if (target) target.title = text;
      if (/^(LABEL|H2)$/.test(element.tagName) && !element.querySelector(".help-button")) {
        const button = document.createElement("button");
        button.type = "button";
        button.className = "help-button";
        button.textContent = "?";
        button.dataset.helpOpen = name;
        button.setAttribute("aria-label", `Help: ${element.textContent.trim()}`);
        element.append(" ", button);
      }
    }
    for (const select of root.querySelectorAll("select[name=axes]")) {
      for (const option of select.options) {
        const text = shortText("axes_methods", option.value);
        if (text) option.title = text;
      }
    }
    for (const element of root.querySelectorAll("[data-metric]")) {
      const text = shortText("metrics", element.dataset.metric);
      if (text) element.title = `${text} Click for details.`;
    }
    for (const element of root.querySelectorAll("[data-group]")) {
      const text = groupText(element.dataset.group);
      if (text) element.title = text;
    }
  }

  document.addEventListener("click", (event) => {
    const control = event.target.closest("[data-help-open]");
    if (control) {
      event.preventDefault(); // do not activate the label's control
      const owner = control.closest("[data-help]");
      const title = owner ? owner.textContent.replace(/\s*\?$/, "").trim() : control.dataset.helpOpen;
      open("controls", control.dataset.helpOpen, title);
      return;
    }
    const metric = event.target.closest("[data-metric]");
    if (metric) open("metrics", metric.dataset.metric, metric.textContent);
  });

  fetch("/api/descriptors")
    .then((response) => (response.ok ? response.json() : null))
    .then((data) => {
      help = data;
      refresh();
    })
    .catch(() => {}); // help is optional: the page works without it

  return { refresh, open };
})();
