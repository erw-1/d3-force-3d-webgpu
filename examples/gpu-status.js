// The GPU line shared by the demos: which adapter WebGPU handed us, and a "wrong GPU?" button
// that says how to switch when a laptop's integrated GPU was picked instead of the fast one.
//
//     var adapter = await mountGPUStatus(document.getElementById("gpu"));   // null: no WebGPU
//
// The colours follow the page's text colour, so it fits a dark or a light page.

var CSS = `
.gpu-status { --gpu-ok: #3ecf8e; --gpu-bad: #f0605d; --gpu-warn: #d9a441; }
.gpu-line { display: flex; align-items: flex-start; gap: 8px; }
.gpu-dot { flex: none; width: 8px; height: 8px; margin-top: .5em; border-radius: 50%; background: #6b7488; }
.gpu-ok .gpu-dot { background: var(--gpu-ok); box-shadow: 0 0 6px color-mix(in srgb, var(--gpu-ok) 55%, transparent); }
.gpu-bad .gpu-dot { background: var(--gpu-bad); }
.gpu-name { min-width: 0; opacity: .85; }
/* important: the host page has its own button rules, often with ids */
.gpu-status button { font: inherit !important; color: inherit !important; background: none !important; cursor: pointer;
  border: 1px solid color-mix(in srgb, currentColor 35%, transparent) !important; border-radius: 999px !important; padding: 0 .6em !important; }
.gpu-help { flex: none; font-size: .85em !important; opacity: .75; white-space: nowrap; }
.gpu-help:hover, .gpu-help[aria-expanded="true"] { opacity: 1; }
.gpu-suspect .gpu-help { color: var(--gpu-warn) !important; border-color: currentColor !important; opacity: 1; }
.gpu-tip { margin: 6px 0 0; padding: 8px 10px; font-size: .92em; line-height: 1.45; text-align: left;
  background: color-mix(in srgb, currentColor 7%, transparent);
  border: 1px solid color-mix(in srgb, currentColor 22%, transparent); border-radius: 6px; }
.gpu-tip[hidden] { display: none; }
.gpu-tip p { margin: 0 0 6px; }
.gpu-tip ol { margin: 0 0 6px; padding-left: 1.3em; }
.gpu-tip li { margin: 4px 0; }
.gpu-tip code { padding: 0 .3em; border-radius: 3px; background: color-mix(in srgb, currentColor 12%, transparent); overflow-wrap: anywhere; }
.gpu-tip small { display: block; opacity: .75; }
.gpu-tip button { font-size: .85em !important; padding: 0 .5em !important; margin-left: .5em; border-radius: 4px !important; }
`;

function element(tag, className, text) {
  var e = document.createElement(tag);
  if (className) e.className = className;
  if (text !== undefined) e.textContent = text;
  return e;
}

// chrome://, edge:// or brave://
function flagsScheme() {
  if (navigator.brave) return "brave";
  return /\bEdg\//.test(navigator.userAgent) ? "edge" : "chrome";
}

function tip() {
  var flag = flagsScheme() + "://flags/#force-high-performance-gpu",
      box = element("div", "gpu-tip"),
      copy = element("button", "", "copy");
  box.hidden = true;
  box.innerHTML = "<p><strong>Wrong GPU?</strong> On a laptop with two GPUs, this page asks the browser for the fast one, "
      + "but Chrome on Windows ignores that and uses the integrated one. To switch:</p>"
      + "<ol><li>Set <code></code> to Enabled, then restart the browser.<small>Paste the address into a new tab.</small></li>"
      + "<li>Or, in Windows Settings › System › Display › Graphics, add your browser, choose Options › High performance, "
      + "and restart the browser.</li></ol>"
      + "<p>Then reload: the name above should change.</p>";
  box.querySelector("code").textContent = flag;
  copy.type = "button";
  copy.addEventListener("click", function() {
    if (!navigator.clipboard) return;
    navigator.clipboard.writeText(flag).then(function() {
      copy.textContent = "copied";
      setTimeout(function() { copy.textContent = "copy"; }, 1500);
    }, function() {});
  });
  box.querySelector("li").insertBefore(copy, box.querySelector("small"));
  return box;
}

export async function mountGPUStatus(host) {
  if (!document.getElementById("gpu-status-style")) {
    var style = element("style", "", CSS);
    style.id = "gpu-status-style";
    document.head.appendChild(style);
  }

  var root = element("div", "gpu-status"), line = element("div", "gpu-line"), name = element("span", "gpu-name", "checking WebGPU…");
  line.append(element("span", "gpu-dot"), name);
  root.appendChild(line);
  host.textContent = "";
  host.appendChild(root);

  function fail(text) {
    root.classList.add("gpu-bad");
    name.textContent = text;
    return null;
  }

  if (!navigator.gpu) return fail(window.isSecureContext ? "WebGPU: not supported by this browser" : "WebGPU: needs https or localhost");
  try {
    var adapter = await navigator.gpu.requestAdapter({powerPreference: "high-performance"});
    if (!adapter) return fail("WebGPU: supported, but no GPU adapter found");
    var info = adapter.info || {};
    root.classList.add("gpu-ok");
    name.textContent = "WebGPU: " + ([info.vendor, info.architecture, info.description].filter(Boolean).join(" · ") || "GPU adapter");

    // Intel and AMD make integrated GPUs (and Chrome on Windows picks the integrated one on a
    // laptop): stress the button for them.
    if (/windows/i.test(navigator.userAgent) && /^(intel|amd)$/i.test(info.vendor || "")) root.classList.add("gpu-suspect");

    var help = element("button", "gpu-help", "wrong GPU?"), box = tip();
    help.type = "button";
    help.setAttribute("aria-expanded", "false");
    help.addEventListener("click", function() {
      box.hidden = !box.hidden;
      help.setAttribute("aria-expanded", String(!box.hidden));
    });
    line.appendChild(help);
    root.appendChild(box);
    return adapter;
  } catch (error) {
    return fail("WebGPU: " + (error && error.message || "could not start"));
  }
}
