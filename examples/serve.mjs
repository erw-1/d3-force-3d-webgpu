// Minimal static file server for the examples (no dependencies):
//
//     node examples/serve.mjs [port]
//
// then open http://localhost:8080/examples/benchmark.html. The examples import the
// library straight from src/ and its d3-* dependencies from jsDelivr (import map), so
// serve from the repository root and stay online. The same files run on GitHub Pages.

import {createServer} from "http";
import {readFile, stat} from "fs/promises";
import {extname, join, normalize, resolve, sep} from "path";
import {fileURLToPath} from "url";

var root = resolve(fileURLToPath(new URL("..", import.meta.url))),
    port = +process.argv[2] || 8080,
    types = {
      ".html": "text/html; charset=utf-8",
      ".js": "text/javascript; charset=utf-8",
      ".mjs": "text/javascript; charset=utf-8",
      ".json": "application/json",
      ".css": "text/css",
      ".png": "image/png",
      ".svg": "image/svg+xml",
      ".map": "application/json"
    };

createServer(async function(request, response) {
  try {
    var path = normalize(join(root, decodeURIComponent(new URL(request.url, "http://x").pathname)));
    if (path !== root && !path.startsWith(root + sep)) throw Object.assign(new Error("forbidden"), {code: 403});
    if ((await stat(path)).isDirectory()) path = join(path, "index.html");
    var body = await readFile(path); // read before replying, so a failure can still become a 404
    response.writeHead(200, {"content-type": types[extname(path)] || "application/octet-stream", "cache-control": "no-store"});
    response.end(body);
  } catch (error) {
    response.writeHead(error.code === 403 ? 403 : 404, {"content-type": "text/plain"});
    response.end(error.code === 403 ? "forbidden" : "not found");
  }
}).listen(port, function() {
  console.log("Serving " + root + " at http://localhost:" + port + "/");
});
