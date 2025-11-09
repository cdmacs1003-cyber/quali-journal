/* QualiJournal Admin Service Worker
   Version is injected by GitHub Actions: __BUILD_ID__
*/
const BUILD = "__BUILD_ID__";
const PREFIX = "qj-admin-";
const CACHE = PREFIX + BUILD;

self.addEventListener("install", (event) => {
  // Take over immediately
  self.skipWaiting();
});

self.addEventListener("activate", (event) => {
  // Remove all old versioned caches and claim clients
  event.waitUntil((async () => {
    const keys = await caches.keys();
    await Promise.all(
      keys
        .filter((k) => k.startsWith(PREFIX) && k !== CACHE)
        .map((k) => caches.delete(k))
    );
    await self.clients.claim();
    try { console.log("[SW] activated", BUILD); } catch(_) {}
  })());
});

self.addEventListener("fetch", (event) => {
  const req = event.request;
  if (req.method !== "GET") return; // only cache GET

  const url = new URL(req.url);

  // Never cache HTML or the SW file itself
  if (
    url.pathname === "/" ||
    url.pathname.endsWith("/index.html") ||
    url.pathname === "/service-worker.js"
  ) {
    return;
  }

  // Cache hashed static assets under /assets/ with content hash in filename
  const hashedAsset = /^\/assets\/.+\.[a-f0-9]{8,}\.(js|css|png|jpg|svg|woff2?)$/i.test(
    url.pathname
  );

  if (hashedAsset) {
    event.respondWith(
      (async () => {
        const cache = await caches.open(CACHE);
        const hit = await cache.match(req);
        if (hit) return hit;
        const res = await fetch(req);
        // only cache successful (200) responses
        if (res && res.ok) { await cache.put(req, res.clone()); }
        return res;
      })()
    );
  }
});

// Optional: support a manual skip-waiting message
self.addEventListener("message", (event) => {
  if (!event.data) return;
  if (event.data === "SKIP_WAITING") {
    self.skipWaiting();
  }
});
