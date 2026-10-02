// Offline cache for /games/. Bump VERSION whenever any file below changes.
const VERSION = 'games-v4';
const FILES = [
  '/games/',
  '/games/index.html',
  '/games/dino.html',
  '/games/tictactoe.html',
  '/games/2048.html',
  '/games/blocks.html',
  '/games/space.html',
  '/games/abc.html',
  '/games/numbers.html',
  '/games/games.css',
  '/games/register.js',
  '/games/manifest.webmanifest',
  '/games/icons/icon-192.png',
  '/games/icons/icon-512.png',
  '/games/icons/maskable-192.png',
  '/games/icons/maskable-512.png',
  '/games/icons/apple-touch-icon.png'
];

self.addEventListener('install', (e) => {
  e.waitUntil(
    caches.open(VERSION)
      .then((c) => c.addAll(FILES.map((u) => new Request(u, {cache: 'reload'}))))
      .then(() => self.skipWaiting())
  );
});

self.addEventListener('activate', (e) => {
  e.waitUntil(
    caches.keys()
      .then((keys) => Promise.all(keys.filter((k) => k.startsWith('games-') && k !== VERSION).map((k) => caches.delete(k))))
      .then(() => self.clients.claim())
  );
});

self.addEventListener('fetch', (e) => {
  const req = e.request;
  if (req.method !== 'GET') return;
  const url = new URL(req.url);
  if (url.origin !== location.origin || !url.pathname.startsWith('/games/')) return;
  e.respondWith(
    caches.open(VERSION).then((cache) =>
      cache.match(req, {ignoreSearch: true}).then((hit) => {
        if (hit) return hit;
        return fetch(req).then((res) => {
          if (res.ok && res.type === 'basic') cache.put(req, res.clone());
          return res;
        }).catch(() => req.mode === 'navigate'
          ? cache.match('/games/').then((r) => r || Response.error())
          : Response.error());
      })
    )
  );
});
