# water4 — Updates

> Terse log of every task: what was asked → what was done. Newest first.
> Older entries: `updates-archive.md`.

## 2026-09-02 — add Crowsnest analytics
One `<script defer src="https://cn.dested.com/cn.js">` tag in index.html head; deployed and verified (tag served, window.cn defined, pageview rows in the crowsnest DB).
Touched: index.html

## 2026-09-02 — create GitHub repo, commit, deploy via Drydock to water.dested.com
Repo dested/water4 (public). Drydock project `water`: static·bun·xs, dnsZone dested.com, A record → box, CI wired.
Touched: Dockerfile, drydock.yaml, .github/workflows/drydock.yml (Drydock-generated), cliffnotes.md, decisions.md

## 2026-09-02 — build a WebGPU fluid maze solver (millions of particles, pour from top)
Greenfield: Vite+React+TS+Tailwind+portless scaffold, maze gen + exact SDF, FLIP/PIC compute solver with atomic P2G, additive/bloom renderer, control panel, pointer pouring, solved detection, GPU timestamp stats. Cliffnotes kit created.
Touched: package.json, vite.config.ts, portless.json, index.html, src/**, cliffnotes.md, ui.md, decisions.md, verify.md
