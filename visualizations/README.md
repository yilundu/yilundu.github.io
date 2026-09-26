# Publication visualizations

Place this `visualizations` folder in your website root and use the accompanying `index.html` as the website's root page.

Media are physically grouped into venue-and-year folders such as `neurips2026/`, `corl2026/`, and `icml2025/`. All 166 original media files and the ten new NeurIPS figures are preserved without changing their bytes. The thirteen NeurIPS 2026 figures are together in `neurips2026/`.

Each folder contains its media, a gallery (`index.html`), and asset metadata (`assets.json`). Open `visualizations/index.html` to browse folders. `path-map.json` records every old and new media path.

Journal and thesis media use their listed venue and year. Entries without a listed venue use `other2022/`, `other2023/`, or `other2025/`; they are not assigned an unsupported conference.

The accompanying publication page changes only local media paths. Its 29 selected papers, 160 total publications, NeurIPS venue labels, and ordering are preserved.

## Web-optimized derivatives

The homepage displays these files at about 210×128 px, so it loads smaller derivatives that sit next to the originals. The originals are unchanged and are still what the galleries show.

- `*_thumb.jpg`: still images over 200 KB or wider than 1200 px, resized to at most 960 px wide. Transparent areas are flattened onto the thumbnail background (`#f1f4f3`).
- `*_poster.jpg`: one frame per preview video. The page script sets it as the video's poster just before the video scrolls into view, so previews are not blank when autoplay is off.
- `neurips2026/graph-energy-matching_preview.mp4`: H.264 version of the 14.8 MB `graph-energy-matching.gif`.
