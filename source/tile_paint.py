#!/usr/bin/env python3
"""Paint a map as a growing store of fixed 1024-pixel tiles on a global grid.

Tiles sit at multiples of an 800-pixel stride in the camera's pixel lattice, so a tile keeps its
key when the map grows. Each tile's source is rendered on demand by tile_surface; its input is
that source with every earlier tile composited in store order, so finished neighbours guide it.
Up to --tries seeds are painted per tile; the sharpest try on built-up pixels that invents no
water and loses none is kept. A stored tile is never repainted: extending the frame paints only
the missing keys. This is sequential context infill, not shared-latent denoising.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
import json
import math
from pathlib import Path
import sys

import numpy as np
from PIL import Image
from scipy import ndimage as ndi

import local_style as style
from project import digest, ownership, read, recipe_path, write
import tile_surface as ts

WINDOW, STRIDE, FEATHER = 1024, 800, 128
WATER_MARGIN = 256          # source pixels around a tile used to find whole water bodies
WATER_MIN, FLAG_PX = 800, 3000


# ---------- checks ----------
def watery(a):
    r, g, b = (a[..., i].astype(int) for i in range(3))
    return (b - r >= 28) & (g - r >= 15) & (b >= g) & (b < 230)


def keep_water(source):
    """Open water to keep from the source: big water-coloured bodies, closed, filled, eroded 8 px."""
    w = (lambda r, g, b: (b - r >= 40) & (g - r >= 25) & (b >= g) & (b < 200))(*(source[..., i].astype(int) for i in range(3)))
    lab, n = ndi.label(w)
    if not n:
        return np.zeros(w.shape, bool)
    sizes = ndi.sum(w, lab, range(1, n + 1))
    big = np.isin(lab, 1 + np.nonzero(sizes > 15000)[0])
    big = ndi.binary_fill_holes(ndi.binary_closing(big, iterations=4))
    return ndi.binary_erosion(big, iterations=8)


def blobs(mask):
    lab, _ = ndi.label(ndi.binary_opening(mask, iterations=2))
    return sorted((int((lab[s] == k).sum()) for k, s in enumerate(ndi.find_objects(lab), 1) if (lab[s] == k).sum() >= WATER_MIN), reverse=True)


def check(source, art, kept):
    """Invented water (art water where the source has none) and lost water (the reverse)."""
    src_w = ndi.binary_dilation(watery(source), iterations=12) | kept
    invented = blobs(watery(art) & ~src_w)
    lost = blobs(ndi.binary_erosion(watery(source), iterations=6) & ~watery(art) & ~kept)
    return {'invented_water_px': invented[:5], 'lost_water_px': lost[:5],
            'water_ok': sum(invented) < FLAG_PX and sum(lost) < FLAG_PX}


def built_up(source, kept):
    r, g, b = (source[..., i].astype(int) for i in range(3))
    return ~kept & ~ndi.binary_dilation((g > r + 8) & (g > b), iterations=3)


def sharpness(image, mask):
    if not mask.any():
        return 0.0
    return float(np.abs(ndi.laplace(image.astype(np.float32).mean(-1)))[mask].mean())


# ---------- roof colour ----------
RECOLOUR_DELTA = 30       # minimum chroma distance (YCbCr, 0..255 scale) before a roof is corrected
RECOLOUR_MIN_PX = 150     # visible roof pixels needed, after a 1-pixel inset from the roof edge
PHOTO_MIN_CHROMA = 30     # an orthophoto colour must be this clearly coloured to count as evidence


def chroma(rgb):
    rgb = np.asarray(rgb, np.float32)
    y = rgb[..., 0] * .299 + rgb[..., 1] * .587 + rgb[..., 2] * .114
    return y, np.stack([rgb[..., 2] - y, rgb[..., 0] - y], -1)


def from_chroma(y, cbcr):
    b, r = cbcr[..., 0] + y, cbcr[..., 1] + y
    g = (y - .299 * r - .114 * b) / .587
    return np.clip(np.stack([r, g, b], -1) + .5, 0, 255).astype(np.uint8)


def roof_targets(surface, table, use_photo=False):
    """Building id -> (target colour, evidence) where the roof colour is tagged in OSM.

    Orthophoto medians mix roof, shadow, dormers and glare; in Lübeck they recoloured ordinary roofs
    lavender and teal, so they only count as evidence when use_photo is set."""
    targets = {}
    for k, at in enumerate(surface.attrs):
        if at is None:
            continue
        if at['roof'] is not None:
            targets[k] = (np.asarray(at['roof'], np.float32), 'osm-tag')
        elif use_photo and table['has_roof'][k]:
            colour = table['roofc'][k]
            if np.linalg.norm(chroma(colour)[1]) >= PHOTO_MIN_CHROMA:
                targets[k] = (colour, 'orthophoto-median')
    return targets


def recolour(art, ids, kind, kept, surface, targets):
    """Scale each strongly off-colour roof, channel by channel, so its median becomes the tagged
    colour. Relative shading and texture stay the model's; only the roof's own pixels change."""
    out = art.copy()
    _, c = chroma(art)
    records = []
    roofs = (kind == 1) & ~kept
    for b in np.unique(ids[roofs]):
        if b not in targets:
            continue
        mask = ndi.binary_erosion(roofs & (ids == b), iterations=1)
        if mask.sum() < RECOLOUR_MIN_PX:
            continue
        target, evidence = targets[b]
        before = np.median(art[mask].astype(np.float32), 0)
        delta = float(np.linalg.norm(np.median(c[mask], 0) - chroma(target)[1]))
        if delta < RECOLOUR_DELTA:
            continue
        gain = target / np.maximum(before, 1)
        out[mask] = np.clip(art[mask] * gain + .5, 0, 255).astype(np.uint8)
        at = surface.attrs[b]
        records.append({'building': int(b), 'osm': at['osm'], 'name': at['name'], 'evidence': evidence,
                        'target_rgb': [round(float(v)) for v in target], 'painted_median_rgb': [round(float(v)) for v in before],
                        'chroma_delta': round(delta, 1), 'pixels': int(mask.sum())})
    return out, records


# ---------- store ----------
class Store:
    def __init__(self, folder, identity):
        self.folder = Path(folder); self.path = self.folder / 'store.json'
        if self.path.exists():
            data = read(self.path)
            if data['identity'] != identity:
                raise ValueError('Store belongs to a different camera lattice, prompt or model; use a new --store')
            self.tiles = data['tiles']
        else:
            self.tiles = []
            self.folder.mkdir(parents=True, exist_ok=True)
        self.identity = identity
        for t in self.tiles:
            if digest(self.folder / t['art']) != t['art_sha256']:
                raise ValueError(f"Stored tile {t['key']} changed on disk; resume refused")

    def keys(self):
        return {t['key'] for t in self.tiles}

    def add(self, record):
        self.tiles.append(record)
        write(self.path, {'identity': self.identity, 'tiles': self.tiles})


def key_of(i, j):
    return f'{i}_{j}'


def alpha_for(tile, earlier):
    """context_infill's feather: a later tile fades in over pixels that earlier tiles already cover."""
    covered = np.zeros((WINDOW, WINDOW), bool)
    for t in earlier:
        dx, dy = t['x'] - tile['x'], t['y'] - tile['y']
        x0, y0, x1, y1 = max(dx, 0), max(dy, 0), min(dx + WINDOW, WINDOW), min(dy + WINDOW, WINDOW)
        if x0 < x1 and y0 < y1:
            covered[y0:y1, x0:x1] = True
    if not covered.any():
        return np.ones((WINDOW, WINDOW), np.float32)
    if covered.all():
        return np.zeros((WINDOW, WINDOW), np.float32)
    distance = ndi.distance_transform_edt(covered)
    return np.where(covered, np.clip(1 - distance / FEATHER, 0, 1), 1).astype(np.float32)


def compose(store, source, x0, y0, tiles=None):
    """Composite stored tiles, in order, over a source window whose top-left pixel is (x0, y0)."""
    h, w = source.shape[:2]
    canvas = source.astype(np.float32)
    tiles = store.tiles if tiles is None else tiles
    for n, t in enumerate(tiles):
        ox, oy = t['x'] - x0, t['y'] - y0
        a0, b0, a1, b1 = max(ox, 0), max(oy, 0), min(ox + WINDOW, w), min(oy + WINDOW, h)
        if a0 >= a1 or b0 >= b1:
            continue
        alpha = alpha_for(t, tiles[:n])[b0 - oy:b1 - oy, a0 - ox:a1 - ox, None]
        art = np.asarray(Image.open(store.folder / t['art']).convert('RGB'), np.float32)[b0 - oy:b1 - oy, a0 - ox:a1 - ox]
        canvas[b0:b1, a0:a1] = np.clip(canvas[b0:b1, a0:a1] * (1 - alpha) + art * alpha + .5, 0, 255).astype(np.uint8)
    return canvas.astype(np.uint8)


# ---------- painting ----------
def model_painter(caption, preset_path, backend='auto'):
    preset = json.loads(Path(preset_path).read_text())

    def paint(input_path, folder, seed):
        folder.mkdir(parents=True, exist_ok=True)
        window_preset = deepcopy(preset); window_preset['inference']['seed'] = seed
        preset_file = folder / 'preset.json'; write(preset_file, window_preset)
        prepared = {'image': str(input_path), 'sha256': digest(input_path), 'conditioning': style.native_dimensions(WINDOW, WINDOW)}
        candidate = style.style_plan(prepared, folder, caption, backend=backend, preset_path=preset_file)
        receipt_path = folder / 'receipt.json'
        if receipt_path.exists():
            receipt = read(receipt_path)
            if receipt['source']['sha256'] != prepared['sha256'] or not style.same_style_job(receipt['style_plan'], candidate):
                raise ValueError(f'Tile try {folder} changed identity; resume refused')
        else:
            receipt = {'source': prepared, 'style_plan': candidate, 'state': 'prepared'}
            write(receipt_path, receipt)
        while not style.completed_art(receipt):
            style.queue_style(receipt, receipt_path, wait=True)
            receipt = read(receipt_path)
        return Path(receipt['art'][0]['path'])
    return paint


def seed_for(base, i, j, attempt):
    return (base + 7919 * i + 104729 * j + 1_000_003 * attempt) % 2_147_483_647


def tile_range(x0, y0, w, h):
    i0, j0 = math.floor(x0 / STRIDE), math.floor(y0 / STRIDE)
    i1, j1 = math.floor((x0 + w - WINDOW) / STRIDE), math.floor((y0 + h - WINDOW) / STRIDE)
    i1 = max(i1, i0); j1 = max(j1, j0)
    # extend so the last tile reaches the frame edge
    while i1 * STRIDE + WINDOW < x0 + w: i1 += 1
    while j1 * STRIDE + WINDOW < y0 + h: j1 += 1
    return [(i, j) for j in range(j0, j1 + 1) for i in range(i0, i1 + 1)]


def frame_of(city, anchor):
    """The recipe frame in the store's anchored pixel lattice; it must sit on whole pixels."""
    cam = city['surface']['camera']; s = float(cam['pixels_per_metre'])
    (u0, u1), (y0, y1) = cam['u_range_m'], cam['y_range_screen_m']
    dx, dy = (u0 - anchor['u0']) * s, (y0 - anchor['y0']) * s
    if abs(dx - round(dx)) > 1e-6 or abs(dy - round(dy)) > 1e-6:
        raise ValueError('Frame is not on the store pixel lattice; widen u/y ranges by whole pixels')
    return round(dx), round(dy), round((u1 - u0) * s), round((y1 - y0) * s)


def anchored(city, anchor):
    """Copy of the recipe whose camera origin is the store anchor, so tile pixels use one lattice."""
    city = deepcopy(city); cam = city['surface']['camera']
    w, h = (cam['u_range_m'][1] - cam['u_range_m'][0]), (cam['y_range_screen_m'][1] - cam['y_range_screen_m'][0])
    cam['u_range_m'] = [anchor['u0'], anchor['u0'] + w]; cam['y_range_screen_m'] = [anchor['y0'], anchor['y0'] + h]
    return city


def paint(city, surface, osm, store_dir, caption, *, preset_path=None, tries=3, backend='auto', plan_only=False, painter=None,
          recolour_roofs=True, log=lambda line: print(line, flush=True)):
    preset_path = Path(preset_path or style.DEFAULT_PRESET).resolve()
    preset = json.loads(preset_path.read_text())
    cam = city['surface']['camera']
    prompt = caption if caption.startswith(preset['adapter']['trigger']) else preset['adapter']['trigger'] + ' ' + caption
    store_dir = Path(store_dir).resolve()       # the native worker re-derives its command from absolute paths
    anchor_path = store_dir / 'store.json'
    anchor = read(anchor_path)['identity']['anchor'] if anchor_path.exists() else {'u0': cam['u_range_m'][0], 'y0': cam['y_range_screen_m'][0]}
    identity = {'algorithm': 'tile-store-v1', 'window': WINDOW, 'stride': STRIDE, 'feather': FEATHER, 'anchor': anchor,
                'camera': {k: cam[k] for k in ('centre_projected_m', 'z0_m', 'bearing_deg', 'elevation_deg', 'pixels_per_metre')},
                'resolution_m': city['surface']['resolution_m'], 'crs': city['surface']['crs'], 'prompt': prompt,
                'preset_sha256': digest(preset_path)}
    frame = frame_of(city, anchor)
    keys = tile_range(*frame)
    with ownership(store_dir):
        store = Store(store_dir, identity)
        todo = [(i, j) for i, j in keys if key_of(i, j) not in store.keys()]
        plan = {'frame': frame, 'tiles': len(keys), 'stored': len(keys) - len(todo), 'to_paint': len(todo), 'tries': tries}
        if plan_only:
            return plan
        log(json.dumps(plan))
        surf = ts.Surface(anchored(city, anchor), surface, osm)
        table = surf.building_table()
        targets = roof_targets(surf, table) if recolour_roofs else {}
        painter = painter or model_painter(caption, preset_path, backend)
        for i, j in todo:
            key, x, y = key_of(i, j), i * STRIDE, j * STRIDE
            folder = store.folder / 'tiles' / key
            folder.mkdir(parents=True, exist_ok=True)
            m = WATER_MARGIN
            wide, wide_ids, wide_kind = surf.render_tile(table, x - m, y - m, WINDOW + 2 * m, WINDOW + 2 * m, buildings=True)
            source = np.ascontiguousarray(wide[m:m + WINDOW, m:m + WINDOW])
            ids, kind = wide_ids[m:m + WINDOW, m:m + WINDOW], wide_kind[m:m + WINDOW, m:m + WINDOW]
            kept = keep_water(wide)[m:m + WINDOW, m:m + WINDOW]
            Image.fromarray(source).save(folder / 'source.png')
            inp = compose(store, source, x, y)
            Image.fromarray(inp).save(folder / 'input.png')
            mask = built_up(source, kept)
            base = sharpness(inp, mask)
            results = []
            for attempt in range(tries):
                seed = seed_for(preset['inference']['seed'], i, j, attempt)
                raw = painter(folder / 'input.png', folder / f'try{attempt}', seed)
                art = np.asarray(Image.open(raw).convert('RGB'))
                if art.shape != source.shape:
                    raise ValueError(f'Tile {key} try {attempt} has wrong dimensions')
                art = np.where(kept[..., None], source, art).astype(np.uint8)
                result = {'attempt': attempt, 'seed': seed, 'raw': str(Path(raw).relative_to(store.folder)), 'raw_sha256': digest(raw),
                          'sharpness': round(sharpness(art, mask), 2), **check(source, art, kept)}
                results.append((result, art))
                log(json.dumps({'tile': key, **{k: result[k] for k in ('attempt', 'seed', 'sharpness', 'water_ok')}}))
            good = [r for r in results if r[0]['water_ok']]
            if not good:
                write(folder / 'rejected.json', {'tile': key, 'tries': [r for r, _ in results]})
                raise ValueError(f'Every try for tile {key} failed the water check; inspect {folder} before continuing')
            best, art = max(good, key=lambda r: r[0]['sharpness'])
            Image.fromarray(art).save(folder / 'art_raw.png')
            art, recoloured = recolour(art, ids, kind, kept, surf, targets)
            Image.fromarray(art).save(folder / 'art.png')
            store.add({'key': key, 'i': i, 'j': j, 'x': x, 'y': y, 'art': f'tiles/{key}/art.png', 'art_sha256': digest(folder / 'art.png'),
                       'source_sha256': digest(folder / 'source.png'), 'input_sha256': digest(folder / 'input.png'),
                       'kept_px': int(kept.sum()), 'input_sharpness': round(base, 2), 'chosen': best['attempt'],
                       'tries': [r for r, _ in results], 'art_raw_sha256': digest(folder / 'art_raw.png'),
                       'recolour': recoloured, 'visual_acceptance': 'unreviewed'})
            log(json.dumps({'tile': key, 'chosen': best['attempt'], 'sharpness': best['sharpness'], 'recoloured': len(recoloured), 'stored': len(store.tiles)}))
        return {**plan, 'stored': len(store.tiles), 'to_paint': 0}


def recolour_stored(city, surface, osm, store_dir, log=lambda line: print(line, flush=True)):
    """Apply the roof correction to stored tiles painted before it existed. Each tile's source is
    re-rendered and must match its stored hash, so the building masks are those of its own pixels."""
    store_dir = Path(store_dir).resolve()
    with ownership(store_dir):
        store = Store(store_dir, read(store_dir / 'store.json')['identity'])
        surf = ts.Surface(anchored(city, store.identity['anchor']), surface, osm)
        table = surf.building_table(); targets = roof_targets(surf, table)
        done = []
        for t in store.tiles:
            if 'recolour' in t:
                continue
            folder, m = store.folder / 'tiles' / t['key'], WATER_MARGIN
            wide, wide_ids, wide_kind = surf.render_tile(table, t['x'] - m, t['y'] - m, WINDOW + 2 * m, WINDOW + 2 * m, buildings=True)
            source = np.ascontiguousarray(wide[m:m + WINDOW, m:m + WINDOW])
            Image.fromarray(source).save(folder / 'source-check.png')
            if digest(folder / 'source-check.png') != t['source_sha256']:
                raise ValueError(f"Tile {t['key']} source no longer reproduces; refusing to recolour")
            (folder / 'source-check.png').unlink()
            kept = keep_water(wide)[m:m + WINDOW, m:m + WINDOW]
            art = np.asarray(Image.open(folder / 'art.png').convert('RGB'))
            Image.fromarray(art).save(folder / 'art_raw.png')
            fixed, records = recolour(art, wide_ids[m:m + WINDOW, m:m + WINDOW], wide_kind[m:m + WINDOW, m:m + WINDOW], kept, surf, targets)
            Image.fromarray(fixed).save(folder / 'art.png')
            t.update(art_raw_sha256=digest(folder / 'art_raw.png'), art_sha256=digest(folder / 'art.png'), recolour=records,
                     recolour_after_store=True)
            write(store.path, {'identity': store.identity, 'tiles': store.tiles})
            done.append(t['key']); log(json.dumps({'tile': t['key'], 'recoloured': len(records)}))
        return {'recoloured_tiles': done}


def export(city, surface, osm, store_dir, out):
    """Render the frame's source and composite every stored tile over it."""
    store = Store(store_dir, read(Path(store_dir) / 'store.json')['identity'])
    anchor = store.identity['anchor']
    x0, y0, w, h = frame_of(city, anchor)
    covered = {(t['x'], t['y']) for t in store.tiles}
    missing = [k for k in tile_range(x0, y0, w, h) if (k[0] * STRIDE, k[1] * STRIDE) not in covered]
    if missing:
        raise ValueError(f'{len(missing)} tiles of this frame are not painted yet; run paint first')
    out = Path(out); out.mkdir(parents=True, exist_ok=True)
    ts.render(city, surface, osm, out, depth=True)   # source.png, depth.png and source.json for this frame
    source = np.asarray(Image.open(out / 'source.png').convert('RGB'))
    art = compose(store, source, x0, y0)
    Image.fromarray(art).save(out / 'map.png')
    record = {'store': str(Path(store_dir).resolve()), 'frame': [x0, y0, w, h], 'tiles': [t['key'] for t in store.tiles],
              'depth_sha256': digest(out / 'depth.png'),
              'map_sha256': digest(out / 'map.png'), 'source_sha256': digest(out / 'source.png'), 'visual_acceptance': 'unreviewed'}
    write(out / 'export.json', record)
    return record


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest='action', required=True)
    for name in ('paint', 'export', 'recolour'):
        p = sub.add_parser(name)
        p.add_argument('--city', required=True); p.add_argument('--surface', type=Path, required=True)
        p.add_argument('--osm', type=Path, required=True); p.add_argument('--store', type=Path, required=True)
        if name == 'paint':
            p.add_argument('--caption-file', type=Path, required=True); p.add_argument('--preset', type=Path)
            p.add_argument('--tries', type=int, default=3); p.add_argument('--backend', choices=['auto', 'imagine', 'mflux'], default='auto')
            p.add_argument('--plan-only', action='store_true')
        elif name == 'export':
            p.add_argument('--out', type=Path, required=True)
    a = ap.parse_args(argv)
    try:
        city = read(recipe_path(a.city))
        if a.action == 'paint':
            if not 1 <= a.tries <= 5:
                raise ValueError('Use 1..5 tries per tile')
            result = paint(city, a.surface, a.osm, a.store, a.caption_file.read_text().strip(), preset_path=a.preset,
                           tries=a.tries, backend=a.backend, plan_only=a.plan_only)
        elif a.action == 'recolour':
            result = recolour_stored(city, a.surface, a.osm, a.store)
        else:
            result = export(city, a.surface, a.osm, a.store, a.out)
        print(json.dumps(result, indent=2))
        return 0
    except (ValueError, OSError) as error:
        print(json.dumps({'status': 'failed', 'reason': str(error)}), file=sys.stderr)
        return 1


if __name__ == '__main__':
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    raise SystemExit(main())
