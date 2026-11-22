#!/usr/bin/env python3
"""Write a map bundle: the interface between map making and the apps that use a map.

A bundle holds the finished art, source and height tiles, a per-pixel building hit map, the
camera, buildings with their OSM/Wikidata identity, a walkable street graph and the credits. It
carries no app code and no Wikipedia text; apps fetch those themselves from the Wikidata ids.

  atlas.py bundle --city RECIPE --surface NPZ --osm OSM --map-dir EXPORT --out BUNDLE

EXPORT is a tile export (map.png, source.png, depth.png, source.json); map.png may carry reviewed
landmark repaints. The hit map is rendered fresh from the same surface and camera, so every art
pixel can be resolved to the building whose surface is visible there.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import shutil
import sys

import numpy as np
from PIL import Image
from shapely.geometry import Point, Polygon

from project import digest, read, recipe_path, write
import surface_patch as sp
from tiles import build as build_tiles
import tile_surface as ts

FORMAT = 'pixel-map-bundle/1'
WALK = {'pedestrian', 'footway', 'path', 'steps', 'living_street', 'residential', 'service', 'unclassified',
        'tertiary', 'secondary', 'primary', 'cycleway', 'track'}


def encode_ids(ids):
    """Building index as 24-bit RGB: r = low byte, g = middle, b = high; 0 means no building."""
    ids = ids.astype(np.uint32)
    return np.stack([ids & 255, (ids >> 8) & 255, (ids >> 16) & 255], -1).astype(np.uint8)


def decode_ids(rgb):
    rgb = rgb.astype(np.uint32)
    return rgb[..., 0] | (rgb[..., 1] << 8) | (rgb[..., 2] << 16)


def address(tags):
    """'Mengstraße 4, 23552 Lübeck' from OSM addr:* tags, or None."""
    street = tags.get('addr:street') or tags.get('addr:place')
    if not street:
        return None
    line = f"{street} {tags['addr:housenumber']}" if tags.get('addr:housenumber') else street
    town = ' '.join(v for v in (tags.get('addr:postcode'), tags.get('addr:city')) if v)
    return f'{line}, {town}' if town else line


def buildings_and_owner(surf, osm_path):
    """Group renderer building ids under their parent OSM building; parts join the footprint they sit in."""
    elements = {f"{e['type']}/{e['id']}": e for e in json.loads(Path(osm_path).read_text())['elements']}
    parents, polys = [], []
    for k, at in enumerate(surf.attrs):
        if at is None:
            continue
        e = elements.get(at['osm'])
        if e is None or at['part'] or 'geometry' not in e:
            continue
        try:
            poly = Polygon([(p['lon'], p['lat']) for p in e['geometry']]).buffer(0)
        except ValueError:
            continue
        parents.append(k); polys.append(poly)
    from shapely.strtree import STRtree
    tree = STRtree(polys)
    owner = np.zeros(len(surf.attrs), np.int64)          # renderer id -> bundle index (1-based), 0 none
    index = {}
    for k in parents:
        index[k] = len(index) + 1
        owner[k] = index[k]
    for k, at in enumerate(surf.attrs):
        if at is None or owner[k]:
            continue
        e = elements.get(at['osm'])
        if e is None or 'geometry' not in e:
            continue
        g = e['geometry']
        c = Point(sum(p['lon'] for p in g) / len(g), sum(p['lat'] for p in g) / len(g))
        hit = [parents[i] for i in tree.query(c) if polys[i].contains(c)]
        if hit:
            owner[k] = index[min(hit, key=lambda p: polys[parents.index(p)].area)]
        else:                                             # a free-standing part becomes its own entry
            index[k] = len(index) + 1
            owner[k] = index[k]
    records = []
    for k, i in sorted(index.items(), key=lambda kv: kv[1]):
        e = elements[surf.attrs[k]['osm']]
        t = e.get('tags', {})
        records.append({'id': i, 'osm': surf.attrs[k]['osm'], 'name': t.get('name'), 'address': address(t), 'wikidata': t.get('wikidata'),
                        'wikipedia': t.get('wikipedia'), 'kind': t.get('building') or t.get('building:part'),
                        'historic': t.get('historic'), 'start_date': t.get('start_date')})
    return owner, records


def close_gaps(hit, across=10, down=4):
    """Fill empty slits inside a building's pixels: walls rasterised in columns and ground seen between
    buttresses leave gaps up to ~10 px across that break outlines and highlights. Only empty pixels
    between two pixels of the same building are filled; neighbouring buildings stay apart."""
    hit = hit.copy()
    for axis, ws in ((1, range(1, across + 1)), (0, range(1, down + 1))):
        a = hit if axis == 1 else hit.T
        for w in ws:
            n = a.shape[1] - (w + 1)
            left, right = a[:, :n], a[:, w + 1:]
            ok = (left == right) & (left > 0)
            for k in range(1, w + 1):
                ok &= a[:, k:k + n] == 0
            for k in range(1, w + 1):
                a[:, k:k + n][ok] = left[ok]
    return hit


def project(surf, lon, lat):
    """Geographic points to native map pixels at the measured ground height."""
    tf = sp.transformer(surf.spec, None)
    e, n = tf.transform(np.asarray(lon), np.asarray(lat))
    r = np.clip(((surf.N - n) / surf.res).astype(int), 0, surf.dtm.shape[0] - 1)
    c = np.clip(((e - surf.W) / surf.res).astype(int), 0, surf.dtm.shape[1] - 1)
    M = np.array(sp.pixel_matrix(surf.spec['camera']))
    return (M @ np.stack([e, n, surf.dtm[r, c], np.ones_like(e)])).T


def ground_grid(surf, step=10.0):
    """Measured ground height on a coarse grid, so apps can place geographic points (e.g. memorial
    stones) on the map: pixel = camera.pixel_matrix @ [east, north, ground(east, north), 1]."""
    k = max(1, round(step / surf.res))
    h = surf.dtm[k // 2::k, k // 2::k]
    return {'crs': surf.spec['crs'], 'west': surf.W + (k // 2 + .5) * surf.res, 'north': surf.N - (k // 2 + .5) * surf.res,
            'step_m': k * surf.res, 'columns': int(h.shape[1]), 'rows': int(h.shape[0]),
            'heights_dm': [int(v) for v in (h.ravel() * 10).round()]}


def walkable(surf, osm_path):
    """Streets and paths as a node/edge graph in native map pixels, with OSM ids and names."""
    nodes, edges, seen = [], [], {}
    for e in json.loads(Path(osm_path).read_text())['elements']:
        t = e.get('tags', {})
        if e['type'] != 'way' or t.get('highway') not in WALK or 'geometry' not in e or t.get('tunnel') == 'yes' or t.get('access') == 'private':
            continue
        xy = project(surf, [p['lon'] for p in e['geometry']], [p['lat'] for p in e['geometry']])
        ids = []
        for node, (x, y) in zip(e['nodes'], xy):
            if node not in seen:
                seen[node] = len(nodes); nodes.append([round(float(x), 1), round(float(y), 1)])
            ids.append(seen[node])
        edges.append({'osm': f"way/{e['id']}", 'highway': t['highway'], 'name': t.get('name'), 'nodes': ids})
    return {'coordinates': 'native map pixels', 'nodes': nodes, 'ways': edges}


def bundle(city, surface, osm, map_dir, out, *, tile=1024, log=print):
    map_dir, out = Path(map_dir), Path(out)
    if out.exists() and any(out.iterdir()):
        raise ValueError('Bundle directory must be new or empty')
    meta = read(map_dir / 'source.json')
    surf = ts.Surface(city, surface, osm)
    if [surf.width, surf.height] != meta['size'] or sp.pixel_matrix(surf.spec['camera']) != meta['camera']['pixel_matrix']:
        raise ValueError('Surface camera differs from the exported map; rebuild the export first')
    for name in ('map.png', 'source.png', 'depth.png'):
        with Image.open(map_dir / name) as im:
            if im.size != (surf.width, surf.height):
                raise ValueError(f'{name} does not match the map size')
    table = surf.building_table()
    owner, records = buildings_and_owner(surf, osm)
    hit = np.zeros((surf.height, surf.width), np.uint32)
    for y0 in range(0, surf.height, tile):
        for x0 in range(0, surf.width, tile):
            w, h = min(tile, surf.width - x0), min(tile, surf.height - y0)
            _, ids, _ = surf.render_tile(table, x0, y0, w, h, buildings=True)
            hit[y0:y0 + h, x0:x0 + w] = owner[ids]
            log(json.dumps({'hit_tile': [x0, y0, w, h]}))
    hit = close_gaps(hit)
    out.mkdir(parents=True, exist_ok=True)
    for name in ('map.png', 'source.png', 'depth.png'):
        shutil.copyfile(map_dir / name, out / name)
    Image.fromarray(encode_ids(hit)).save(out / 'hitmap.png', optimize=False)
    # bounding boxes and centres from the hit map: what a click can actually reach
    present = np.bincount(hit.ravel(), minlength=len(records) + 1)
    ys, xs = np.nonzero(hit)
    order = np.argsort(hit[ys, xs], kind='stable'); ys, xs, vals = ys[order], xs[order], hit[ys, xs][order]
    starts = np.searchsorted(vals, np.arange(len(records) + 2))
    for r in records:
        a, b = starts[r['id']], starts[r['id'] + 1]
        r['visible_px'] = int(present[r['id']])
        if b > a:
            r['box'] = [int(xs[a:b].min()), int(ys[a:b].min()), int(xs[a:b].max()) + 1, int(ys[a:b].max()) + 1]
            r['centre'] = [int(np.median(xs[a:b])), int(np.median(ys[a:b]))]
    write(out / 'buildings.json', {'format': FORMAT, 'hitmap_encoding': 'id = r + 256*g + 65536*b; 0 = none',
                                   'buildings': [r for r in records if r['visible_px']]})
    write(out / 'walkable.json', walkable(surf, osm))
    write(out / 'ground.json', ground_grid(surf))
    camera = dict(meta['camera'], depth=meta.get('depth'))
    write(out / 'camera.json', camera)
    pyramids = {name: build_tiles(out / f'{name}.png', out / f'{name}.dzi', out / f'{name}_files',
                                  resample='nearest' if name == 'hitmap' else 'box')
                for name in ('map', 'source', 'depth', 'hitmap')}
    credits = {'attribution': surf.spec['attribution'], 'art': 'FLUX.2 Klein 9B (FLUX Non-Commercial) with Pixel Art Redmond (Apache-2.0); non-commercial use only',
               'osm': '© OpenStreetMap contributors, ODbL 1.0'}
    manifest = {'format': FORMAT, 'city': city['id'], 'title': city.get('title'), 'size': [surf.width, surf.height],
                'tiles': pyramids, 'credits': credits,
                'files': {str(p.relative_to(out)): digest(p) for p in sorted(out.glob('*')) if p.is_file()}}
    write(out / 'manifest.json', manifest)
    return {'bundle': str(out), 'buildings': sum(1 for r in records if r['visible_px']),
            'walkable_ways': len(read(out / 'walkable.json')['ways']), 'size': [surf.width, surf.height]}


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--city', required=True); ap.add_argument('--surface', type=Path, required=True)
    ap.add_argument('--osm', type=Path, required=True); ap.add_argument('--map-dir', type=Path, required=True)
    ap.add_argument('--out', type=Path, required=True)
    a = ap.parse_args(argv)
    try:
        print(json.dumps(bundle(read(recipe_path(a.city)), a.surface, a.osm, a.map_dir, a.out), indent=2))
        return 0
    except (ValueError, OSError) as error:
        print(json.dumps({'status': 'failed', 'reason': str(error)}), file=sys.stderr)
        return 1


if __name__ == '__main__':
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    raise SystemExit(main())
