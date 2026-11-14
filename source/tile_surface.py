#!/usr/bin/env python3
"""Render a measured DSM/DTM surface in independent screen tiles that join pixel-exactly.

surface_patch.render draws a whole frame at once, so its memory grows with the map. Here the OSM
classes and footprints are rasterised once for the whole surface (a byte and an int per cell),
per-building colours are measured once, and every tile then processes only the cells that can
project into it, plus a fixed margin covering every neighbourhood filter. All filters are
fixed-radius and computed point by point, and the rasteriser works in global pixel coordinates,
so a tile is bit-identical to the same window of one larger render.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import sys

import numpy as np
from numba import njit
from PIL import Image
from scipy import ndimage as ndi

import surface_patch as sp

MARGIN = 96   # cells around a tile's core: hole fill 40 + gaussian(6) radius 24 + closing/box/gradient chains
HOLE = 40     # canopy holes up to this many cells across are filled; larger openings stay open
SCREEN = 2    # pixels rendered around each tile so depth-edge outlines see their neighbours
DISK2 = [(dy * dy + dx * dx, dy, dx) for dy in range(-2, 3) for dx in range(-2, 3) if 0 < dy * dy + dx * dx <= 4]
DISK2.sort()


def disk(radius):
    r = int(radius)
    y, x = np.mgrid[-r:r + 1, -r:r + 1]
    return y * y + x * x <= radius * radius


def box(a, size):
    """Mean over size x size cells, summed per point (scipy's uniform_filter keeps a running sum)."""
    w = np.full(size, 1 / size, a.dtype)
    return ndi.correlate1d(ndi.correlate1d(a, w, 0), w, 1)


def near_ids(bid):
    """Footprint id, else the id of the nearest footprint within 2 cells; ties go to the first offset."""
    out = bid.copy()
    pad = np.pad(bid, 2)
    ny, nx = bid.shape
    for _, dy, dx in DISK2:
        shifted = pad[2 + dy:2 + dy + ny, 2 + dx:2 + dx + nx]
        take = (out == 0) & (shifted > 0)
        out[take] = shifted[take]
    return out


def fill_small_holes(mask, size):
    lab, count = ndi.label(~mask)
    if not count:
        return mask
    small = np.zeros(count + 1, bool)
    for k, s in enumerate(ndi.find_objects(lab), 1):
        small[k] = s[0].stop - s[0].start <= size and s[1].stop - s[1].start <= size
    small[np.unique(np.r_[lab[0], lab[-1], lab[:, 0], lab[:, -1]])] = False   # open to the data edge
    small[0] = False
    return mask | small[lab]


@njit
def raster_window(X, Y, D, Hg, walla, wshade, opn, colr, nz, out, zb, ox, oy, lab, ids, kind):
    """surface_patch.raster in global pixel coordinates, writing only the window at (ox, oy).

    ids/kind record, per pixel, the building id of the visible surface and whether it is a
    roof (1) or a wall (2); 0 where no building is visible."""
    Hh, Wd = zb.shape
    ny, nx = X.shape
    for i in range(ny - 1):
        for j in range(nx - 1):
            for tri in range(2):
                if tri == 0:
                    a0, a1, b0, b1, c0, c1 = i, j, i + 1, j, i, j + 1
                else:
                    a0, a1, b0, b1, c0, c1 = i + 1, j, i + 1, j + 1, i, j + 1
                xa, ya, xb, yb, xc, yc = X[a0, a1], Y[a0, a1], X[b0, b1], Y[b0, b1], X[c0, c1], Y[c0, c1]
                minx = max(int(math.floor(min(xa, xb, xc))), ox); maxx = min(int(math.ceil(max(xa, xb, xc))), ox + Wd - 1)
                miny = max(int(math.floor(min(ya, yb, yc))), oy); maxy = min(int(math.ceil(max(ya, yb, yc))), oy + Hh - 1)
                if minx > maxx or miny > maxy: continue
                den = (yb - yc) * (xa - xc) + (xc - xb) * (ya - yc)
                if abs(den) < 1e-9: continue
                steep = nz[i, j, tri] < 0.55
                hi0, hi1 = a0, a1
                if Hg[b0, b1] > Hg[hi0, hi1]: hi0, hi1 = b0, b1
                if Hg[c0, c1] > Hg[hi0, hi1]: hi0, hi1 = c0, c1
                for py in range(miny, maxy + 1):
                    for px in range(minx, maxx + 1):
                        fx, fy = px + .5, py + .5
                        wa = ((yb - yc) * (fx - xc) + (xc - xb) * (fy - yc)) / den
                        wb = ((yc - ya) * (fx - xc) + (xa - xc) * (fy - yc)) / den
                        wc = 1 - wa - wb
                        if wa < -1e-4 or wb < -1e-4 or wc < -1e-4: continue
                        d = wa * D[a0, a1] + wb * D[b0, b1] + wc * D[c0, c1]
                        if d < zb[py - oy, px - ox]:
                            zb[py - oy, px - ox] = d
                            b = lab[hi0, hi1]
                            ids[py - oy, px - ox] = b
                            kind[py - oy, px - ox] = 0 if b == 0 else (2 if steep else 1)
                            if steep:
                                for k in range(3): out[py - oy, px - ox, k] = min(255.0, walla[hi0, hi1, k] * wshade[i, j] * opn[hi0, hi1])
                            else:
                                for k in range(3): out[py - oy, px - ox, k] = colr[i, j, tri, k]


class Surface:
    """Whole-surface inputs: heights, orthophoto, rasterised OSM classes and footprints, camera."""

    def __init__(self, city, surface, osm, grid=None):
        self.spec = spec = sp.validate(city['surface'])
        if spec.get('landmarks'):
            raise ValueError('Tiled rendering does not support landmark overlays')
        self.city, self.paths = city, {'surface': str(surface), 'osm': str(osm), 'grid': str(grid) if grid else None}
        L = np.load(surface)
        self.dsm, self.dtm = L['dsm'].astype(np.float32), L['dtm'].astype(np.float32)
        self.rgb = L['rgb'] if 'rgb' in L.files else None
        self.W, self.S, self.E, self.N = map(float, L['bounds'])
        self.res = float(spec['resolution_m'])
        if self.dsm.shape != self.dtm.shape or not np.isfinite(self.dsm).all() or not np.isfinite(self.dtm).all():
            raise ValueError('DSM/DTM must share a finite, complete grid')
        if not np.array_equal(L['bounds'], np.asarray(spec['bounds_projected_m'])):
            raise ValueError('Surface cache bounds do not match the city recipe')
        if self.dsm.shape != (round((self.N - self.S) / self.res), round((self.E - self.W) / self.res)):
            raise ValueError('Surface bounds, shape or resolution mismatch')
        if self.rgb is not None and (self.rgb.shape != (*self.dsm.shape, 3) or self.rgb.dtype != np.uint8):
            raise ValueError('Orthophoto rgb must be uint8 on the DSM/DTM grid')
        self.P = {**sp.DEFAULT_PALETTE, **spec.get('palette', {})}
        feats = sp.load_osm(osm, sp.transformer(spec, grid), self.W, self.N, self.res, self.P)
        self.ground, self.bid, self.attrs, self.G = sp.raster_layers(feats, self.dsm.shape, self.res, self.P)
        cam = spec['camera']
        self.centre, self.u0, self.y0 = cam['centre_projected_m'], cam['u_range_m'][0], cam['y_range_screen_m'][0]
        self.z0, self.bearing, self.elevation = cam['z0_m'], cam['bearing_deg'], cam['elevation_deg']
        self.s = float(cam['pixels_per_metre'])
        self.sin = .5 if self.elevation == 30 else math.sin(math.radians(self.elevation))
        self.cos = math.sqrt(3) / 2 if self.elevation == 30 else math.cos(math.radians(self.elevation))
        self.width = round((cam['u_range_m'][1] - cam['u_range_m'][0]) * self.s)
        self.height = round((cam['y_range_screen_m'][1] - cam['y_range_screen_m'][0]) * self.s)
        self.hmin, self.hmax = float(self.dtm.min()), float(self.dsm.max())

    # ---- per-cell layers on a crop [r0:r1, c0:c1] of the global grid ----
    def layers(self, r0, r1, c0, c1):
        res, P, G = self.res, self.P, self.G
        dsm, dtm = self.dsm[r0:r1, c0:c1], self.dtm[r0:r1, c0:c1]
        ground, bid = self.ground[r0:r1, c0:c1], self.bid[r0:r1, c0:c1]
        ndsm = dsm - dtm
        foot = bid > 0
        bidd = near_ids(bid)
        far3 = ~ndi.binary_dilation(foot, disk(3))            # nearest footprint more than 3 cells away
        bmask = (bidd > 0) & (ndsm > 1.5)
        hp = dsm - box(dsm, 5)
        rough = np.sqrt(box(hp * hp, 5))
        road = np.isin(ground, [G['asphalt'], G['sett'], G['parking'], G['ballast']])
        canopy = (~bmask) & far3 & (ndsm > 2.0) & (rough > 0.35)
        canopy = ndi.binary_opening(canopy, iterations=1)
        canopy = fill_small_holes(ndi.binary_closing(canopy, iterations=3), HOLE) & (~bmask) & far3 & (ndsm > 1.0)
        structure = (~bmask) & (~canopy) & (ndsm > 1.2) & ~(road & (ndsm < 4.5))
        H = np.where(bmask | canopy | structure, dsm, dtm).astype(np.float32)
        cd = ndi.distance_transform_edt(np.pad(canopy, 1))[1:-1, 1:-1] * res
        crown = dtm + np.clip(ndi.gaussian_filter(dsm, 1.2) - dtm, 0, None) * (.45 + .55 * np.sqrt(np.clip(cd / 3.0, 0, 1)))
        H = np.where(canopy, crown, H)
        H = np.where(bmask, ndi.median_filter(dsm, 3), H)
        gy, gx = np.gradient(ndi.gaussian_filter(H, .7), res)
        slope = np.degrees(np.arctan(np.hypot(gx, gy)))
        return dict(dsm=dsm, dtm=dtm, ground=ground, bidd=bidd, bmask=bmask, road=road, canopy=canopy,
                    structure=structure, H=H, slope=slope)

    # ---- per-building colours, measured once over each whole building ----
    def building_table(self, chunk=512):
        n = len(self.attrs)
        roofc = np.zeros((n, 3), np.float32); wallc = np.zeros((n, 3), np.float32); has_roof = np.zeros(n, bool)
        for k, at in enumerate(self.attrs):
            if at is None: continue
            if at['roof'] is not None: roofc[k] = at['roof']; has_roof[k] = True
            wallc[k] = at['wall'] if at['wall'] is not None else self.P['sandstone']
        tag_wall = np.array([a is not None and a['wall'] is not None for a in self.attrs])
        boxes = ndi.find_objects(self.bid)
        ny, nx = self.bid.shape
        groups = {}
        for k, b in enumerate(boxes, 1):
            if b is None: continue
            r0, r1, c0, c1 = max(b[0].start - 2, 0), min(b[0].stop + 2, ny), max(b[1].start - 2, 0), min(b[1].stop + 2, nx)
            key = ((r0 + r1) // 2 // chunk, (c0 + c1) // 2 // chunk)
            g = groups.setdefault(key, [ny, 0, nx, 0, []])
            g[0], g[1], g[2], g[3] = min(g[0], r0), max(g[1], r1), min(g[2], c0), max(g[3], c1)
            g[4].append(k)
        cnt = np.zeros(n); flat = np.zeros(n); photo = {}
        for key in sorted(groups):
            r0, r1, c0, c1, ids = groups[key]
            R0, R1, C0, C1 = max(r0 - MARGIN, 0), min(r1 + MARGIN, ny), max(c0 - MARGIN, 0), min(c1 + MARGIN, nx)
            lay = self.layers(R0, R1, C0, C1)
            lab = np.where(lay['bmask'], lay['bidd'], 0)
            wanted = np.zeros(n, bool); wanted[ids] = True
            sel = wanted[lab] & (lab > 0)
            cnt += np.bincount(lab[sel], minlength=n)
            flat += np.bincount(lab[sel], weights=(lay['slope'][sel] < 10), minlength=n)
            if self.rgb is not None:
                core = ndi.binary_erosion(lay['bmask'], iterations=2) & (lay['bidd'] > 0)
                core &= wanted[lay['bidd']]
                kk, cells = lay['bidd'][core], self.rgb[R0:R1, C0:C1][core]
                order = np.argsort(kk, kind='stable'); kk, cells = kk[order], cells[order]
                ids_, start = np.unique(kk, return_index=True)
                for b, part in zip(ids_, np.split(cells, start[1:])):
                    photo[int(b)] = np.median(part, 0)
        for b, colour in photo.items():
            if not has_roof[b]: roofc[b] = colour; has_roof[b] = True
        modern_wall = np.all(wallc == self.P['modern'], axis=1)
        pale = (flat / np.maximum(cnt, 1) > .8) & (cnt > 200) & ~tag_wall
        wallc[pale] = self.P['modern']
        return dict(roofc=roofc, has_roof=has_roof, modern_wall=modern_wall, wallc=wallc)

    # ---- one screen tile ----
    def cells_for(self, x0, y0, w, h):
        """Global cell range [r0, r1) x [c0, c1) whose cells can project into the pixel window."""
        s, b = self.s, math.radians(self.bearing)
        xa, xb = x0 - SCREEN - 2, x0 + w + SCREEN + 2
        ya, yb = y0 - SCREEN - 2, y0 + h + SCREEN + 2
        ua, ub = self.u0 + xa / s, self.u0 + xb / s
        A, B = ya / s + self.y0, yb / s + self.y0
        va = (-B - self.cos * (self.hmax - self.z0)) / self.sin
        vb = (-A - self.cos * (self.hmin - self.z0)) / self.sin
        es, ns = [], []
        for u in (ua, ub):
            for v in (va, vb):
                es.append(self.centre[0] + u * math.cos(b) + v * math.sin(b))
                ns.append(self.centre[1] - u * math.sin(b) + v * math.cos(b))
        res, ny, nx = self.res, *self.dsm.shape
        c0 = max(int(math.floor((min(es) - self.W) / res)) - 2, 0); c1 = min(int(math.ceil((max(es) - self.W) / res)) + 3, nx)
        r0 = max(int(math.floor((self.N - max(ns)) / res)) - 2, 0); r1 = min(int(math.ceil((self.N - min(ns)) / res)) + 3, ny)
        return r0, r1, c0, c1

    def render_tile(self, table, x0, y0, w, h, variant='edges', buildings=False):
        P, G, res, s = self.P, self.G, self.res, self.s
        r0, r1, c0, c1 = self.cells_for(x0, y0, w, h)
        ny, nx = self.dsm.shape
        R0, R1, C0, C1 = max(r0 - MARGIN, 0), min(r1 + MARGIN, ny), max(c0 - MARGIN, 0), min(c1 + MARGIN, nx)
        lay = self.layers(R0, R1, C0, C1)
        H, slope, bmask, bidd, canopy, structure = lay['H'], lay['slope'], lay['bmask'], lay['bidd'], lay['canopy'], lay['structure']
        alb = np.zeros((*H.shape, 3), np.float32)
        for name, g in G.items():
            alb[lay['ground'] == g] = P[{'wood': 'tree'}.get(name, name)]
        tslope = np.degrees(np.arctan(np.hypot(*np.gradient(ndi.gaussian_filter(lay['dtm'], 2), res))))
        rocky = (~bmask) & (tslope > 38) & ~lay['road']
        alb[rocky] = P['rock']
        alb[canopy] = P['tree']; alb[structure] = P['structure']
        lab = np.where(bmask, bidd, 0)
        bl = lab[bmask]
        alb[bmask] = np.where(table['has_roof'][bl, None], table['roofc'][bl],
                              np.where((slope[bmask] < 10)[:, None],
                                       np.where(table['modern_wall'][bl, None], P['modern'], P['flat_roof']), P['slate']))
        walla = np.where(bmask[..., None], table['wallc'][lab], alb)
        walla[structure] = P['structure']; walla[canopy] = np.array(P['tree']) * .7
        walla[rocky | ((~bmask) & (slope > 45) & ~canopy & ~structure)] = P['rock']
        # global cell coordinates keep every projected value identical to a larger render
        jj, ii = np.meshgrid(np.arange(C0, C1), np.arange(R0, R1))
        e = self.W + (jj + .5) * res - self.centre[0]; n = self.N - (ii + .5) * res - self.centre[1]
        u, v = sp.horizontal(e, n, self.bearing)
        X = ((u - self.u0) * s).astype(np.float64)
        Y = ((-(self.sin * v + self.cos * (H - self.z0)) - self.y0) * s).astype(np.float64)
        D = (self.cos * v - self.sin * H).astype(np.float64)
        Lv = np.array([-.62, -.30, .72]); Lv /= np.linalg.norm(Lv)
        z = H.astype(np.float64)
        nzs = np.zeros((*H.shape, 2)); shade = np.zeros((*H.shape, 2))
        for t, (A_, B_, C_) in enumerate((((0, 0), (1, 0), (0, 1)), ((1, 0), (1, 1), (0, 1)))):
            def P3(o):
                zz = np.pad(z, ((0, 1), (0, 1)), mode='edge')[o[0]:o[0] + H.shape[0], o[1]:o[1] + H.shape[1]]
                return np.stack([o[1] * res + 0 * zz, -o[0] * res + 0 * zz, zz], -1)
            pa, pb, pc = P3(A_), P3(B_), P3(C_)
            nn = np.cross(pb - pa, pc - pa); nn *= np.sign(nn[..., 2:3] + 1e-12)
            nn /= np.linalg.norm(nn, axis=-1, keepdims=True)
            nzs[..., t] = nn[..., 2]
            shade[..., t] = .36 + .78 * np.clip(nn @ Lv, 0, 1)
        open_ = np.clip(1 - np.clip(ndi.gaussian_filter(H, 6) - H, 0, 8) / 8 * .35, .65, 1)
        colr = np.clip(alb[:, :, None, :] * shade[..., None] * open_[:, :, None, None], 0, 255).astype(np.uint8)
        hs = ndi.gaussian_filter(H, 2.0); sgy, sgx = np.gradient(hs, res)
        hn = np.stack([-sgx, sgy, np.zeros_like(sgx)], -1)
        hn /= np.maximum(np.linalg.norm(hn, axis=-1, keepdims=True), 1e-6)
        wshade = (.40 + .80 * np.clip(hn @ Lv, 0, 1)).astype(np.float32)
        # rasterise only the core cells; the margin exists solely to make their filters exact
        k = (slice(r0 - R0, r1 - R0), slice(c0 - C0, c1 - C0))
        Wd, Hh = w + 2 * SCREEN, h + 2 * SCREEN
        img = np.zeros((Hh, Wd, 3), np.uint8); img[:] = (150, 150, 150); zb = np.full((Hh, Wd), 1e18)
        ids = np.zeros((Hh, Wd), np.int32); kind = np.zeros((Hh, Wd), np.uint8)
        raster_window(np.ascontiguousarray(X[k]), np.ascontiguousarray(Y[k]), np.ascontiguousarray(D[k]),
                      np.ascontiguousarray(H[k]), np.ascontiguousarray(walla[k].astype(np.float32)),
                      np.ascontiguousarray(wshade[k]), np.ascontiguousarray(open_[k].astype(np.float32)),
                      np.ascontiguousarray(colr[k]), np.ascontiguousarray(nzs[k]), img, zb, x0 - SCREEN, y0 - SCREEN,
                      np.ascontiguousarray(lab[k].astype(np.int32)), ids, kind)
        if variant == 'edges':
            dz = np.maximum(np.abs(np.diff(zb, axis=0, prepend=zb[:1])), np.abs(np.diff(zb, axis=1, prepend=zb[:, :1])))
            edge = dz > 2.5
            img[edge] = (img[edge] * .62).astype(np.uint8)
        crop = (slice(SCREEN, SCREEN + h), slice(SCREEN, SCREEN + w))
        if buildings:
            return img[crop], ids[crop], kind[crop]
        return img[crop]

    def metadata(self, image_path, tile, table_sha):
        cam = self.spec['camera']
        return dict(source=self.spec['description'], renderer='tile_surface', tile_size=tile, margin_cells=MARGIN,
                    camera=dict(type='orthographic surface', crs=self.spec['crs'], bearing_deg=self.bearing,
                                elevation_deg=self.elevation, px_per_m=self.s, width=self.width, height=self.height,
                                pixel_matrix=sp.pixel_matrix(cam),
                                pixel_matrix_coordinates=['projected_easting_m', 'projected_northing_m', 'elevation_m'],
                                centre_projected_m=self.centre, u_range_m=cam['u_range_m'], y_range_screen_m=cam['y_range_screen_m'],
                                z0_m=self.z0),
                    size=[self.width, self.height], palette=self.P, colour_evidence=self.spec['colour_evidence'],
                    roof_colour_from_orthophoto=self.rgb is not None, attribution=self.spec['attribution'],
                    input_sha256={k: sp.sha(v) if v else None for k, v in self.paths.items()},
                    building_table_sha256=table_sha, city_id=self.city['id'], surface_spec=self.spec,
                    renderer_sha256=sp.sha(__file__), source_sha256=sp.sha(image_path))


def table_digest(table):
    h = hashlib.sha256()
    for key in sorted(table):
        h.update(key.encode()); h.update(np.ascontiguousarray(table[key]).tobytes())
    return h.hexdigest()


def render(city, surface, osm, out, *, grid=None, tile=1024):
    surf = Surface(city, surface, osm, grid)
    table = surf.building_table()
    out = Path(out); out.mkdir(parents=True, exist_ok=True)
    image = np.zeros((surf.height, surf.width, 3), np.uint8)
    for y0 in range(0, surf.height, tile):
        for x0 in range(0, surf.width, tile):
            w, h = min(tile, surf.width - x0), min(tile, surf.height - y0)
            image[y0:y0 + h, x0:x0 + w] = surf.render_tile(table, x0, y0, w, h)
            print(json.dumps({'tile': [x0, y0, w, h]}), flush=True)
    Image.fromarray(image).save(out / 'source.png', optimize=False)
    meta = surf.metadata(out / 'source.png', tile, table_digest(table))
    (out / 'source.json').write_text(json.dumps(meta, indent=1) + '\n')
    print(json.dumps({'size': [surf.width, surf.height], 'source_sha256': meta['source_sha256'], 'out': str(out)}), flush=True)
    return meta


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--city', required=True, help='City recipe JSON with a surface section')
    ap.add_argument('--surface', type=Path, required=True, help='NPZ with dsm, dtm, bounds and optional rgb')
    ap.add_argument('--osm', type=Path, required=True, help='Overpass body-geom JSON')
    ap.add_argument('--grid', type=Path, help='Optional pinned horizontal transformation grid')
    ap.add_argument('--tile', type=int, default=1024)
    ap.add_argument('--out', type=Path, required=True)
    a = ap.parse_args(argv)
    try:
        from project import read, recipe_path
        render(read(recipe_path(a.city)), a.surface, a.osm, a.out, grid=a.grid, tile=a.tile)
        return 0
    except (ValueError, OSError) as error:
        print(json.dumps({'status': 'failed', 'reason': str(error)}), file=sys.stderr)
        return 1


if __name__ == '__main__':
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    raise SystemExit(main())
