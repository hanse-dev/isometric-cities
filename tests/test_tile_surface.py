"""Tiled surface rendering joins pixel-exactly and keeps per-building colours whole."""
import json
from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'source'))
import tile_surface as ts


def city(size=240, res=1.):
    half = size * res / 2
    return {'id': 'fixture', 'bbox': [-.003, -.003, .003, .003], 'surface': {
        'crs': 'EPSG:3857', 'resolution_m': res, 'bounds_projected_m': [-half, -half, half, half],
        'description': 'Synthetic measured surface', 'colour_evidence': 'Fixture', 'attribution': ['Fixture'], 'rasters': [],
        'camera': {'centre_projected_m': [0., 0.], 'u_range_m': [-110., 110.], 'y_range_screen_m': [-90., 70.],
                   'z0_m': 0., 'bearing_deg': 313.6, 'elevation_deg': 35.264389682754654, 'pixels_per_metre': 2.}}}


def square(cx, cy, r):
    m = 111320.0  # EPSG:3857 metres per degree near the equator
    return [{'lon': (cx + dx) / m, 'lat': (cy + dy) / m} for dx, dy in ((-r, -r), (r, -r), (r, r), (-r, r), (-r, -r))]


def scene(folder, size=240, res=1.):
    rng = np.random.default_rng(7)
    dtm = np.add.outer(np.linspace(0, 3, size), np.linspace(0, 2, size)).astype(np.float32)
    dsm = dtm.copy()
    yy, xx = np.mgrid[:size, :size]
    elements, k = [], 1
    for cx in range(-90 * size // 240, 91 * size // 240, 45):
        for cy in range(-90 * size // 240, 91 * size // 240, 45):
            r = 8 + (k % 3) * 4
            j, i = int((cx + size * res / 2) / res), int((size * res / 2 - cy) / res)
            inside = (abs(xx - j) <= r / res) & (abs(yy - i) <= r / res)
            dsm[inside] = dtm[inside] + 6 + (k % 4) * 5 + (np.abs(xx - j)[inside] < 2) * 3   # ridge
            tags = {'building': 'yes'}
            if k % 5 == 0: tags['roof:colour'] = '#335577'
            elements.append({'type': 'way', 'id': k, 'tags': tags, 'geometry': square(cx, cy, r)})
            k += 1
    trees = (rng.random((size, size)) < .02) & (dsm == dtm)
    dsm += ndimage_tree(trees, rng)
    tower = (np.abs(xx - size // 2 - 5) < 2) & (np.abs(yy - size // 2 - 20) < 2)
    dsm[tower] += 60                                                                 # tall spire crosses tile rows
    rgb = np.zeros((size, size, 3), np.uint8); rgb[..., 0] = 170; rgb[..., 1] = 80 + (xx % 50); rgb[..., 2] = 60
    np.savez_compressed(folder / 'surface.npz', dsm=dsm.astype(np.float32), dtm=dtm, bounds=np.array(city(size, res)['surface']['bounds_projected_m']), rgb=rgb)
    elements.append({'type': 'way', 'id': 999, 'tags': {'highway': 'residential'}, 'geometry': [{'lon': -100 / 111320, 'lat': 3 / 111320}, {'lon': 100 / 111320, 'lat': 3 / 111320}]})
    (folder / 'osm.json').write_text(json.dumps({'elements': elements}))


def ndimage_tree(trees, rng):
    from scipy import ndimage as ndi
    crowns = ndi.binary_dilation(trees, iterations=3)
    return (crowns * (4 + 6 * rng.random(trees.shape))).astype(np.float32)


class TileSurface(unittest.TestCase):
    def test_tiles_join_bit_exactly_with_one_large_render(self):
        with tempfile.TemporaryDirectory() as t:
            folder = Path(t); scene(folder, 640)
            surf = ts.Surface(city(640), folder / 'surface.npz', folder / 'osm.json')
            table = surf.building_table(chunk=64)
            crops = []
            whole = surf.render_tile(table, 0, 0, surf.width, surf.height)
            self.assertEqual(whole.shape, (surf.height, surf.width, 3))
            self.assertGreater(len(np.unique(whole.reshape(-1, 3), axis=0)), 50)
            for size in (97, 160):
                tiled = np.zeros_like(whole)
                for y0 in range(0, surf.height, size):
                    for x0 in range(0, surf.width, size):
                        w, h = min(size, surf.width - x0), min(size, surf.height - y0)
                        tiled[y0:y0 + h, x0:x0 + w] = surf.render_tile(table, x0, y0, w, h)
                        r0, r1, c0, c1 = surf.cells_for(x0, y0, w, h)
                        crops.append((min(r1 + ts.MARGIN, 640) - max(r0 - ts.MARGIN, 0)) * (min(c1 + ts.MARGIN, 640) - max(c0 - ts.MARGIN, 0)))
                np.testing.assert_array_equal(tiled, whole)
            self.assertLess(max(crops), .7 * 640 * 640)   # every tile really worked on a partial grid

    def test_building_table_does_not_depend_on_chunking(self):
        with tempfile.TemporaryDirectory() as t:
            folder = Path(t); scene(folder)
            surf = ts.Surface(city(), folder / 'surface.npz', folder / 'osm.json')
            a, b = surf.building_table(chunk=40), surf.building_table(chunk=1000)
            for key in a:
                np.testing.assert_array_equal(a[key], b[key])
            self.assertTrue(a['has_roof'][1:].all())                       # photo colour for untagged roofs
            np.testing.assert_array_equal(a['roofc'][5], [0x33, 0x55, 0x77])  # tag wins over photo

    def test_small_holes_fill_but_open_ground_stays(self):
        m = np.ones((20, 60), bool)
        m[5:8, 5:8] = False          # small enclosed gap
        m[2:18, 20:58] = False       # large opening
        out = ts.fill_small_holes(m, 10)
        self.assertTrue(out[5:8, 5:8].all())
        self.assertFalse(out[2:18, 20:58].any())

    def test_near_ids_prefers_closest_footprint(self):
        bid = np.zeros((7, 9), np.int32); bid[3, 1] = 4; bid[3, 7] = 9
        out = ts.near_ids(bid)
        self.assertEqual(out[3, 3], 4); self.assertEqual(out[3, 5], 9); self.assertEqual(out[3, 4], 0)


if __name__ == '__main__':
    unittest.main()


class TileBuildings(unittest.TestCase):
    def test_building_ids_join_exactly_and_leave_the_image_unchanged(self):
        with tempfile.TemporaryDirectory() as t:
            folder = Path(t); scene(folder, 640)
            surf = ts.Surface(city(640), folder / 'surface.npz', folder / 'osm.json')
            table = surf.building_table()
            img, ids, kind = surf.render_tile(table, 0, 0, surf.width, surf.height, buildings=True)
            np.testing.assert_array_equal(img, surf.render_tile(table, 0, 0, surf.width, surf.height))
            self.assertTrue((kind == 1).any() and (kind == 2).any())
            self.assertTrue(((ids > 0) == (kind > 0)).all())
            tiled_ids, tiled_kind = np.zeros_like(ids), np.zeros_like(kind)
            for y0 in range(0, surf.height, 150):
                for x0 in range(0, surf.width, 150):
                    w, h = min(150, surf.width - x0), min(150, surf.height - y0)
                    _, a, b = surf.render_tile(table, x0, y0, w, h, buildings=True)
                    tiled_ids[y0:y0 + h, x0:x0 + w], tiled_kind[y0:y0 + h, x0:x0 + w] = a, b
            np.testing.assert_array_equal(tiled_ids, ids); np.testing.assert_array_equal(tiled_kind, kind)
            # tagged roofs (#335577) are visible as roof pixels of their own building ids
            tagged = [k for k, a in enumerate(surf.attrs) if a and a['roof'] == (0x33, 0x55, 0x77)]
            self.assertGreater(np.isin(ids, tagged)[kind == 1].sum(), 50)


class TileDepth(unittest.TestCase):
    def test_height_above_ground_joins_exactly_and_marks_buildings(self):
        with tempfile.TemporaryDirectory() as t:
            folder = Path(t); scene(folder, 640)
            surf = ts.Surface(city(640), folder / 'surface.npz', folder / 'osm.json')
            table = surf.building_table()
            img, ids, kind, hag = surf.render_tile(table, 0, 0, surf.width, surf.height, buildings='all')
            np.testing.assert_array_equal(img, surf.render_tile(table, 0, 0, surf.width, surf.height))
            tiled = np.zeros_like(hag)
            for y0 in range(0, surf.height, 150):
                for x0 in range(0, surf.width, 150):
                    w, h = min(150, surf.width - x0), min(150, surf.height - y0)
                    tiled[y0:y0 + h, x0:x0 + w] = surf.render_tile(table, x0, y0, w, h, buildings='all')[3]
            np.testing.assert_array_equal(tiled, hag)
            roofs, ground = hag[kind == 1], hag[(kind == 0)]
            self.assertGreater(np.median(roofs), 5)                  # fixture roofs are 6..21 m up
            self.assertLess(np.percentile(ground, 25), .5)         # open ground; the rest is tree canopy
            self.assertGreater(hag.max(), 55)                         # the 60 m spire
            meta = ts.render(city(640), folder / 'surface.npz', folder / 'osm.json', folder / 'out', depth=True)
            from PIL import Image
            with Image.open(folder / 'out/depth.png') as d:
                self.assertEqual((d.mode, d.size), ('L', (surf.width, surf.height)))
            self.assertEqual(meta['depth']['metres_per_level'], .5)


class MeasuredWater(unittest.TestCase):
    def test_nir_moves_shores_but_keeps_inland_shadows_land(self):
        with tempfile.TemporaryDirectory() as t:
            folder = Path(t); scene(folder, 240)
            npz = dict(np.load(folder / 'surface.npz'))
            osm = json.loads((folder / 'osm.json').read_text())
            m = 111320.0
            lake = [{'lon': x / m, 'lat': y / m} for x, y in ((-120, -120), (-60, -120), (-60, 120), (-120, 120), (-120, -120))]
            osm['elements'].append({'type': 'way', 'id': 5000, 'tags': {'natural': 'water'}, 'geometry': lake})
            (folder / 'osm.json').write_text(json.dumps(osm))
            nir = np.full((240, 240), 180, np.uint8)
            nir[:, :50] = 20                           # real water ends 10 m before the mapped shore (x=-60 m -> col 60)
            nir[100:110, 150:170] = 20                 # dark shadow far inland
            npz['nir'] = nir; np.savez_compressed(folder / 'surface.npz', **npz)
            surf = ts.Surface(city(240), folder / 'surface.npz', folder / 'osm.json')
            water = surf.ground == surf.G['water']
            self.assertTrue((water | (surf.bid > 0))[:, 5:45].all())  # fixture buildings stand in the lake and stay land
            self.assertFalse(water[:, 52:60].any())     # mapped water that is bright in NIR became land
            self.assertFalse(water[100:110, 150:170].any())
            self.assertGreater(surf.water_evidence['cells_water_to_land'], 0)
