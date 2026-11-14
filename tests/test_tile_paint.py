"""Tile store: best try wins, extension paints only new keys, water failures are rejected."""
import copy
import json
from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'source'))
sys.path.insert(0, str(ROOT / 'tests'))
import tile_paint as tp
from test_tile_surface import city as base_city, scene


def city(u=(-300., 300.)):
    c = base_city(640)
    c['surface']['camera'].update(u_range_m=list(u), y_range_screen_m=[-250., 250.])
    return c


class Painter:
    """Stand-in for the model: sharper for some seeds, and optionally paints a lake."""
    def __init__(self, lake_seeds=()):
        self.calls, self.lake_seeds = [], set(lake_seeds)

    def __call__(self, input_path, folder, seed):
        self.calls.append((Path(folder).parent.name, seed))
        folder.mkdir(parents=True, exist_ok=True)
        a = np.asarray(Image.open(input_path).convert('RGB')).astype(int)
        amp = seed % 7 * 4
        a = a + ((np.indices(a.shape[:2]).sum(0) % 2) * 2 - 1)[..., None] * amp
        if seed in self.lake_seeds:
            a[300:500, 300:500] = (60, 110, 150)
        out = folder / 'raw.png'
        Image.fromarray(np.clip(a, 0, 255).astype(np.uint8)).save(out)
        return out


class Yellow(Painter):
    """Paints every roof warm ochre, like the model sometimes does."""
    def __call__(self, input_path, folder, seed):
        out = super().__call__(input_path, folder, seed)
        a = np.asarray(Image.open(out)).astype(int); a[..., 0] += 40; a[..., 1] += 25; a[..., 2] -= 30
        Image.fromarray(np.clip(a, 0, 255).astype(np.uint8)).save(out)
        return out


class TilePaint(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(); self.folder = Path(self.tmp.name); scene(self.folder, 640)
        self.npz, self.osm = self.folder / 'surface.npz', self.folder / 'osm.json'

    def tearDown(self):
        self.tmp.cleanup()

    def run_paint(self, c, painter, store='store', **kw):
        return tp.paint(c, self.npz, self.osm, self.folder / store, 'Pixel Art, PixArFK. test', tries=3, painter=painter, log=lambda *_: None, **kw)

    def test_best_try_extension_and_export(self):
        p = Painter()
        plan = self.run_paint(city(), p, plan_only=True)
        self.assertEqual((plan['tiles'], plan['to_paint']), (2, 2))
        self.run_paint(city(), p)
        store = json.loads((self.folder / 'store/store.json').read_text())
        self.assertEqual([t['key'] for t in store['tiles']], ['0_0', '1_0'])
        for t in store['tiles']:
            best = max(t['tries'], key=lambda r: r['sharpness'])
            self.assertEqual(t['chosen'], best['attempt'])
        before = {t['key']: t['art_sha256'] for t in store['tiles']}
        # widen the frame by whole pixels: only the new tile is painted, stored tiles stay untouched
        p2 = Painter()
        self.run_paint(city((-300., 700.)), p2)
        self.assertEqual({k for k, _ in p2.calls}, {'2_0'})
        store = json.loads((self.folder / 'store/store.json').read_text())
        self.assertEqual({t['key']: t['art_sha256'] for t in store['tiles'] if t['key'] in before}, before)
        record = tp.export(city((-300., 700.)), self.npz, self.osm, self.folder / 'store', self.folder / 'out')
        with Image.open(self.folder / 'out/map.png') as im:
            self.assertEqual(im.size, (2000, 1000))
        again = tp.export(city((-300., 700.)), self.npz, self.osm, self.folder / 'store', self.folder / 'out2')
        self.assertEqual(record['map_sha256'], again['map_sha256'])
        with self.assertRaisesRegex(ValueError, 'not painted yet'):
            tp.export(city((-300., 1100.)), self.npz, self.osm, self.folder / 'store', self.folder / 'out3')

    def test_invented_water_try_is_rejected(self):
        c = city((-300., 200.))
        seeds = [tp.seed_for(4601, 0, 0, a) for a in range(3)]
        sharpest = max(seeds, key=lambda s: s % 7)
        self.run_paint(c, Painter(lake_seeds={sharpest}), store='a')
        tile = json.loads((self.folder / 'a/store.json').read_text())['tiles'][0]
        bad = next(r for r in tile['tries'] if r['seed'] == sharpest)
        self.assertFalse(bad['water_ok'])
        self.assertNotEqual(tile['tries'][tile['chosen']]['seed'], sharpest)
        with self.assertRaisesRegex(ValueError, 'failed the water check'):
            self.run_paint(c, Painter(lake_seeds=set(seeds)), store='b')

    def test_painter_receives_absolute_paths(self):
        import os
        seen = []
        def painter(input_path, folder, seed):
            seen.append((Path(input_path).is_absolute(), Path(folder).is_absolute()))
            return Painter()(input_path, folder, seed)
        cwd = os.getcwd(); os.chdir(self.folder)
        try:
            tp.paint(city((-300., 200.)), self.npz, self.osm, Path('relative-store'), 'x', tries=1, painter=painter, log=lambda *_: None)
        finally:
            os.chdir(cwd)
        self.assertTrue(seen and all(a and b for a, b in seen))

    def test_off_colour_tagged_roofs_are_recoloured_now_or_later(self):
        c = city((-300., 200.))
        self.run_paint(c, Yellow(), store='now')
        tile = json.loads((self.folder / 'now/store.json').read_text())['tiles'][0]
        fixed = [r for r in tile['recolour'] if r['evidence'] == 'osm-tag']
        self.assertTrue(fixed and all(r['target_rgb'] == [0x33, 0x55, 0x77] for r in fixed))
        raw = np.asarray(Image.open(self.folder / 'now/tiles/0_0/art_raw.png')).astype(int)
        art = np.asarray(Image.open(self.folder / 'now/tiles/0_0/art.png')).astype(int)
        changed = np.any(raw != art, axis=2)
        self.assertTrue(changed.any())
        target = np.array([0x33, 0x55, 0x77])
        self.assertLess(np.abs(np.median(art[changed], 0) - target).max(), 12)   # roofs now match their tag
        # the same correction applied after storing gives the same tile
        self.run_paint(c, Yellow(), store='later', recolour_roofs=False)
        self.assertEqual(json.loads((self.folder / 'later/store.json').read_text())['tiles'][0]['recolour'], [])
        later = json.loads((self.folder / 'later/store.json').read_text())
        for t in later['tiles']: t.pop('recolour')
        (self.folder / 'later/store.json').write_text(json.dumps(later))
        tp.recolour_stored(c, self.npz, self.osm, self.folder / 'later', log=lambda *_: None)
        self.assertEqual(json.loads((self.folder / 'later/store.json').read_text())['tiles'][0]['art_sha256'], tile['art_sha256'])

    def test_recolour_uses_one_gain_per_roof_and_spares_walls_and_water(self):
        class Fake: attrs = [None, {'osm': 'way/1', 'name': 'Kirche'}]
        rng = np.random.default_rng(1)
        art = np.clip(np.full((60, 60, 3), (200, 190, 170)) + rng.normal(0, 12, (60, 60, 1)), 0, 255).astype(np.uint8)
        ids = np.zeros((60, 60), np.int32); ids[10:50, 10:50] = 1
        kind = np.where(ids > 0, 1, 0).astype(np.uint8); kind[40:50, 10:50] = 2          # bottom rows are wall
        kept = np.zeros((60, 60), bool); kept[10:20, 10:20] = True                       # kept water inside the roof
        out, rec = tp.recolour(art, ids, kind, kept, Fake(), {1: (np.array([112., 164., 142.]), 'osm-tag')})
        self.assertEqual(len(rec), 1)
        changed = np.any(out != art, axis=2)
        self.assertFalse(changed[40:50].any() or changed[10:20, 10:20].any() or changed[:10].any())
        sel = changed
        ratio = out[sel].astype(float) / art[sel]
        self.assertLess(ratio.std(0).max(), .02)
        self.assertLess(np.abs(np.median(out[sel], 0) - [112, 164, 142]).max(), 3)

    def test_frame_must_stay_on_the_store_lattice(self):
        self.run_paint(city((-300., 200.)), Painter(), store='s')
        with self.assertRaisesRegex(ValueError, 'pixel lattice'):
            self.run_paint(city((-300.25, 200.)), Painter(), store='s')

    def test_store_refuses_a_changed_tile(self):
        self.run_paint(city((-300., 200.)), Painter(), store='s')
        art = self.folder / 's/tiles/0_0/art.png'
        a = np.asarray(Image.open(art)).copy(); a[0, 0] = 0; Image.fromarray(a).save(art)
        with self.assertRaisesRegex(ValueError, 'changed on disk'):
            self.run_paint(city((-300., 200.)), Painter(), store='s')


if __name__ == '__main__':
    unittest.main()
