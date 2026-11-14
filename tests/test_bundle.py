"""Map bundle: hit map resolves art pixels to buildings, parts join their parent, files are complete."""
import json
from pathlib import Path
import shutil
import sys
import tempfile
import unittest

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'source'))
sys.path.insert(0, str(ROOT / 'tests'))
import bundle as bd
import tile_surface as ts
from test_tile_surface import city, scene, square


class Bundle(unittest.TestCase):
    def test_bundle_hitmap_buildings_and_walkable(self):
        with tempfile.TemporaryDirectory() as t:
            folder = Path(t); scene(folder, 640)
            osm = json.loads((folder / 'osm.json').read_text())
            for e in osm['elements']:
                if 'highway' in e.get('tags', {}):
                    e['nodes'] = [9001, 9002]
            centre = lambda e: (sum(p['lon'] for p in e['geometry'][:4]) / 4, sum(p['lat'] for p in e['geometry'][:4]) / 4)
            main = min((e for e in osm['elements'] if e.get('tags', {}).get('building')), key=lambda e: sum(v * v for v in centre(e)))
            main['tags'].update(name='Holstentor', wikidata='Q321229')
            cx = sum(p['lon'] for p in main['geometry'][:4]) / 4 * 111320; cy = sum(p['lat'] for p in main['geometry'][:4]) / 4 * 111320
            osm['elements'].append({'type': 'way', 'id': 99977, 'tags': {'building:part': 'yes', 'height': '30'}, 'geometry': square(cx, cy, 3)})
            (folder / 'osm.json').write_text(json.dumps(osm))
            c = city(640)
            ts.render(c, folder / 'surface.npz', folder / 'osm.json', folder / 'map', depth=True)
            shutil.copyfile(folder / 'map/source.png', folder / 'map/map.png')
            result = bd.bundle(c, folder / 'surface.npz', folder / 'osm.json', folder / 'map', folder / 'b', log=lambda *_: None)
            b = folder / 'b'
            for name in ('manifest.json', 'camera.json', 'buildings.json', 'walkable.json', 'hitmap.png', 'map.dzi', 'hitmap.dzi', 'depth.dzi'):
                self.assertTrue((b / name).is_file(), name)
            buildings = json.loads((b / 'buildings.json').read_text())['buildings']
            self.assertEqual(result['buildings'], len(buildings))
            holsten = next(r for r in buildings if r['name'] == 'Holstentor')
            self.assertEqual(holsten['wikidata'], 'Q321229')
            self.assertFalse(any(r['osm'] == 'way/99977' for r in buildings))         # the part joined its parent
            hit = bd.decode_ids(np.asarray(Image.open(b / 'hitmap.png')))
            x, y = holsten['centre']
            self.assertEqual(int(hit[y, x]), holsten['id'])
            self.assertEqual(int((hit == holsten['id']).sum()), holsten['visible_px'])
            with Image.open(b / f"hitmap_files/{json.loads((b / 'manifest.json').read_text())['tiles']['hitmap']['max_level'] - 1}/0_0.png") as low:
                ids = set(bd.decode_ids(np.asarray(low)).ravel().tolist())
            self.assertTrue(ids <= {0} | {r['id'] for r in buildings})               # nearest: no blended ids
            walk = json.loads((b / 'walkable.json').read_text())
            self.assertEqual(len(walk['nodes']), 2); self.assertEqual(walk['ways'][0]['nodes'], [0, 1])
            manifest = json.loads((b / 'manifest.json').read_text())
            self.assertIn('non-commercial', manifest['credits']['art'])
            with self.assertRaisesRegex(ValueError, 'new or empty'):
                bd.bundle(c, folder / 'surface.npz', folder / 'osm.json', folder / 'map', folder / 'b', log=lambda *_: None)

    def test_id_encoding_round_trips(self):
        ids = np.array([[0, 1, 255, 256, 70000, 2**24 - 1]], np.uint32)
        np.testing.assert_array_equal(bd.decode_ids(bd.encode_ids(ids)), ids)


if __name__ == '__main__':
    unittest.main()
