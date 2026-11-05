import sys
sys.path.insert(0, 'experiments')
from geom import *

call = sys.argv[1] if len(sys.argv) > 1 else '000001'
x0, x1, y0, y1 = [float(a) for a in sys.argv[2:6]] if len(sys.argv) > 5 else (-0.6, 0.1, -0.75, -0.2)
cd, resp = load_call(latest_run() / 'calls' / call)
meta = find_obs_meta(resp)
rgb = np.array(Image.open(cd / 'image_01.png'))
dep = np.array(Image.open(cd / 'image_03.png')).astype(np.float64) * 0.001
P = cam_points(dep, meta['camera_intrinsics']['front']['matrix'], meta['camera_poses']['front'])
res = 0.004
nx, ny = int((x1 - x0) / res), int((y1 - y0) / res)
hmap = np.full((nx, ny), np.nan)
cmap = np.zeros((nx, ny, 3), np.uint8)
pts = P.reshape(-1, 3)
cols = rgb.reshape(-1, 3)
ix = ((pts[:, 0] - x0) / res).astype(int)
iy = ((pts[:, 1] - y0) / res).astype(int)
ok = (ix >= 0) & (ix < nx) & (iy >= 0) & (iy < ny)
order = np.argsort(pts[ok, 2])  # higher points overwrite
for i, j, zz, c in zip(ix[ok][order], iy[ok][order], pts[ok, 2][order], cols[ok][order]):
    hmap[i, j] = zz
    cmap[i, j] = c
# image: rows = -x (far at top?) we draw x increasing downward (toward camera), y increasing to right
img = Image.fromarray(cmap).resize((ny * 4, nx * 4), Image.NEAREST)
img.save('output/views/topdown_rgb.png')
zt = np.nanpercentile(hmap, 50)
hz = np.nan_to_num(hmap, nan=0)
norm = np.clip((hz - 0.94) / 0.2, 0, 1)
Image.fromarray((norm * 255).astype(np.uint8)).resize((ny * 4, nx * 4), Image.NEAREST).save('output/views/topdown_h.png')
print('grid', nx, ny, 'median z', zt)
# print coarse height table every 2 cm
step = 5
for i in range(0, nx, step):
    row = []
    for j in range(0, ny, step):
        blk = hmap[i:i + step, j:j + step]
        v = np.nanmax(blk) if np.isfinite(blk).any() else np.nan
        row.append('  .  ' if np.isnan(v) else f'{(v - 0.954) * 100:5.1f}')
    print(f'x={x0 + i * res:+.2f} ' + ' '.join(row))
print('cols y from', y0, 'step', step * res)
