
import glob
import pickle

paths = glob.glob('data_fps60/*/*/*.pkl')
print('found', len(paths), 'files')
if not paths:
    raise SystemExit('no .pkl files found')

p = paths[0]
with open(p, 'rb') as f:
    d = pickle.load(f)

print('path:', p)
print('keys:', list(d.keys()))
for k, v in d.items():
    t = type(v)
    shape = getattr(v, "shape", None)
    if shape is not None:
        print(k, t, shape)
    else:
        try:
            print(k, t, len(v))
        except TypeError:
            print(k, t, v)