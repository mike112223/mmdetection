import sys
import re
from collections import deque

log_fp = sys.argv[1]
csv_fp = sys.argv[2]
with open(log_fp) as f:
    lines = [x.strip() for x in f.readlines()]

ranges = deque((
    (0, 64), (64, 256), (256, 576), (576, 1024),
    (1024, 9216), (9216, 25000000), (0, 25000000),
    (64, 100), (100, 144), (144, 196), (196, 256)
))
pts = deque((
    r'Easy\s*Val AP: (\d\.\d*)',
    r'Medium\s*Val AP: (\d\.\d*)',
    r'Hard\s*Val AP: (\d\.\d*)',
))

table1 = \
    '{p(0, 64)}, {p(64, 256)}, {p(256, 576)}, {p(576, 1024)}, ' \
    '{p(1024, 9216)}, {p(9216, 25000000)}, {p(0, 25000000)}, ' \
    '{easy}, {medium}, {hard}, ' \
    '{r(0, 64)}, {r(64, 256)}, {r(256, 576)}, {r(576, 1024)}, ' \
    '{r(1024, 9216)}, {r(9216, 25000000)}, {r(0, 25000000)}'
table2 = \
    '{p(64, 100)}, {p(100, 144)}, {p(144, 196)}, {p(196, 256)}, ' \
    '{r(64, 100)}, {r(100, 144)}, {r(144, 196)}, {r(196, 256)}'
t = {}

i = 0
range_ = None
while True:
    if len(ranges) == 0 and 'r(196, 256)' in t.keys():
        break

    range_ = str(ranges.popleft()) if range_ is None else range_
    if range_ in lines[i]:
        # print(lines[i])
        i += 5
        assert 'face' in lines[i]
        line = [x.strip() for x in lines[i].split('|') if x]
        t[f'p{range_}'] = line[-1]
        t[f'r{range_}'] = line[-2]
        range_ = None
        # print(lines[i])
    i += 1

pt = None
while True:
    if i >= len(lines) or len(pts) == 0 and 'hard' in t.keys():
        break

    pt = pts.popleft() if pt is None else pt
    key = {'E': 'easy', 'M': 'medium', 'H': 'hard'}[pt[0]]
    ret = re.search(pt, lines[i])
    if ret:
        # print(lines[i])
        t[key] = ret.group(1)
        pt = None
    i += 1

with open(csv_fp, 'w') as wf:
    t1 = table1.format(**t)
    t2 = table2.format(**t)
    print(t1)
    print(t2)
    wf.write(f'{t1}\n{t2}\n')
