"""Read-only, terminal-size-aware Stage I dashboard with the entire loss history."""
import argparse
import json
import math
import os
from pathlib import Path
import shutil
import sys
import textwrap
import time


def read_json(path):
    try:
        return json.loads(path.read_text())
    except (OSError, ValueError):
        return {}


def history(run):
    groups = {'train': {}, 'validation': {}}
    try:
        lines = (run / 'metrics.jsonl').read_text().splitlines()
    except OSError:
        lines = []
    for line in lines:
        try:
            row = json.loads(line)
            value = float(row['flow'])
            if row.get('split') in groups and math.isfinite(value):
                groups[row['split']][int(row['step'])] = row
        except (ValueError, KeyError, TypeError):
            continue  # The writer may be partway through the last line.
    return tuple([rows[k] for k in sorted(rows)] for rows in groups.values())


def average(values, window=50):
    result, running = [], 0.0
    for i, value in enumerate(values):
        running += value
        if i >= window:
            running -= values[i-window]
        result.append(running / min(i+1, window))
    return result


def duration(seconds):
    seconds = max(0, int(seconds))
    days, rest = divmod(seconds, 86400)
    hours, rest = divmod(rest, 3600)
    minutes = rest // 60
    return (f'{days}d ' if days else '') + f'{hours:02d}h {minutes:02d}m'


def chart(train, validation, width, height, color=False):
    """Plot every recorded step; collisions share a cell, with mean/validation on top."""
    points = [(int(r['step']), float(r['flow'])) for r in train + validation]
    if not points:
        return ['Waiting for the first action-loss measurement.']
    width, height = max(12, width-12), max(4, height)
    positive = [y for _, y in points if y > 0]
    floor = min(positive)*0.5 if positive else 1e-12
    transformed = [math.log10(max(y, floor)) for _, y in points]
    low, high = min(transformed), max(transformed)
    margin = max(0.12, (high-low)*0.06)
    low, high = low-margin, high+margin
    xmax = max(1, max(x for x, _ in points))
    canvas = [[' ']*width for _ in range(height)]

    def location(x, y):
        col = round(x/xmax*(width-1))
        row = round((high-math.log10(max(y, floor)))/(high-low)*(height-1))
        return max(0, min(width-1, col)), max(0, min(height-1, row))

    def put(x, y, symbol):
        col, row = location(x, y)
        canvas[row][col] = symbol

    for row in train:
        put(row['step'], row['flow'], '.')
    smooth = average([r['flow'] for r in train])
    previous = None
    for record, value in zip(train, smooth):
        point = location(record['step'], value)
        if previous is not None:
            dx, dy = point[0]-previous[0], point[1]-previous[1]
            steps = max(abs(dx), abs(dy), 1)
            for i in range(steps+1):
                x, y = round(previous[0]+dx*i/steps), round(previous[1]+dy*i/steps)
                canvas[y][x] = '*'
        canvas[point[1]][point[0]] = '*'
        previous = point
    for row in validation:
        put(row['step'], row['flow'], 'O')
    colors = {'.': '\033[90m', '*': '\033[36m', 'O': '\033[33;1m'}
    result = []
    for i, row in enumerate(canvas):
        label = f'{10**(high-(high-low)*i/(height-1)):9.2e}' if i in (0, height//2, height-1) else ' '*9
        pixels = ''.join(colors.get(c, '')+c+'\033[0m' if color and c in colors else c for c in row)
        result.append(label+' |'+pixels)
    result.append(' '*10+'+'+'-'*width)
    axis = [' ']*width
    for value, position in [(0, 0), (xmax//2, width//2), (xmax, width-len(str(xmax)))]:
        for i, char in enumerate(str(value)):
            if position+i < width:
                axis[position+i] = char
    result.append(' '*11+''.join(axis))
    return result


def render(run, width=104, height=30, color=False):
    width = max(60, width-1)
    state = read_json(run/'status.json')
    args = read_json(run/'arguments.json')
    train, validation = history(run)
    step = int(state.get('step', train[-1]['step'] if train else 0))
    total = int(state.get('total_steps', args.get('steps', 4000)))
    accumulation = int(args.get('accumulation', 32))
    micro = int(state.get('micro', 0))
    fraction = min(1, step/max(1, total))
    size = max(10, width-35)
    filled = int(size*fraction)
    now = time.time()
    alive = (Path('/proc')/str(state.get('pid', 'missing'))).exists()
    phase = state.get('phase', 'starting')
    health = 'LIVE' if alive else 'DONE' if phase == 'completed' else 'PROCESS NOT RUNNING'
    age = max(0, int(now-state.get('updated_at', now)))
    gpu_label = ','.join(str(i) for i in state.get('gpu_ids', [5]))
    lines = [f'ROBOTWIN | STAGE I POSTERIOR TEACHER | GPU {gpu_label} | {time.strftime("%m-%d %H:%M:%S")}',
             f'[{"#"*filled}{"-"*(size-filled)}] {step:4d}/{total}  {fraction*100:5.1f}%',
             f'{health} | {phase} | micro {micro}/{accumulation} | heartbeat {age}s ago | PID {state.get("pid", "?")}']
    elapsed = duration(now-state.get('started_at', now))
    eta = duration(state['eta_seconds']) if 'eta_seconds' in state else 'pending'
    lines.append(f'Elapsed {elapsed} | ETA ~{eta} (recent updates; excludes future saving/validation)')
    if train:
        values = [r['flow'] for r in train]
        window = min(50, len(values))
        first, last = sum(values[:window])/window, sum(values[-window:])/window
        change = (last/first-1)*100 if first else 0
        lines.append(f'Action loss: latest {values[-1]:.6g} | mean{window} {last:.6g} | first mean{window} {first:.6g} ({change:+.2f}%)')
    else:
        lines.append('Action loss: awaiting first optimizer update')
    if validation:
        best = min(validation, key=lambda row: row['flow'])
        lines.append(f'Validation: latest {validation[-1]["flow"]:.6g} @ {validation[-1]["step"]} | best {best["flow"]:.6g} @ {best["step"]}')
    else:
        lines.append('Validation: pending')
    lines.append(f'FULL ACTION LOSS HISTORY | log10 y | . train ({len(train)} steps)  * mean50  O fixed validation')
    val_text = 'Validation history: '+('  '.join(f'{r["step"]}:{r["flow"]:.5g}' for r in validation) or 'pending')
    val_lines = textwrap.wrap(val_text, width=width)
    checkpoint = Path(state['checkpoint']).name if state.get('checkpoint') else 'none'
    footer = [f'Checkpoint: {checkpoint} | next regular save: {min(total, (step//500+1)*500)}',
              'Action loss = masked flow MSE; not task success rate. Full raw history: metrics.jsonl / loss_history.csv',
              'Refresh 5s | detach: Ctrl-b then d | scroll: Ctrl-b then [ | resize to expand chart']
    if state.get('error'):
        footer = textwrap.wrap('ERROR: '+state['error'], width=width)+footer
    plot_height = max(5, min(30, height-len(lines)-len(val_lines)-len(footer)-4))
    lines.extend(chart(train, validation, width, plot_height, color))
    lines.extend(val_lines)
    lines.extend(footer)
    # Chart rows contain ANSI styling; all other rows can be safely clipped.
    return '\n'.join(line if '\033[' in line else line[:width] for line in lines)


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('--run', required=True, type=Path)
    parser.add_argument('--once', action='store_true')
    parser.add_argument('--width', type=int)
    parser.add_argument('--height', type=int)
    args = parser.parse_args()
    try:
        if not args.once:
            sys.stdout.write('\033[?25l')
        while True:
            size = shutil.get_terminal_size((110, 32))
            screen = render(args.run.resolve(), args.width or size.columns,
                            args.height or size.lines, sys.stdout.isatty())
            prefix = '' if args.once else '\033[H\033[2J'
            print(prefix+screen, flush=True)
            if args.once:
                break
            time.sleep(5)
    except KeyboardInterrupt:
        pass
    finally:
        if not args.once:
            print('\033[?25h', end='', flush=True)


if __name__ == '__main__':
    main()
