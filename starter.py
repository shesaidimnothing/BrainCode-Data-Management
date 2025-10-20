import json
import random
import test_solution
import datetime
import os
import sys
import time


def solve(dataset_txt, restarts=5, rect_samples=800, joker_samples=800, max_rect_span=20, seed=None, verbose=False, max_seconds=None, optimize_for_large=False, aggressive_mode=False):
    dataset = json.loads(dataset_txt)

    target_grid = dataset['grid']
    max_actions = dataset['maxActions']
    max_jokers = dataset['maxJokers']
    max_joker_size = dataset['maxJokerSize']
    grid_height = len(target_grid)
    grid_width = len(target_grid[0])

    def evaluate_correct(grid):
        return sum(1 for y in range(grid_height) for x in range(grid_width) if grid[y][x] == target_grid[y][x])

    def apply_rect(grid, x1, y1, x2, y2, color):
        for yy in range(y1, y2 + 1):
            row = grid[yy]
            for xx in range(x1, x2 + 1):
                row[xx] = color

    def apply_joker(grid, x1, y1, x2, y2):
        for yy in range(y1, y2 + 1):
            tgt_row = target_grid[yy]
            row = grid[yy]
            for xx in range(x1, x2 + 1):
                row[xx] = tgt_row[xx]

    def rect_net_gain(grid, x1, y1, x2, y2, color):
        gain = 0
        for yy in range(y1, y2 + 1):
            for xx in range(x1, x2 + 1):
                cur = grid[yy][xx]
                tgt = target_grid[yy][xx]
                if color == tgt and cur != tgt:
                    gain += 1
                elif cur == tgt and color != tgt:
                    gain -= 1
        return gain

    def joker_gain(grid, x1, y1, x2, y2):
        gain = 0
        for yy in range(y1, y2 + 1):
            for xx in range(x1, x2 + 1):
                if grid[yy][xx] != target_grid[yy][xx]:
                    gain += 1
        return gain

    def find_best_joker_tile(grid, max_trials=300):
        mismatches = []
        # Sample more mismatches for large images
        sample_size = 400 if optimize_for_large and grid_width * grid_height > 500000 else 200
        for _ in range(sample_size):
            rx = random.randint(0, grid_width - 1)
            ry = random.randint(0, grid_height - 1)
            if grid[ry][rx] != target_grid[ry][rx]:
                mismatches.append((rx, ry))
        if not mismatches:
            return 0, None, None, None, None
        best = (0, None, None, None, None)
        trials = max(50, min(max_trials, (grid_width * grid_height) // 5000))
        if optimize_for_large and grid_width * grid_height > 500000:
            trials = max(100, trials)
        
        for _ in range(trials):
            x0, y0 = random.choice(mismatches)
            w = h = 1
            x1 = x0
            y1 = y0
            x2 = x0
            y2 = y0
            max_expansions = 30 if optimize_for_large else 20
            for _exp in range(max_expansions):
                dir_choice = random.randint(0, 3)
                nx1, ny1, nx2, ny2 = x1, y1, x2, y2
                if dir_choice == 0 and nx2 + 1 < grid_width:
                    if (nx2 + 1 - nx1 + 1) * (ny2 - ny1 + 1) <= max_joker_size:
                        nx2 += 1
                elif dir_choice == 1 and ny2 + 1 < grid_height:
                    if (nx2 - nx1 + 1) * (ny2 + 1 - ny1 + 1) <= max_joker_size:
                        ny2 += 1
                elif dir_choice == 2 and nx1 - 1 >= 0:
                    if (nx2 - (nx1 - 1) + 1) * (ny2 - ny1 + 1) <= max_joker_size:
                        nx1 -= 1
                elif dir_choice == 3 and ny1 - 1 >= 0:
                    if (nx2 - nx1 + 1) * (ny2 - (ny1 - 1) + 1) <= max_joker_size:
                        ny1 -= 1
                g_old = joker_gain(grid, x1, y1, x2, y2)
                g_new = joker_gain(grid, nx1, ny1, nx2, ny2)
                if g_new >= g_old:
                    x1, y1, x2, y2 = nx1, ny1, nx2, ny2
            g_final = joker_gain(grid, x1, y1, x2, y2)
            if g_final > best[0]:
                best = (g_final, x1, y1, x2, y2)
        return best

    def prepaint_multiscale(grid, actions, budget_actions, min_ratio=0.92, sizes=None):
        if sizes is None:
            sizes = []
            max_side = max(grid_width, grid_height)
            if optimize_for_large and max_side >= 800:
                for s in [512, 384, 256, 192, 128, 96, 64, 48, 32, 24, 16]:
                    if s <= max_side:
                        sizes.append(s)
            else:
                for s in [256, 192, 128, 96, 64, 48, 32, 24, 16, 12, 8]:
                    if s <= max_side:
                        sizes.append(s)
        used = 0
        for tile in sizes:
            if used >= budget_actions:
                break
            y = 0
            while y < grid_height:
                x = 0
                y2 = min(grid_height - 1, y + tile - 1)
                while x < grid_width:
                    x2 = min(grid_width - 1, x + tile - 1)
                    counts = [0] * 8
                    total = (y2 - y + 1) * (x2 - x + 1)
                    for yy in range(y, y2 + 1):
                        row_t = target_grid[yy]
                        for xx in range(x, x2 + 1):
                            c = row_t[xx]
                            if 0 <= c <= 7:
                                counts[c] += 1
                    best_color = max(range(8), key=lambda c: counts[c])
                    ratio = counts[best_color] / max(1, total)
                    if ratio >= min_ratio:
                        gain = rect_net_gain(grid, x, y, x2, y2, best_color)
                        if gain > 0 and (len(actions) + used) < max_actions:
                            apply_rect(grid, x, y, x2, y2, best_color)
                            actions.append(f"RECT {x} {y} {x2} {y2} {best_color}")
                            used += 1
                            if verbose and used % 50 == 0:
                                correct = evaluate_correct(grid)
                                print(f"  Prepaint {used}/{budget_actions} | size={tile} correct={correct}")
                            if used >= budget_actions:
                                break
                    x = x2 + 1
                y = y2 + 1
        return used

    def sample_rect_coords():
        if optimize_for_large and grid_width * grid_height > 500000:
            w_span = random.randint(max_rect_span//2, max_rect_span)
            h_span = random.randint(max_rect_span//2, max_rect_span)
        else:
            w_span = random.randint(0, max_rect_span)
            h_span = random.randint(0, max_rect_span)
        x1 = random.randint(0, grid_width - 1)
        y1 = random.randint(0, grid_height - 1)
        x2 = min(grid_width - 1, x1 + w_span)
        y2 = min(grid_height - 1, y1 + h_span)
        return x1, y1, x2, y2

    def sample_joker_coords():
        if max_joker_size <= 0:
            return None
        max_w = min(grid_width, max_joker_size)
        max_h = min(grid_height, max_joker_size)
        w = random.randint(1, max_w)
        max_h_for_w = min(max_h, max_joker_size // w if w > 0 else 1)
        if max_h_for_w <= 0:
            return None
        h = random.randint(1, max_h_for_w)
        x1 = random.randint(0, grid_width - w)
        y1 = random.randint(0, grid_height - h)
        x2 = x1 + w - 1
        y2 = y1 + h - 1
        return x1, y1, x2, y2

    best_overall_solution = None
    best_overall_score = -1

    base_seed = seed if seed is not None else random.randrange(1 << 30)

    start_time = time.time()
    for restart in range(restarts):
        random.seed(base_seed + restart)
        actions = []
        jokers_used = 0
        grid = [[0 for _ in range(grid_width)] for _ in range(grid_height)]

        if aggressive_mode and optimize_for_large and grid_width * grid_height > 500000:
            if verbose:
                print(f"  Starting multi-pass aggressive strategy...")
            
            joker_pass_budget = min(max_jokers * 2 // 3, max_jokers - 100)
            jokers_used_pass1 = 0
            
            for _ in range(joker_pass_budget):
                if len(actions) >= max_actions:
                    break
                    
                max_w = min(grid_width, max_joker_size)
                max_h = min(grid_height, max_joker_size)
                
                w = random.randint(max_w * 3 // 4, max_w)
                max_h_for_w = min(max_h, max_joker_size // w if w > 0 else 1)
                if max_h_for_w <= 0:
                    continue
                h = random.randint(max_h_for_w * 3 // 4, max_h_for_w)
                
                x1 = random.randint(0, grid_width - w)
                y1 = random.randint(0, grid_height - h)
                x2 = x1 + w - 1
                y2 = y1 + h - 1
                
                gain = joker_gain(grid, x1, y1, x2, y2)
                if gain > 30:
                    apply_joker(grid, x1, y1, x2, y2)
                    actions.append(f"JOKER {x1} {y1} {x2} {y2}")
                    jokers_used += 1
                    jokers_used_pass1 += 1
                    
                    if verbose and jokers_used_pass1 % 100 == 0:
                        correct = evaluate_correct(grid)
                        print(f"  Pass 1 JOKER x{jokers_used_pass1} -> correct={correct}")
            
            if verbose:
                print(f"  Pass 1 completed: {jokers_used_pass1} JOKERs")
            
            if len(actions) < max_actions:
                if verbose:
                    print(f"  Starting Pass 2: Hierarchical RECT tiling...")
                
                large_tile_sizes = [128, 96, 64, 48, 32, 24, 16, 12, 8]
                tiles_added_pass2 = 0
                
                for tile_size in large_tile_sizes:
                    if len(actions) >= max_actions:
                        break
                        
                    tiles_in_size = 0
                    for y in range(0, grid_height, tile_size):
                        for x in range(0, grid_width, tile_size):
                            if len(actions) >= max_actions:
                                break
                                
                            x2 = min(grid_width - 1, x + tile_size - 1)
                            y2 = min(grid_height - 1, y + tile_size - 1)
                            
                            color_counts = [0] * 8
                            for yy in range(y, y2 + 1):
                                for xx in range(x, x2 + 1):
                                    color_counts[target_grid[yy][xx]] += 1
                            
                            best_color = max(range(8), key=lambda c: color_counts[c])
                            area = (x2 - x + 1) * (y2 - y + 1)
                            dominance = color_counts[best_color] / area
                            
                            if dominance >= 0.6:
                                gain = rect_net_gain(grid, x, y, x2, y2, best_color)
                                if gain > 0:
                                    apply_rect(grid, x, y, x2, y2, best_color)
                                    actions.append(f"RECT {x} {y} {x2} {y2} {best_color}")
                                    tiles_in_size += 1
                                    tiles_added_pass2 += 1
                    
                    if verbose and tiles_in_size > 0:
                        correct = evaluate_correct(grid)
                        print(f"  Pass 2 tiles (size {tile_size}): {tiles_in_size} tiles -> correct={correct}")
                
                if verbose:
                    print(f"  Pass 2 completed: {tiles_added_pass2} RECT tiles")

        elif aggressive_mode and optimize_for_large and grid_width * grid_height > 500000:
            if verbose:
                print(f"  Starting aggressive JOKER-first strategy...")
            
            joker_budget = min(max_jokers - 50, max_jokers * 3 // 4)
            jokers_used_early = 0
            
            for _ in range(joker_budget):
                if len(actions) >= max_actions:
                    break
                    
                max_w = min(grid_width, max_joker_size)
                max_h = min(grid_height, max_joker_size)
                
                w = random.randint(max_w//2, max_w)
                max_h_for_w = min(max_h, max_joker_size // w if w > 0 else 1)
                if max_h_for_w <= 0:
                    continue
                h = random.randint(max_h_for_w//2, max_h_for_w)
                
                x1 = random.randint(0, grid_width - w)
                y1 = random.randint(0, grid_height - h)
                x2 = x1 + w - 1
                y2 = y1 + h - 1
                
                gain = joker_gain(grid, x1, y1, x2, y2)
                if gain > 50:
                    apply_joker(grid, x1, y1, x2, y2)
                    actions.append(f"JOKER {x1} {y1} {x2} {y2}")
                    jokers_used += 1
                    jokers_used_early += 1
                    
                    if verbose and jokers_used_early % 50 == 0:
                        correct = evaluate_correct(grid)
                        print(f"  Aggressive JOKER x{jokers_used_early} -> correct={correct}")
            
            if verbose:
                print(f"  Aggressive JOKER phase used {jokers_used_early} JOKERs")

        pre_budget = max(0, min(max_actions // 2, max_actions - 1))
        pre_used = prepaint_multiscale(grid, actions, pre_budget)
        if verbose and pre_used > 0:
            print(f"  Prepaint used {pre_used} actions")

        while len(actions) < max_actions:
            if max_seconds is not None and max_seconds > 0 and (time.time() - start_time) > max_seconds:
                if verbose:
                    print(f"Time budget reached, stopping early (restart {restart+1}/{restarts}).")
                break
            best_action = None
            best_gain = 0

            if jokers_used < max_jokers and max_joker_size > 0:
                for _ in range(joker_samples):
                    coords = sample_joker_coords()
                    if coords is None:
                        continue
                    x1, y1, x2, y2 = coords
                    g = joker_gain(grid, x1, y1, x2, y2)
                    if g > best_gain:
                        best_gain = g
                        best_action = ("JOKER", x1, y1, x2, y2, None)

            for _ in range(rect_samples):
                x1, y1, x2, y2 = sample_rect_coords()
                color_counts = [0] * 8
                for yy in range(y1, y2 + 1):
                    for xx in range(x1, x2 + 1):
                        tgt = target_grid[yy][xx]
                        if 0 <= tgt <= 7:
                            color_counts[tgt] += 1
                
                if optimize_for_large and grid_width * grid_height > 500000:
                    priority_colors = [2, 0, 1] + [c for c in range(8) if c not in [2, 0, 1]]
                    ordered_colors = sorted(priority_colors, key=lambda c: -color_counts[c])
                else:
                    ordered_colors = sorted(range(8), key=lambda c: -color_counts[c])
                
                for color in ordered_colors:
                    g = rect_net_gain(grid, x1, y1, x2, y2, color)
                    if g > best_gain:
                        best_gain = g
                        best_action = ("RECT", x1, y1, x2, y2, color)

            if best_action is None or best_gain <= 0:
                break

            kind, x1, y1, x2, y2, color = best_action
            if kind == "JOKER":
                apply_joker(grid, x1, y1, x2, y2)
                actions.append(f"JOKER {x1} {y1} {x2} {y2}")
                jokers_used += 1
            else:
                apply_rect(grid, x1, y1, x2, y2, color)
                actions.append(f"RECT {x1} {y1} {x2} {y2} {color}")

            if verbose and (len(actions) % 2 == 0):
                correct = evaluate_correct(grid)
                print(f"  Step {len(actions)}/{max_actions} | correct={correct}")

        post_jokers_cap = min(max_jokers - jokers_used, 100 if optimize_for_large else 50)
        used_in_post = 0
        while len(actions) < max_actions and jokers_used < max_jokers and used_in_post < post_jokers_cap:
            gain, jx1, jy1, jx2, jy2 = find_best_joker_tile(grid)
            if gain <= 0 or jx1 is None:
                break
            apply_joker(grid, jx1, jy1, jx2, jy2)
            actions.append(f"JOKER {jx1} {jy1} {jx2} {jy2}")
            jokers_used += 1
            used_in_post += 1
            if verbose and (used_in_post % 10 == 0):
                correct = evaluate_correct(grid)
                print(f"  Post JOKER x{used_in_post} -> correct={correct}")

        if optimize_for_large and len(actions) < max_actions:
            if verbose:
                print(f"  Starting advanced post-processing...")
            
            remaining_actions = max_actions - len(actions)
            targeted_rects = 0
            
            if aggressive_mode:
                tile_sizes = [64, 32, 16, 8, 4] if grid_width * grid_height > 500000 else [32, 16, 8, 4]
                
                for tile_size in tile_sizes:
                    if len(actions) >= max_actions:
                        break
                        
                    tiles_added = 0
                    for y in range(0, grid_height, tile_size):
                        for x in range(0, grid_width, tile_size):
                            if len(actions) >= max_actions:
                                break
                                
                            x2 = min(grid_width - 1, x + tile_size - 1)
                            y2 = min(grid_height - 1, y + tile_size - 1)
                            
                            color_counts = [0] * 8
                            for yy in range(y, y2 + 1):
                                for xx in range(x, x2 + 1):
                                    color_counts[target_grid[yy][xx]] += 1
                            
                            best_color = max(range(8), key=lambda c: color_counts[c])
                            area = (x2 - x + 1) * (y2 - y + 1)
                            dominance = color_counts[best_color] / area
                            
                            if dominance >= 0.8:
                                gain = rect_net_gain(grid, x, y, x2, y2, best_color)
                                if gain > 0:
                                    apply_rect(grid, x, y, x2, y2, best_color)
                                    actions.append(f"RECT {x} {y} {x2} {y2} {best_color}")
                                    tiles_added += 1
                                    targeted_rects += 1
                    
                    if verbose and tiles_added > 0:
                        correct = evaluate_correct(grid)
                        print(f"  Hierarchical tiles (size {tile_size}): {tiles_added} tiles -> correct={correct}")
            
            for _ in range(min(remaining_actions - targeted_rects, 300)):
                if len(actions) >= max_actions:
                    break
                    
                mismatches = []
                for _ in range(200):
                    rx = random.randint(0, grid_width - 1)
                    ry = random.randint(0, grid_height - 1)
                    if grid[ry][rx] != target_grid[ry][rx]:
                        mismatches.append((rx, ry))
                
                if not mismatches:
                    break
                    
                x0, y0 = random.choice(mismatches)
                target_color = target_grid[y0][x0]
                
                x1, y1, x2, y2 = x0, y0, x0, y0
                
                max_expansions = 25 if aggressive_mode else 15
                for _ in range(max_expansions):
                    best_expansion = None
                    best_gain = rect_net_gain(grid, x1, y1, x2, y2, target_color)
                    
                    for direction in range(4):
                        nx1, ny1, nx2, ny2 = x1, y1, x2, y2
                        if direction == 0 and nx2 + 1 < grid_width:
                            nx2 += 1
                        elif direction == 1 and ny2 + 1 < grid_height:
                            ny2 += 1
                        elif direction == 2 and nx1 - 1 >= 0:
                            nx1 -= 1
                        elif direction == 3 and ny1 - 1 >= 0:
                            ny1 -= 1
                        else:
                            continue
                            
                        area = (nx2 - nx1 + 1) * (ny2 - ny1 + 1)
                        target_count = 0
                        for yy in range(ny1, ny2 + 1):
                            for xx in range(nx1, nx2 + 1):
                                if target_grid[yy][xx] == target_color:
                                    target_count += 1
                        
                        threshold = 0.6 if aggressive_mode else 0.7
                        if target_count >= area * threshold:
                            gain = rect_net_gain(grid, nx1, ny1, nx2, ny2, target_color)
                            if gain > best_gain:
                                best_gain = gain
                                best_expansion = (nx1, ny1, nx2, ny2)
                    
                    if best_expansion:
                        x1, y1, x2, y2 = best_expansion
                    else:
                        break
                
                gain = rect_net_gain(grid, x1, y1, x2, y2, target_color)
                if gain > 0:
                    apply_rect(grid, x1, y1, x2, y2, target_color)
                    actions.append(f"RECT {x1} {y1} {x2} {y2} {target_color}")
                    targeted_rects += 1
                    
                    if verbose and targeted_rects % 50 == 0:
                        correct = evaluate_correct(grid)
                        print(f"  Targeted RECT x{targeted_rects} -> correct={correct}")
            
            if verbose and targeted_rects > 0:
                print(f"  Advanced post-processing added {targeted_rects} targeted RECT operations")

        solution_txt = "\n".join(actions)
        score, is_valid, _ = test_solution.get_solution_score(solution_txt, dataset_txt)
        if is_valid and score > best_overall_score:
            best_overall_score = score
            best_overall_solution = solution_txt

    return best_overall_solution or ""


if __name__ == '__main__':
    datasets_dir = 'datasets'
    solutions_dir = 'solutions'
    os.makedirs(solutions_dir, exist_ok=True)

    dataset_files = [f for f in os.listdir(datasets_dir) if f.endswith('.json')]
    dataset_files.sort()

    only_set = None
    time_override = None
    nolimit = False
    for arg in sys.argv[1:]:
        if arg.startswith('--only='):
            bases = arg.split('=', 1)[1]
            only_set = set(b.strip() for b in bases.split(',') if b.strip())
        elif arg.startswith('--time='):
            try:
                time_override = int(arg.split('=', 1)[1])
            except:
                time_override = None
        elif arg == '--nolimit':
            nolimit = True

    for dataset_file in dataset_files:
        dataset_path = os.path.join(datasets_dir, dataset_file)
        dataset_txt = open(dataset_path, 'r').read()

        print('---------------------------------')
        print(f'Solving {dataset_file}')
        data = json.loads(dataset_txt)
        h = len(data['grid'])
        w = len(data['grid'][0])
        area = max(1, w * h)
        scale = min(1.0, (200*200) / area)
        rect_samples = max(200, int(1000 * scale))
        joker_samples = max(200, int(1000 * scale))
        max_rect_span = max(8, int(min(w, h) * 0.25))
        
        base_name = os.path.splitext(dataset_file)[0]
        if base_name == '6_wplace':
            rect_samples = max(1200, rect_samples)
            joker_samples = max(1200, joker_samples)
            max_rect_span = max(40, max_rect_span)
        if area <= 200*200:
            time_budget = 20
        elif area <= 300*300:
            time_budget = 35
        elif area <= 500*500:
            time_budget = 60
        else:
            time_budget = 90
        if time_override is not None:
            time_budget = time_override
        if nolimit or (time_override is not None and time_override <= 0):
            time_budget = None

        if only_set is not None:
            base_name = os.path.splitext(dataset_file)[0]
            if base_name not in only_set:
                print('Skipping (filtered by --only)')
                continue

        optimize_large = (base_name == '6_wplace')
        aggressive_mode = (base_name == '6_wplace')
        
        print(f"  Grid: {w}x{h}, area={area}, rect_samples={rect_samples}, joker_samples={joker_samples}, max_rect_span={max_rect_span}, optimize_large={optimize_large}, aggressive={aggressive_mode}")
        solution = solve(dataset_txt, restarts=5, rect_samples=rect_samples, joker_samples=joker_samples, max_rect_span=max_rect_span, verbose=True, max_seconds=time_budget, optimize_for_large=optimize_large, aggressive_mode=aggressive_mode)
        score, is_valid, message = test_solution.get_solution_score(solution, dataset_txt)
        print(message)
        if is_valid:
            date = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
            base = os.path.splitext(dataset_file)[0]
            out_name = f'{base}_{score}_{date}.txt'
            out_path = os.path.join(solutions_dir, out_name)
            with open(out_path, 'w') as f:
                f.write(solution)
            print(f'Saved: {out_path}')
        else:
            print('Solution invalid, not saved')



