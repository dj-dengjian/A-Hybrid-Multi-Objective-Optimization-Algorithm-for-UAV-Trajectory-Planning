import numpy as np
import matplotlib.pyplot as plt
from matplotlib.path import Path
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
import os
import math
import random
from scipy.interpolate import make_interp_spline
import scipy.stats as stats
import pandas as pd

hill_count = 4
hill_amplitude = [110, 120, 120, 100]
hill_width = [400, 200, 300, 200]
hill_center_x = [500, 1500, 2800, 2000]
hill_center_y = [500, 800, 2200, 2800]
noise_intensity = 15
base_height = 20

radar_count = 5
radars = [
    {"pos": (1800, 1200), "radius": 200, "rings": 3, "height": 120},
    {"pos": (2800, 1200), "radius": 200, "rings": 3, "height": 120},
    {"pos": (800, 1800), "radius": 200, "rings": 3, "height": 120},
    {"pos": (550, 3000), "radius": 200, "rings": 3, "height": 120},
    {"pos": (1500, 2500), "radius": 200, "rings": 3, "height": 120}
]

ADAPTIVE_PC_INIT = 0.75
ADAPTIVE_PC_FINAL = 0.65
ADAPTIVE_PM_BASE = 0.25
ADAPTIVE_PM_MAX = 0.3
LOCAL_SEARCH_RATIO = 0.2
DIVERSITY_THRESHOLD = 0.15

def generate_terrain(x_range=(0, 3500), y_range=(0, 3500), grid_size=100):
    x = np.linspace(x_range[0], x_range[1], grid_size)
    y = np.linspace(y_range[0], y_range[1], grid_size)
    X, Y = np.meshgrid(x, y)
    Z = base_height + noise_intensity * np.random.rand(*X.shape)
    for i in range(hill_count):
        Z += hill_amplitude[i] * np.exp(-(((X - hill_center_x[i]) ** 2) / (2 * hill_width[i] ** 2) +
                                          ((Y - hill_center_y[i]) ** 2) / (2 * hill_width[i] ** 2)))
    Z = np.clip(Z, 0, 120)
    return x, y, X, Y, Z

def original_crossover(parent1, parent2, crossover_rate=0.8):
    if random.random() > crossover_rate:
        return parent1, parent2
    child1_chrom = []
    child2_chrom = []
    for g1, g2 in zip(parent1.chromosome, parent2.chromosome):
        if random.random() < 0.5:
            child1_chrom.append(g1)
            child2_chrom.append(g2)
        else:
            child1_chrom.append(g2)
            child2_chrom.append(g1)
    return Individual(child1_chrom), Individual(child2_chrom)

def original_mutate(individual, mutation_rate=0.1, mutation_range=(-50, 50)):
    new_chrom = []
    for i, gene in enumerate(individual.chromosome):
        if random.random() < mutation_rate:
            new_gene = gene + random.uniform(*mutation_range)
            if (i + 1) % 3 == 0:
                new_gene = np.clip(new_gene, 0, 120)
            new_chrom.append(new_gene)
        else:
            new_chrom.append(gene)
    return Individual(new_chrom)

def generate_start_target(x, y, Z, safe_height=50):
    start_xy = (200, 3200)
    start_idx_x = np.argmin(np.abs(x - start_xy[0]))
    start_idx_y = np.argmin(np.abs(y - start_xy[1]))
    start_z = Z[start_idx_y, start_idx_x] + safe_height
    start = (start_xy[0], start_xy[1], start_z)

    target_xy = (3200, 500)
    target_idx_x = np.argmin(np.abs(x - target_xy[0]))
    target_idx_y = np.argmin(np.abs(y - target_xy[1]))
    target_z = Z[target_idx_y, target_idx_x] + safe_height
    target = (target_xy[0], target_xy[1], target_z)

    return start, target

class Individual:
    def __init__(self, chromosome):
        self.chromosome = chromosome
        self.objectives = []
        self.rank = None
        self.crowding_distance = 0
        self.strength = 0
        self.raw_fitness = 0
        self.density = 0

def decode_chromosome(chromosome, start, target, num_points=50):
    num_control_points = len(chromosome) // 3
    points = [(start[0], start[1], start[2])]
    for i in range(num_control_points):
        points.append((
            chromosome[i * 3],
            chromosome[i * 3 + 1],
            chromosome[i * 3 + 2]
        ))
    points.append((target[0], target[1], target[2]))
    t = np.linspace(0, 1, len(points))
    t_new = np.linspace(0, 1, num_points)
    spl_x = make_interp_spline(t, [p[0] for p in points], k=3)
    spl_y = make_interp_spline(t, [p[1] for p in points], k=3)
    spl_z = make_interp_spline(t, [p[2] for p in points], k=3)
    path_x = spl_x(t_new)
    path_y = spl_y(t_new)
    path_z = spl_z(t_new)
    path_x = np.clip(path_x, 0, 3400)
    path_y = np.clip(path_y, 0, 3400)
    path_z = np.clip(path_z, 0, 110)
    return np.column_stack((path_x, path_y, path_z))

def calculate_distance(path):
    return np.sum(np.sqrt(np.sum(np.diff(path, axis=0) ** 2, axis=1)))

def calculate_collision_risk(path, radars, terrain, x_grid, y_grid, safe_distance=200):
    risk = 0
    for point in path:
        idx_x = np.argmin(np.abs(x_grid - point[0]))
        idx_y = np.argmin(np.abs(y_grid - point[1]))
        terrain_z = terrain[idx_y, idx_x]
        height_diff = terrain_z + 50 - point[2]
        if height_diff > 0:
            risk += 200 * height_diff / 50
    for radar in radars:
        rx, ry = radar["pos"]
        radius = radar["radius"]
        height = radar["height"]
        terrain_z = terrain[np.argmin(np.abs(y_grid - ry)), np.argmin(np.abs(x_grid - rx))]
        bottom_z = terrain_z + 1
        top_z = bottom_z + height
        for point in path:
            dist_horiz = math.sqrt((point[0] - rx) ** 2 + (point[1] - ry) ** 2)
            if dist_horiz <= radius and point[2] >= bottom_z and point[2] <= top_z:
                risk += 1500
            elif radius < dist_horiz <= radius + safe_distance:
                proximity = (radius + safe_distance - dist_horiz) / safe_distance
                if proximity > 0.8:
                    risk += 300 * proximity
                elif proximity > 0.5:
                    risk += 200 * proximity
                else:
                    risk += 100 * proximity
    return risk

def calculate_smoothness(path):
    diff1 = np.diff(path, axis=0)
    diff2 = np.diff(diff1, axis=0)
    curvature = []
    for i in range(len(diff2)):
        cross = np.cross(diff1[i], diff2[i])
        norm_cross = np.linalg.norm(cross)
        norm_diff1 = np.linalg.norm(diff1[i]) ** 3
        curvature.append(norm_cross / norm_diff1 if norm_diff1 > 0 else 0)
    return np.sum(np.abs(np.diff(curvature)))

def evaluate_individual(individual, start, target, radars, terrain, x_grid, y_grid):
    path = decode_chromosome(individual.chromosome, start, target)
    distance = calculate_distance(path)
    collision_risk = calculate_collision_risk(path, radars, terrain, x_grid, y_grid)
    smoothness = calculate_smoothness(path)
    individual.objectives = [distance, collision_risk, smoothness]
    return individual

def fast_non_dominated_sort(population):
    fronts = [[]]
    S = [[] for _ in range(len(population))]
    n = [0 for _ in range(len(population))]
    for p in range(len(population)):
        for q in range(len(population)):
            if p == q:
                continue
            p_dominates_q = all(population[p].objectives[i] <= population[q].objectives[i] for i in range(3)) and \
                            any(population[p].objectives[i] < population[q].objectives[i] for i in range(3))
            q_dominates_p = all(population[q].objectives[i] <= population[p].objectives[i] for i in range(3)) and \
                            any(population[q].objectives[i] < population[p].objectives[i] for i in range(3))
            if p_dominates_q:
                S[p].append(q)
            elif q_dominates_p:
                n[p] += 1
        if n[p] == 0:
            population[p].rank = 0
            fronts[0].append(p)
    i = 0
    while fronts[i]:
        Q = []
        for p in fronts[i]:
            for q in S[p]:
                n[q] -= 1
                if n[q] == 0:
                    population[q].rank = i + 1
                    Q.append(q)
        i += 1
        fronts.append(Q)
    return fronts

def crowding_distance_assignment(front, population):
    distance = [0.0] * len(front)
    num_objectives = 3
    for m in range(num_objectives):
        sorted_indices = sorted(enumerate(front), key=lambda x: population[x[1]].objectives[m])
        if not sorted_indices:
            continue
        distance[sorted_indices[0][0]] = 1e6
        distance[sorted_indices[-1][0]] = 1e6
        min_val = population[sorted_indices[0][1]].objectives[m]
        max_val = population[sorted_indices[-1][1]].objectives[m]
        if max_val - min_val < 1e-9:
            continue
        for k in range(1, len(sorted_indices) - 1):
            prev_idx = sorted_indices[k - 1][1]
            next_idx = sorted_indices[k + 1][1]
            delta = population[next_idx].objectives[m] - population[prev_idx].objectives[m]
            distance[sorted_indices[k][0]] += max(delta / (max_val - min_val), 0.0)
    for i, idx in enumerate(front):
        population[idx].crowding_distance = distance[i]

def tournament_selection(population, tournament_size=2):
    new_population = []
    for _ in range(len(population)):
        competitors = random.sample(population, tournament_size)
        min_rank = min(c.rank for c in competitors)
        candidates = [c for c in competitors if c.rank == min_rank]
        if len(candidates) == 1:
            new_population.append(candidates[0])
        else:
            new_population.append(max(candidates, key=lambda x: x.crowding_distance))
    return new_population

def calculate_fitness(individual):
    return 0.5 * individual.objectives[0] + 0.3 * individual.objectives[1] + 0.2 * individual.objectives[2]

def get_elite_guide(front_individuals, top_k=3):
    if len(front_individuals) == 0:
        return None
    sorted_by_crowding = sorted(front_individuals, key=lambda ind: ind.crowding_distance, reverse=True)
    elites = sorted_by_crowding[:top_k]
    chroms = np.array([ind.chromosome for ind in elites])
    return np.mean(chroms, axis=0)

def dynamic_guide_population_enhanced(population, gen, max_gen, start, target, radars, terrain, x_grid, y_grid):
    fronts = fast_non_dominated_sort(population)
    if len(fronts[0]) == 0:
        return population
    front_individuals = [population[i] for i in fronts[0]]
    progress = gen / max_gen
    guide_strength = 0.5 - 0.4 * progress
    guide_mean = get_elite_guide(front_individuals, top_k=min(5, len(front_individuals)))
    if guide_mean is None:
        return population
    guide_ratio = 0.15
    guide_count = int(len(population) * guide_ratio)
    individual_scores = []
    for ind in population:
        score = 0
        if ind.rank > 0:
            score += 2
        elif ind.crowding_distance < 0.1:
            score += 1
        individual_scores.append(score)
    indices_by_score = np.argsort(individual_scores)[::-1]
    guide_indices = indices_by_score[:guide_count]
    for idx in guide_indices:
        new_chrom = np.array(population[idx].chromosome) + guide_strength * (
                guide_mean - np.array(population[idx].chromosome))
        new_chrom[2::3] = np.clip(new_chrom[2::3], 0, 120)
        new_chrom[0::3] = np.clip(new_chrom[0::3], 0, 3500)
        new_chrom[1::3] = np.clip(new_chrom[1::3], 0, 3500)
        temp_ind = Individual(new_chrom.tolist())
        population[idx] = evaluate_individual(temp_ind, start, target, radars, terrain, x_grid, y_grid)
    diversity = calculate_population_diversity(population)
    if diversity < DIVERSITY_THRESHOLD:
        perturb_count = int(len(population) * 0.15)
        for _ in range(perturb_count):
            base_ind = random.choice(front_individuals)
            perturb_chrom = np.array(base_ind.chromosome)
            mutation_range = 200
            perturb_chrom += np.random.uniform(-mutation_range, mutation_range, size=len(perturb_chrom))
            perturb_chrom[2::3] = np.clip(perturb_chrom[2::3], 0, 120)
            perturb_chrom[0::3] = np.clip(perturb_chrom[0::3], 0, 3500)
            perturb_chrom[1::3] = np.clip(perturb_chrom[1::3], 0, 3500)
            perturb_ind = Individual(perturb_chrom.tolist())
            perturb_ind = evaluate_individual(perturb_ind, start, target, radars, terrain, x_grid, y_grid)
            worst_rank = max(ind.rank for ind in population)
            worst_candidates = [i for i, ind in enumerate(population) if ind.rank == worst_rank]
            if not worst_candidates:
                continue
            min_crowd_idx_in_worst = min(worst_candidates, key=lambda i: population[i].crowding_distance)
            population[min_crowd_idx_in_worst] = perturb_ind
    return population

def calculate_population_diversity(population):
    objectives = np.array([ind.objectives for ind in population])
    obj_min = np.min(objectives, axis=0)
    obj_max = np.max(objectives, axis=0)
    obj_range = obj_max - obj_min
    obj_range[obj_range == 0] = 1.0
    norm_obj = (objectives - obj_min) / obj_range
    pairwise_dist = np.sqrt(np.sum((norm_obj[:, None] - norm_obj[None, :]) ** 2, axis=2))
    upper_tri = pairwise_dist[np.triu_indices_from(pairwise_dist, k=1)]
    return np.std(upper_tri) if len(upper_tri) > 0 else 0.0

def adaptive_crossover(parent1, parent2, gen, max_gen, population, crossover_rate_init=ADAPTIVE_PC_INIT,
                       crossover_rate_final=ADAPTIVE_PC_FINAL):
    base_pc = crossover_rate_init - (gen / max_gen) * (crossover_rate_init - crossover_rate_final)
    diversity = calculate_population_diversity(population)
    diversity_factor = 1.0 + (
            DIVERSITY_THRESHOLD - diversity) / DIVERSITY_THRESHOLD if diversity < DIVERSITY_THRESHOLD else 1.0
    crossover_rate = base_pc * diversity_factor
    crossover_rate = np.clip(crossover_rate, 0.5, 0.95)
    if random.random() > crossover_rate:
        return parent1, parent2
    num_control = len(parent1.chromosome) // 3
    if num_control <= 1:
        cross_point = 3
    else:
        cross_point = random.randint(1, num_control - 1) * 3
    child1_chrom = parent1.chromosome[:cross_point] + parent2.chromosome[cross_point:]
    child2_chrom = parent2.chromosome[:cross_point] + parent1.chromosome[cross_point:]
    return Individual(child1_chrom), Individual(child2_chrom)

def adaptive_mutate(individual, gen, max_gen, population, mutation_rate_base=ADAPTIVE_PM_BASE,
                    mutation_rate_max=ADAPTIVE_PM_MAX):
    diversity = calculate_population_diversity(population)
    valid_parent_ranks = [ind.rank for ind in population if ind.rank is not None]
    avg_parent_rank = np.mean(valid_parent_ranks) if valid_parent_ranks else 2.0
    base_pm = mutation_rate_base - (gen / max_gen) * (mutation_rate_base - 0.01)
    diversity_factor = 1.0 + (
            DIVERSITY_THRESHOLD - diversity) / DIVERSITY_THRESHOLD if diversity < DIVERSITY_THRESHOLD else 1.0
    crowding_factor = 2.0 - np.clip(individual.crowding_distance / 1000, 0, 1.0) if not np.isinf(
        individual.crowding_distance) else 2.0
    mutation_rate = base_pm * diversity_factor * crowding_factor
    mutation_rate = np.clip(mutation_rate, 0.01, 0.4)
    base_step = 80 - 60 * (gen / max_gen)
    individual_rank = individual.rank if individual.rank is not None else avg_parent_rank
    rank_factor = 1.0 + individual_rank * 0.1
    mutate_range = base_step * rank_factor
    mutate_range = np.clip(mutate_range, 5, 100)
    new_chrom = np.array(individual.chromosome, dtype=float)
    for i in range(len(new_chrom)):
        if random.random() < mutation_rate:
            mutation = random.uniform(-mutate_range, mutate_range)
            new_chrom[i] += mutation
    new_chrom[2::3] = np.clip(new_chrom[2::3], 0, 120)
    new_chrom[0::3] = np.clip(new_chrom[0::3], 0, 3500)
    new_chrom[1::3] = np.clip(new_chrom[1::3], 0, 3500)
    return Individual(new_chrom.tolist())

def is_point_safe(xy, radars, safe_distance=250):
    x, y = xy
    for radar in radars:
        rx, ry = radar["pos"]
        dist = math.hypot(x - rx, y - ry)
        if dist <= radar["radius"] + safe_distance:
            return False
    return True

def modified_generate_individual(num_control_points, start, target, radars, x_range, y_range, z_range):
    chromosome = []
    for i in range(num_control_points):
        t = (i + 1) / (num_control_points + 1)
        base_x = start[0] + t * (target[0] - start[0])
        base_y = start[1] + t * (target[1] - start[1])
        if not is_point_safe((base_x, base_y), radars):
            dx = random.uniform(-300, 300)
            dy = random.uniform(-300, 300)
            base_x += dx
            base_y += dy
            base_x = np.clip(base_x, x_range[0], x_range[1])
            base_y = np.clip(base_y, y_range[0], y_range[1])
        base_z = start[2] + t * (target[2] - start[2])
        base_z = np.clip(base_z, z_range[0], z_range[1])
        chromosome.extend([base_x, base_y, base_z])
    temp_ind = Individual(chromosome)
    path = decode_chromosome(temp_ind.chromosome, start, target, num_points=30)
    collision_risk = calculate_collision_risk(path, radars, Z, x, y, safe_distance=200)
    if collision_risk > 5000:
        return modified_generate_individual(num_control_points, start, target, radars, x_range, y_range, z_range)
    return Individual(chromosome)

def original_generate_individual(num_control_points, x_range, y_range, z_range):
    chromosome = []
    for _ in range(num_control_points):
        chromosome.append(random.uniform(x_range[0], x_range[1]))
        chromosome.append(random.uniform(y_range[0], y_range[1]))
        chromosome.append(random.uniform(z_range[0], z_range[1]))
    return Individual(chromosome)

def sa_local_search(individual, start, target, radars, terrain, x_grid, y_grid, T=100, alpha=0.9, max_steps=20):
    current_ind = individual
    current_obj = current_ind.objectives.copy()
    for step in range(max_steps):
        new_chrom = np.array(current_ind.chromosome)
        for i in range(len(new_chrom) // 3):
            new_chrom[i * 3:i * 3 + 3] += np.random.uniform(-50, 50, 3)
        new_chrom[2::3] = np.clip(new_chrom[2::3], 0, 120)
        new_chrom[0::3] = np.clip(new_chrom[0::3], 0, 3500)
        new_chrom[1::3] = np.clip(new_chrom[1::3], 0, 3500)
        new_ind = Individual(new_chrom.tolist())
        new_ind = evaluate_individual(new_ind, start, target, radars, terrain, x_grid, y_grid)
        new_obj = new_ind.objectives
        weight = [0.4, 0.5, 0.1]
        delta = sum(w * (new - current) for w, new, current in zip(weight, new_obj, current_obj))
        if delta < 0 or np.random.rand() < np.exp(-delta / T):
            current_ind = new_ind
            current_obj = new_obj.copy()
        T *= alpha
    return current_ind

def enhance_pareto_solutions(population, start, target, radars, terrain, x_grid, y_grid):
    fronts = fast_non_dominated_sort(population)
    if len(fronts[0]) == 0:
        return population
    front_indices = fronts[0]
    search_count = int(len(front_indices) * LOCAL_SEARCH_RATIO)
    search_indices = random.sample(front_indices, search_count)
    for idx in search_indices:
        population[idx] = sa_local_search(population[idx], start, target, radars, terrain, x_grid, y_grid)
    return population

def nsga2(start, target, radars, terrain, x_grid, y_grid, pop_size=50, generations=30, num_control_points=3):
    population = []
    x_range = (min(start[0], target[0]) - 500, max(start[0], target[0]) + 500)
    y_range = (min(start[1], target[1]) - 500, max(start[1], target[1]) + 500)
    z_range = (max(min(start[2], target[2]) - 10, 0), 120)
    for _ in range(pop_size):
        individual = original_generate_individual(num_control_points, x_range, y_range, z_range)
        population.append(evaluate_individual(individual, start, target, radars, terrain, x_grid, y_grid))
    gen_avg_fitness = []
    gen_best_fitness = []
    initial_fitness = [calculate_fitness(ind) for ind in population]
    gen_avg_fitness.append(np.mean(initial_fitness))
    initial_fronts = fast_non_dominated_sort(population)
    initial_front_fitness = [calculate_fitness(population[i]) for i in initial_fronts[0]]
    global_best_fitness = min(initial_front_fitness)
    gen_best_fitness.append(global_best_fitness)
    gen_avg_fitness.append(np.mean(initial_front_fitness))
    for gen in range(generations):
        if (gen + 1) % (generations // 10) == 0 or (gen + 1) == generations:
            progress = ((gen + 1) / generations) * 100
            print(f"NSGA2 iteration progress: {progress:.0f}%")
        fronts = fast_non_dominated_sort(population)
        for front in fronts:
            if not front:
                break
            crowding_distance_assignment(front, population)
        mating_pool = tournament_selection(population)
        offspring = []
        for i in range(0, pop_size, 2):
            parent1 = mating_pool[i]
            parent2 = mating_pool[i + 1] if (i + 1) < pop_size else mating_pool[i]
            child1, child2 = original_crossover(parent1, parent2)
            child1 = original_mutate(child1)
            child2 = original_mutate(child2)
            offspring.append(evaluate_individual(child1, start, target, radars, terrain, x_grid, y_grid))
            offspring.append(evaluate_individual(child2, start, target, radars, terrain, x_grid, y_grid))
        combined = population + offspring
        combined_fronts = fast_non_dominated_sort(combined)
        for front in combined_fronts:
            if not front:
                break
            crowding_distance_assignment(front, combined)
        new_pop = []
        for front in combined_fronts:
            if not front:
                break
            if len(new_pop) + len(front) <= pop_size:
                new_pop.extend([combined[i] for i in front])
            else:
                front_sorted = sorted(front, key=lambda x: combined[x].crowding_distance, reverse=True)
                new_pop.extend([combined[i] for i in front_sorted[:pop_size - len(new_pop)]])
                break
        population = new_pop
        current_fronts = fast_non_dominated_sort(population)
        current_front_fitness = [calculate_fitness(population[i]) for i in current_fronts[0]]
        current_min = min(current_front_fitness)
        if current_min < global_best_fitness:
            global_best_fitness = current_min
        gen_best_fitness.append(global_best_fitness)
        gen_avg_fitness.append(np.mean(current_front_fitness))
    final_fronts = fast_non_dominated_sort(population)
    best_front = [population[i] for i in final_fronts[0]]
    best_individual = min(best_front, key=lambda ind: calculate_fitness(ind))
    final_best_fitness = calculate_fitness(best_individual)
    return decode_chromosome(best_individual.chromosome, start,
                             target), gen_avg_fitness, gen_best_fitness, population, final_best_fitness, final_fronts

def g_insga2(start, target, radars, terrain, x_grid, y_grid, pop_size=50, generations=30, num_control_points=3):
    population = []
    x_range = (min(start[0], target[0]) - 500, max(start[0], target[0]) + 500)
    y_range = (min(start[1], target[1]) - 500, max(start[1], target[1]) + 500)
    z_range = (max(min(start[2], target[2]) - 10, 0), 120)
    for _ in range(pop_size):
        individual = original_generate_individual(num_control_points, x_range, y_range, z_range)
        population.append(evaluate_individual(individual, start, target, radars, terrain, x_grid, y_grid))
    gen_avg_fitness = []
    gen_best_fitness = []
    initial_fitness = [calculate_fitness(ind) for ind in population]
    gen_avg_fitness.append(np.mean(initial_fitness))
    initial_fronts = fast_non_dominated_sort(population)
    initial_front_fitness = [calculate_fitness(population[i]) for i in initial_fronts[0]]
    global_best_fitness = min(initial_front_fitness)
    gen_best_fitness.append(global_best_fitness)
    gen_avg_fitness.append(np.mean(initial_front_fitness))
    for gen in range(generations):
        if (gen + 1) % (generations // 10) == 0 or (gen + 1) == generations:
            progress = ((gen + 1) / generations) * 100
            print(f"G-INSGA2 iteration progress: {progress:.0f}%")
        fronts = fast_non_dominated_sort(population)
        for front in fronts:
            if not front: break
            crowding_distance_assignment(front, population)
        population = dynamic_guide_population_enhanced(population, gen, generations, start, target, radars, terrain,
                                                       x_grid, y_grid)
        fronts = fast_non_dominated_sort(population)
        for front in fronts:
            if not front: break
            crowding_distance_assignment(front, population)
        mating_pool = tournament_selection(population)
        offspring = []
        for i in range(0, pop_size, 2):
            parent1 = mating_pool[i]
            parent2 = mating_pool[i + 1] if (i + 1) < pop_size else mating_pool[i]
            child1, child2 = original_crossover(parent1, parent2)
            child1 = original_mutate(child1)
            child2 = original_mutate(child2)
            offspring.append(evaluate_individual(child1, start, target, radars, terrain, x_grid, y_grid))
            offspring.append(evaluate_individual(child2, start, target, radars, terrain, x_grid, y_grid))
        combined = population + offspring
        combined_fronts = fast_non_dominated_sort(combined)
        for front in combined_fronts:
            if not front: break
            crowding_distance_assignment(front, combined)
        new_pop = []
        for front in combined_fronts:
            if not front: break
            if len(new_pop) + len(front) <= pop_size:
                new_pop.extend([combined[i] for i in front])
            else:
                front_sorted = sorted(front, key=lambda x: combined[x].crowding_distance, reverse=True)
                new_pop.extend([combined[i] for i in front_sorted[:pop_size - len(new_pop)]])
                break
        population = new_pop
        current_fronts = fast_non_dominated_sort(population)
        current_front_fitness = [calculate_fitness(population[i]) for i in current_fronts[0]]
        current_min = min(current_front_fitness)
        if current_min < global_best_fitness:
            global_best_fitness = current_min
        gen_best_fitness.append(global_best_fitness)
        gen_avg_fitness.append(np.mean(current_front_fitness))
    final_fronts = fast_non_dominated_sort(population)
    best_front = [population[i] for i in final_fronts[0]]
    best_individual = min(best_front, key=lambda ind: calculate_fitness(ind))
    final_best_fitness = calculate_fitness(best_individual)
    return decode_chromosome(best_individual.chromosome, start,
                             target), gen_avg_fitness, gen_best_fitness, population, final_best_fitness, final_fronts

def a_insga2(start, target, radars, terrain, x_grid, y_grid, pop_size=50, generations=30, num_control_points=3):
    population = []
    x_range = (min(start[0], target[0]) - 500, max(start[0], target[0]) + 500)
    y_range = (min(start[1], target[1]) - 500, max(start[1], target[1]) + 500)
    z_range = (max(min(start[2], target[2]) - 10, 0), 120)
    for _ in range(pop_size):
        individual = original_generate_individual(num_control_points, x_range, y_range, z_range)
        population.append(evaluate_individual(individual, start, target, radars, terrain, x_grid, y_grid))
    gen_avg_fitness = []
    gen_best_fitness = []
    initial_fitness = [calculate_fitness(ind) for ind in population]
    gen_avg_fitness.append(np.mean(initial_fitness))
    initial_fronts = fast_non_dominated_sort(population)
    initial_front_fitness = [calculate_fitness(population[i]) for i in initial_fronts[0]]
    global_best_fitness = min(initial_front_fitness)
    gen_best_fitness.append(global_best_fitness)
    gen_avg_fitness.append(np.mean(initial_front_fitness))
    for gen in range(generations):
        if (gen + 1) % (generations // 10) == 0 or (gen + 1) == generations:
            progress = ((gen + 1) / generations) * 100
            print(f"A-INSGA2 iteration progress: {progress:.0f}%")
        fronts = fast_non_dominated_sort(population)
        for front in fronts:
            if not front: break
            crowding_distance_assignment(front, population)
        mating_pool = tournament_selection(population)
        offspring = []
        for i in range(0, pop_size, 2):
            parent1 = mating_pool[i]
            parent2 = mating_pool[i + 1] if (i + 1) < pop_size else mating_pool[i]
            child1, child2 = adaptive_crossover(parent1, parent2, gen, generations, population)
            child1 = adaptive_mutate(child1, gen, generations, population)
            child2 = adaptive_mutate(child2, gen, generations, population)
            offspring.append(evaluate_individual(child1, start, target, radars, terrain, x_grid, y_grid))
            offspring.append(evaluate_individual(child2, start, target, radars, terrain, x_grid, y_grid))
        combined = population + offspring
        combined_fronts = fast_non_dominated_sort(combined)
        for front in combined_fronts:
            if not front: break
            crowding_distance_assignment(front, combined)
        new_pop = []
        for front in combined_fronts:
            if not front: break
            if len(new_pop) + len(front) <= pop_size:
                new_pop.extend([combined[i] for i in front])
            else:
                front_sorted = sorted(front, key=lambda x: combined[x].crowding_distance, reverse=True)
                new_pop.extend([combined[i] for i in front_sorted[:pop_size - len(new_pop)]])
                break
        population = new_pop
        current_fronts = fast_non_dominated_sort(population)
        current_front_fitness = [calculate_fitness(population[i]) for i in current_fronts[0]]
        current_min = min(current_front_fitness)
        if current_min < global_best_fitness:
            global_best_fitness = current_min
        gen_best_fitness.append(global_best_fitness)
        gen_avg_fitness.append(np.mean(current_front_fitness))
    final_fronts = fast_non_dominated_sort(population)
    best_front = [population[i] for i in final_fronts[0]]
    best_individual = min(best_front, key=lambda ind: calculate_fitness(ind))
    final_best_fitness = calculate_fitness(best_individual)
    return decode_chromosome(best_individual.chromosome, start,
                             target), gen_avg_fitness, gen_best_fitness, population, final_best_fitness, final_fronts

def i_insga2(start, target, radars, terrain, x_grid, y_grid, pop_size=50, generations=30, num_control_points=3):
    population = []
    x_range = (min(start[0], target[0]) - 500, max(start[0], target[0]) + 500)
    y_range = (min(start[1], target[1]) - 500, max(start[1], target[1]) + 500)
    z_range = (max(min(start[2], target[2]) - 10, 0), 120)
    for _ in range(pop_size):
        individual = modified_generate_individual(num_control_points, start, target, radars, x_range, y_range, z_range)
        population.append(evaluate_individual(individual, start, target, radars, terrain, x_grid, y_grid))
    gen_avg_fitness = []
    gen_best_fitness = []
    initial_fitness = [calculate_fitness(ind) for ind in population]
    gen_avg_fitness.append(np.mean(initial_fitness))
    initial_fronts = fast_non_dominated_sort(population)
    initial_front_fitness = [calculate_fitness(population[i]) for i in initial_fronts[0]]
    global_best_fitness = min(initial_front_fitness)
    gen_best_fitness.append(global_best_fitness)
    gen_avg_fitness.append(np.mean(initial_front_fitness))
    for gen in range(generations):
        if (gen + 1) % (generations // 10) == 0 or (gen + 1) == generations:
            progress = ((gen + 1) / generations) * 100
            print(f"I-INSGA2 iteration progress: {progress:.0f}%")
        fronts = fast_non_dominated_sort(population)
        for front in fronts:
            if not front: break
            crowding_distance_assignment(front, population)
        mating_pool = tournament_selection(population)
        offspring = []
        for i in range(0, pop_size, 2):
            parent1 = mating_pool[i]
            parent2 = mating_pool[i + 1] if (i + 1) < pop_size else mating_pool[i]
            child1, child2 = original_crossover(parent1, parent2)
            child1 = original_mutate(child1)
            child2 = original_mutate(child2)
            offspring.append(evaluate_individual(child1, start, target, radars, terrain, x_grid, y_grid))
            offspring.append(evaluate_individual(child2, start, target, radars, terrain, x_grid, y_grid))
        combined = population + offspring
        combined_fronts = fast_non_dominated_sort(combined)
        for front in combined_fronts:
            if not front: break
            crowding_distance_assignment(front, combined)
        new_pop = []
        for front in combined_fronts:
            if not front: break
            if len(new_pop) + len(front) <= pop_size:
                new_pop.extend([combined[i] for i in front])
            else:
                front_sorted = sorted(front, key=lambda x: combined[x].crowding_distance, reverse=True)
                new_pop.extend([combined[i] for i in front_sorted[:pop_size - len(new_pop)]])
                break
        population = new_pop
        current_fronts = fast_non_dominated_sort(population)
        current_front_fitness = [calculate_fitness(population[i]) for i in current_fronts[0]]
        current_min = min(current_front_fitness)
        if current_min < global_best_fitness:
            global_best_fitness = current_min
        gen_best_fitness.append(global_best_fitness)
        gen_avg_fitness.append(np.mean(current_front_fitness))
    final_fronts = fast_non_dominated_sort(population)
    best_front = [population[i] for i in final_fronts[0]]
    best_individual = min(best_front, key=lambda ind: calculate_fitness(ind))
    final_best_fitness = calculate_fitness(best_individual)
    return decode_chromosome(best_individual.chromosome, start,
                             target), gen_avg_fitness, gen_best_fitness, population, final_best_fitness, final_fronts

def l_insga2(start, target, radars, terrain, x_grid, y_grid, pop_size=50, generations=30, num_control_points=3):
    population = []
    x_range = (min(start[0], target[0]) - 500, max(start[0], target[0]) + 500)
    y_range = (min(start[1], target[1]) - 500, max(start[1], target[1]) + 500)
    z_range = (max(min(start[2], target[2]) - 10, 0), 120)
    for _ in range(pop_size):
        individual = original_generate_individual(num_control_points, x_range, y_range, z_range)
        population.append(evaluate_individual(individual, start, target, radars, terrain, x_grid, y_grid))
    gen_avg_fitness = []
    gen_best_fitness = []
    initial_fitness = [calculate_fitness(ind) for ind in population]
    gen_avg_fitness.append(np.mean(initial_fitness))
    initial_fronts = fast_non_dominated_sort(population)
    initial_front_fitness = [calculate_fitness(population[i]) for i in initial_fronts[0]]
    global_best_fitness = min(initial_front_fitness)
    gen_best_fitness.append(global_best_fitness)
    gen_avg_fitness.append(np.mean(initial_front_fitness))
    for gen in range(generations):
        if (gen + 1) % (generations // 10) == 0 or (gen + 1) == generations:
            progress = ((gen + 1) / generations) * 100
            print(f"L-INSGA2 iteration progress: {progress:.0f}%")
        fronts = fast_non_dominated_sort(population)
        for front in fronts:
            if not front: break
            crowding_distance_assignment(front, population)
        mating_pool = tournament_selection(population)
        offspring = []
        for i in range(0, pop_size, 2):
            parent1 = mating_pool[i]
            parent2 = mating_pool[i + 1] if (i + 1) < pop_size else mating_pool[i]
            child1, child2 = original_crossover(parent1, parent2)
            child1 = original_mutate(child1)
            child2 = original_mutate(child2)
            offspring.append(evaluate_individual(child1, start, target, radars, terrain, x_grid, y_grid))
            offspring.append(evaluate_individual(child2, start, target, radars, terrain, x_grid, y_grid))
        combined = population + offspring
        combined_fronts = fast_non_dominated_sort(combined)
        for front in combined_fronts:
            if not front: break
            crowding_distance_assignment(front, combined)
        new_pop = []
        for front in combined_fronts:
            if not front: break
            if len(new_pop) + len(front) <= pop_size:
                new_pop.extend([combined[i] for i in front])
            else:
                front_sorted = sorted(front, key=lambda x: combined[x].crowding_distance, reverse=True)
                new_pop.extend([combined[i] for i in front_sorted[:pop_size - len(new_pop)]])
                break
        new_pop = enhance_pareto_solutions(new_pop, start, target, radars, terrain, x_grid, y_grid)
        population = new_pop
        current_fronts = fast_non_dominated_sort(population)
        current_front_fitness = [calculate_fitness(population[i]) for i in current_fronts[0]]
        current_min = min(current_front_fitness)
        if current_min < global_best_fitness:
            global_best_fitness = current_min
        gen_best_fitness.append(global_best_fitness)
        gen_avg_fitness.append(np.mean(current_front_fitness))
    final_fronts = fast_non_dominated_sort(population)
    best_front = [population[i] for i in final_fronts[0]]
    best_individual = min(best_front, key=lambda ind: calculate_fitness(ind))
    final_best_fitness = calculate_fitness(best_individual)
    return decode_chromosome(best_individual.chromosome, start,
                             target), gen_avg_fitness, gen_best_fitness, population, final_best_fitness, final_fronts

def ms_insga2(start, target, radars, terrain, x_grid, y_grid, pop_size=50, generations=30, num_control_points=3):
    population = []
    x_range = (min(start[0], target[0]) - 500, max(start[0], target[0]) + 500)
    y_range = (min(start[1], target[1]) - 500, max(start[1], target[1]) + 500)
    z_range = (max(min(start[2], target[2]) - 10, 0), 120)
    for _ in range(pop_size):
        individual = modified_generate_individual(num_control_points, start, target, radars, x_range, y_range, z_range)
        population.append(evaluate_individual(individual, start, target, radars, terrain, x_grid, y_grid))
    gen_avg_fitness = []
    gen_best_fitness = []
    initial_fitness = [calculate_fitness(ind) for ind in population]
    gen_avg_fitness.append(np.mean(initial_fitness))
    initial_fronts = fast_non_dominated_sort(population)
    initial_front_fitness = [calculate_fitness(population[i]) for i in initial_fronts[0]]
    global_best_fitness = min(initial_front_fitness)
    gen_best_fitness.append(global_best_fitness)
    gen_avg_fitness.append(np.mean(initial_front_fitness))
    for gen in range(generations):
        if (gen + 1) % (generations // 10) == 0 or (gen + 1) == generations:
            progress = ((gen + 1) / generations) * 100
            print(f"MS-INSGA2 iteration progress: {progress:.0f}%")
        fronts = fast_non_dominated_sort(population)
        for front in fronts:
            if not front: break
            crowding_distance_assignment(front, population)
        population = dynamic_guide_population_enhanced(population, gen, generations, start, target, radars, terrain,
                                                       x_grid, y_grid)
        fronts = fast_non_dominated_sort(population)
        for front in fronts:
            if not front: break
            crowding_distance_assignment(front, population)
        mating_pool = tournament_selection(population)
        offspring = []
        for i in range(0, pop_size, 2):
            parent1 = mating_pool[i]
            parent2 = mating_pool[i + 1] if (i + 1) < pop_size else mating_pool[i]
            child1, child2 = adaptive_crossover(parent1, parent2, gen, generations, population)
            child1 = adaptive_mutate(child1, gen, generations, population)
            child2 = adaptive_mutate(child2, gen, generations, population)
            offspring.append(evaluate_individual(child1, start, target, radars, terrain, x_grid, y_grid))
            offspring.append(evaluate_individual(child2, start, target, radars, terrain, x_grid, y_grid))
        combined = population + offspring
        combined_fronts = fast_non_dominated_sort(combined)
        for front in combined_fronts:
            if not front: break
            crowding_distance_assignment(front, combined)
        new_pop = []
        for front in combined_fronts:
            if not front: break
            if len(new_pop) + len(front) <= pop_size:
                new_pop.extend([combined[i] for i in front])
            else:
                front_sorted = sorted(front, key=lambda x: combined[x].crowding_distance, reverse=True)
                new_pop.extend([combined[i] for i in front_sorted[:pop_size - len(new_pop)]])
                break
        new_pop = enhance_pareto_solutions(new_pop, start, target, radars, terrain, x_grid, y_grid)
        population = new_pop
        current_fronts = fast_non_dominated_sort(population)
        current_front_fitness = [calculate_fitness(population[i]) for i in current_fronts[0]]
        current_min = min(current_front_fitness)
        if current_min < global_best_fitness:
            global_best_fitness = current_min
        gen_best_fitness.append(global_best_fitness)
        gen_avg_fitness.append(np.mean(current_front_fitness))
    final_fronts = fast_non_dominated_sort(population)
    best_front = [population[i] for i in final_fronts[0]]
    best_individual = min(best_front, key=lambda ind: calculate_fitness(ind))
    final_best_fitness = calculate_fitness(best_individual)
    return decode_chromosome(best_individual.chromosome, start,
                             target), gen_avg_fitness, gen_best_fitness, population, final_best_fitness, final_fronts

def prim_mst_total_weight(dissimilarity_matrix):
    import heapq
    n = len(dissimilarity_matrix)
    if n <= 1:
        return 0.0
    selected = [False] * n
    selected[0] = True
    total_weight = 0.0
    heap = [(dissimilarity_matrix[0][j], j) for j in range(n) if j != 0]
    heapq.heapify(heap)
    while heap:
        dist, v = heapq.heappop(heap)
        if selected[v]:
            continue
        selected[v] = True
        total_weight += dist
        for u in range(n):
            if not selected[u] and dissimilarity_matrix[v][u] > 0:
                heapq.heappush(heap, (dissimilarity_matrix[v][u], u))
    return total_weight

def calculate_gd(front_individuals, p_star_objectives):
    if not front_individuals or len(front_individuals) == 0 or len(p_star_objectives) == 0:
        return 0.0
    p_objectives = np.array([ind.objectives for ind in front_individuals])
    p_star = p_star_objectives.copy()
    obj_min = np.min(p_star, axis=0)
    obj_max = np.max(p_star, axis=0)
    obj_range = obj_max - obj_min
    obj_range[obj_range == 0] = 1.0
    p_normalized = (p_objectives - obj_min) / obj_range
    p_star_normalized = (p_star - obj_min) / obj_range
    sum_sq_min = 0.0
    for p in p_normalized:
        dist_sq = np.sum((p_star_normalized - p) ** 2, axis=1)
        min_dist_sq = np.min(dist_sq)
        sum_sq_min += min_dist_sq
    gd = np.sqrt(sum_sq_min / len(p_normalized))
    return gd

def calculate_hv(front_individuals, ref_point=None, ref_buffer=1.1, sample_count=10000):
    if not front_individuals:
        return 0.0
    objectives = np.array([ind.objectives for ind in front_individuals])
    obj_min = np.min(objectives, axis=0)
    obj_max = np.max(objectives, axis=0)
    if ref_point is None:
        ref_point = obj_max * ref_buffer
    else:
        ref_point = np.maximum(ref_point, obj_max)
    hypercube_edge = ref_point - obj_min
    hypercube_volume = np.prod(hypercube_edge)
    if hypercube_volume <= 1e-9:
        return 0.0
    rand_points = np.random.uniform(obj_min, ref_point, (sample_count, len(obj_min)))
    objectives_expand = objectives[:, np.newaxis, :]
    rand_points_expand = rand_points[np.newaxis, :, :]
    is_dominated_per_point = np.any(
        np.logical_and(
            np.all(objectives_expand <= rand_points_expand, axis=2),
            np.any(objectives_expand < rand_points_expand, axis=2)
        ), axis=0
    )
    dominated_count = np.sum(is_dominated_per_point)
    hv_value = hypercube_volume * (dominated_count / sample_count)
    return hv_value

def plot_3d_comparison(X, Y, Z, paths, start, target, radars, save_path):
    fig = plt.figure(figsize=(16, 12))
    ax = fig.add_subplot(111, projection="3d")
    ax.set_box_aspect([1, 1, 0.2])
    surf = ax.plot_surface(X, Y, Z, cmap=cm.terrain, alpha=0.7, shade=True)
    for radar in radars:
        cx, cy = radar["pos"]
        radius, height = radar["radius"], radar["height"]
        terrain_z = Z[np.argmin(np.abs(y - cy)), np.argmin(np.abs(x - cx))]
        bottom_z = terrain_z + 1
        top_z = bottom_z + height
        theta = np.linspace(0, 2 * np.pi, 50)
        r_circle = np.linspace(0, radius, 10)
        Theta, R = np.meshgrid(theta, r_circle)
        X_circle = cx + R * np.cos(Theta)
        Y_circle = cy + R * np.sin(Theta)
        Z_circle = np.ones_like(X_circle) * bottom_z
        ax.plot_surface(X_circle, Y_circle, Z_circle, color="#FF0000", alpha=0.8, edgecolor='none')
        Z_side = np.linspace(bottom_z, top_z, 10)
        Theta_side, Z_side_grid = np.meshgrid(theta, Z_side)
        X_side = cx + radius * np.cos(Theta_side)
        Y_side = cy + radius * np.sin(Theta_side)
        ax.plot_surface(X_side, Y_side, Z_side_grid, color="#FF0000", alpha=0.8, edgecolor='none')
        Z_top = np.ones_like(X_circle) * top_z
        ax.plot_surface(X_circle, Y_circle, Z_top, color="#FF0000", alpha=0.8, edgecolor='none')

    colors = ['blue', 'cyan', 'green', 'purple', 'orange', 'red', 'magenta', 'brown', 'pink', 'gray']
    markers = ['o', 's', '^', 'D', 'v', '<', '>', 'p', '*', 'h']
    line_styles = ['-', '--', '-.', ':', '-', '--', '-.', ':', '-', '--']
    for i, (name, path) in enumerate(paths.items()):
        ax.plot(path[:, 0], path[:, 1], path[:, 2],
                color=colors[i % len(colors)],
                linewidth=2.5,
                label=name,
                marker=markers[i % len(markers)],
                markersize=6,
                markevery=5,
                linestyle=line_styles[i % len(line_styles)])
    ax.scatter(start[0], start[1], start[2], color="green", s=120, marker="o", label="Start Point", edgecolors='black',
               linewidth=2)
    ax.scatter(target[0], target[1], target[2], color="orange", s=120, marker="o", label="Target Point",
               edgecolors='black', linewidth=2)
    ax.set_xlabel("X /m", fontsize=18, labelpad=33)
    ax.set_ylabel("Y /m", fontsize=18, labelpad=15)
    ax.set_zlabel("Z /m", fontsize=18, labelpad=10)
    ax.tick_params(axis='x', labelsize=16, pad=15)
    ax.tick_params(axis='y', labelsize=16, pad=5)
    ax.tick_params(axis='z', labelsize=16, pad=5)
    ax.set_xlim(0, 3500)
    ax.set_ylim(0, 3500)
    ax.set_zlim(0, 120)
    ax.legend(fontsize=18, ncol=4, loc='upper center', bbox_to_anchor=(0.55, 0.14), columnspacing=3.5)
    ax.view_init(elev=25, azim=130)
    plt.savefig(os.path.join(save_path, "3d_path_comparison_current.png"), dpi=300, bbox_inches="tight")
    ax.view_init(elev=90, azim=180)
    if ax.elev == 90:
        ax.set_zticklabels([])
        for tick in ax.zaxis.get_major_ticks():
            tick.tick1line.set_visible(False)
            tick.tick2line.set_visible(False)
        ax.zaxis.line.set_visible(False)
    ax.legend(fontsize=16, ncol=4, loc='upper center', bbox_to_anchor=(0.5, 0.05), columnspacing=1.0)
    plt.savefig(os.path.join(save_path, "3d_path_comparison_top.png"), dpi=300, bbox_inches="tight")
    plt.close(fig)

def plot_average_fitness_convergence(avg_data, save_path):
    fig, ax = plt.subplots(figsize=(14, 7))
    colors = ['blue', 'cyan', 'green', 'purple', 'orange', 'red', 'magenta', 'brown', 'pink', 'gray']
    markers = ['o', 's', '^', 'D', 'v', '<', '>', 'p', '*', 'h']
    max_gen = max(len(avg) for avg in avg_data.values()) if avg_data else 100
    zoom_start = int(max_gen * 0.8)
    zoom_end = max_gen - 1
    axins = ax.inset_axes([0.55, 0.55, 0.4, 0.35])
    y_min_zoom = float('inf')
    y_max_zoom = float('-inf')
    for i, (name, avg) in enumerate(avg_data.items()):
        gen_range = range(len(avg))
        shadow = np.std(avg) * 0.5 if len(avg) > 1 else 0
        ax.plot(gen_range, avg, color=colors[i % len(colors)], linewidth=3,
                marker=markers[i % len(markers)], markersize=10, markevery=8, label=name)
        ax.fill_between(gen_range, np.array(avg) + shadow, np.array(avg) - shadow,
                        color=colors[i % len(colors)], alpha=0.2)
        axins.plot(gen_range, avg, color=colors[i % len(colors)], linewidth=2,
                   marker=markers[i % len(markers)], markersize=6, markevery=2)
        if len(avg) > zoom_start:
            zoomed_data = avg[zoom_start:zoom_end + 1]
            if len(zoomed_data) > 0:
                y_min_zoom = min(y_min_zoom, np.min(zoomed_data))
                y_max_zoom = max(y_max_zoom, np.max(zoomed_data))
    axins.set_xlim(zoom_start, zoom_end)
    if y_max_zoom > y_min_zoom:
        y_margin = (y_max_zoom - y_min_zoom) * 0.2
        axins.set_ylim(y_min_zoom - y_margin, y_max_zoom + y_margin)
    axins.grid(True, alpha=0.3)
    axins.tick_params(axis='both', labelsize=12)
    ax.indicate_inset_zoom(axins, edgecolor="black")
    ax.set_xlabel("Generation", fontsize=20)
    ax.set_ylabel("Average Fitness", fontsize=20)
    ax.tick_params(axis='x', labelsize=18)
    ax.tick_params(axis='y', labelsize=18)
    ax.legend(fontsize=18, ncol=6, loc='upper center', bbox_to_anchor=(0.5, -0.12), columnspacing=0.5)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(save_path, "average_fitness_convergence.png"), dpi=300, bbox_inches="tight")
    plt.close()

def plot_best_fitness_convergence(best_data, save_path):
    fig, ax = plt.subplots(figsize=(14, 7))
    colors = ['blue', 'cyan', 'green', 'purple', 'orange', 'red', 'magenta', 'brown', 'pink', 'gray']
    markers = ['o', 's', '^', 'D', 'v', '<', '>', 'p', '*', 'h']
    max_gen = max(len(best) for best in best_data.values()) if best_data else 100
    zoom_start = int(max_gen * 0.8)
    zoom_end = max_gen - 1
    axins = ax.inset_axes([0.55, 0.55, 0.4, 0.35])
    y_min_zoom = float('inf')
    y_max_zoom = float('-inf')
    for i, (name, best) in enumerate(best_data.items()):
        gen_range = range(len(best))
        shadow = np.std(best) * 0.5 if len(best) > 1 else 0
        ax.plot(gen_range, best, color=colors[i % len(colors)], linewidth=3,
                marker=markers[i % len(markers)], markersize=10, markevery=8, label=name)
        ax.fill_between(gen_range, np.array(best) + shadow, np.array(best) - shadow,
                        color=colors[i % len(colors)], alpha=0.2)
        axins.plot(gen_range, best, color=colors[i % len(colors)], linewidth=2,
                   marker=markers[i % len(markers)], markersize=6, markevery=2)
        if len(best) > zoom_start:
            zoomed_data = best[zoom_start:zoom_end + 1]
            if len(zoomed_data) > 0:
                y_min_zoom = min(y_min_zoom, np.min(zoomed_data))
                y_max_zoom = max(y_max_zoom, np.max(zoomed_data))
    axins.set_xlim(zoom_start, zoom_end)
    if y_max_zoom > y_min_zoom:
        y_margin = (y_max_zoom - y_min_zoom) * 0.2
        axins.set_ylim(y_min_zoom - y_margin, y_max_zoom + y_margin)
    axins.grid(True, alpha=0.3)
    axins.tick_params(axis='both', labelsize=12)
    ax.indicate_inset_zoom(axins, edgecolor="black")
    ax.set_xlabel("Generation", fontsize=20)
    ax.set_ylabel("Best Fitness", fontsize=20)
    ax.tick_params(axis='x', labelsize=18)
    ax.tick_params(axis='y', labelsize=18)
    ax.legend(fontsize=18, ncol=6, loc='upper center', bbox_to_anchor=(0.5, -0.12), columnspacing=0.5)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(save_path, "best_fitness_convergence.png"), dpi=300, bbox_inches="tight")
    plt.close()

def plot_fitness_distribution_comparison(fitness_data, save_path):
    fig, ax = plt.subplots(figsize=(12, 8))
    data = [fitness for name, fitness in fitness_data.items()]
    labels = [name for name, fitness in fitness_data.items()]
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#8c564b', '#9467bd',
              '#d62728', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']
    positions = np.arange(1, len(labels) + 1)
    box_width = 0.35
    box_plot = ax.boxplot(data, patch_artist=True, widths=box_width, showfliers=False,
                          positions=positions - box_width / 2)
    for patch, color in zip(box_plot['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
    for i, (fitness, pos, color) in enumerate(zip(data, positions, colors)):
        x_scatter = np.random.normal(pos + box_width / 2, 0.08, len(fitness))
        ax.scatter(x_scatter, fitness, color=color, alpha=0.6, s=20, edgecolors='black', linewidth=0.5)
    for i, (fitness, pos, color) in enumerate(zip(data, positions, colors)):
        kde = stats.gaussian_kde(fitness, bw_method='scott')
        y_vals = np.linspace(min(fitness), max(fitness), 200)
        density = kde(y_vals)
        density_norm = density / np.max(density) if np.max(density) > 0 else 0
        scale = 0.15
        x_density = pos + box_width / 2 + density_norm * scale
        ax.plot(x_density, y_vals, color=color, linewidth=2, alpha=0.8)
    ax.set_xlabel("Algorithm", fontsize=20)
    ax.set_ylabel("Fitness Value", fontsize=20)
    ax.set_xticks(positions)
    ax.set_xticklabels(labels, fontsize=18, rotation=0)
    ax.tick_params(axis='y', labelsize=18)
    plt.tight_layout()
    plt.savefig(os.path.join(save_path, "fitness_distribution_comparison.png"), dpi=300, bbox_inches="tight")
    plt.close()

def plot_metrics_comparison(metrics, save_path):
    fig, axes = plt.subplots(2, 4, figsize=(20, 10))
    axes = axes.flatten()
    metrics_keys = list(metrics.keys())[1:]
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#8c564b', '#9467bd',
              '#d62728', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']
    for i, metric in enumerate(metrics_keys):
        ax = axes[i]
        algorithms = metrics['Algorithm']
        values = metrics[metric]
        bars = ax.bar(algorithms, values, color=colors[:len(algorithms)], alpha=0.7)
        ax.set_title(metric, fontsize=14)
        ax.tick_params(axis='x', labelsize=10, rotation=45)
        ax.tick_params(axis='y', labelsize=10)
        ax.grid(True, alpha=0.3, axis='y')
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width() / 2., height,
                    f'{height:.2f}',
                    ha='center', va='bottom', fontsize=9)
    plt.tight_layout()
    plt.savefig(os.path.join(save_path, "metrics_comparison.png"), dpi=300, bbox_inches="tight")
    plt.close()

if __name__ == "__main__":
    x, y, X, Y, Z = generate_terrain()
    start, target = generate_start_target(x, y, Z)
    save_dir = "./multi_algorithm_comparison"
    os.makedirs(save_dir, exist_ok=True)
    pop_size = 50
    generations = 100
    num_control_points = 5

    algorithms = [
        ("NSGA2", nsga2),
        ("G-INSGA2", g_insga2),
        ("A-INSGA2", a_insga2),
        ("I-INSGA2", i_insga2),
        ("L-INSGA2", l_insga2),
        ("MS-INSGA2", ms_insga2),
    ]

    results = {}
    for name, algo in algorithms:
        print("\n" + "=" * 50)
        print(f"Starting to run {name} algorithm...")
        path, avg, best, pop, final_fitness, final_fronts = algo(
            start, target, radars, Z, x, y, pop_size=pop_size, generations=generations,
            num_control_points=num_control_points
        )
        results[name] = {
            "path": path,
            "avg_fitness": avg,
            "best_fitness": best,
            "population": pop,
            "final_best_fitness": final_fitness,
            "final_fronts": final_fronts
        }

    fitness_data = {name: [calculate_fitness(ind) for ind in res["population"]] for name, res in results.items()}

    p_star_individuals = []
    for name, res in results.items():
        front = [res["population"][i] for i in res["final_fronts"][0]]
        p_star_individuals.extend(front)
    p_star_objectives = np.array([ind.objectives for ind in p_star_individuals])
    if len(p_star_objectives) == 0:
        p_star_objectives = np.array([[0.0, 0.0, 0.0]])

    all_objectives = p_star_objectives.copy() if len(p_star_objectives) > 0 else np.array([])
    global_ref_point = np.max(all_objectives, axis=0) * 1.1 if len(all_objectives) > 0 else np.array([1000, 1000, 1000])

    metrics = {
        'Algorithm': [],
        'Final Best Fitness': [],
        'Average Fitness (Final Gen)': [],
        'Fitness Std': [],
        'Path Length (m)': [],
        'Collision Risk': [],
        'GD Value': [],
        'Hypervolume (HV)': []
    }

    for name, res in results.items():
        front = [res["population"][i] for i in res["final_fronts"][0]]

        p_star_individuals = []
        for other_name, other_res in results.items():
            if other_name != name:
                other_front = [other_res["population"][i] for i in other_res["final_fronts"][0]]
                p_star_individuals.extend(other_front)

        if not p_star_individuals:
            p_star_objectives = np.array([ind.objectives for ind in front])
        else:
            p_star_objectives = np.array([ind.objectives for ind in p_star_individuals])

        gd = calculate_gd(front, p_star_objectives)
        hv = calculate_hv(front, ref_point=global_ref_point)

        path_length = calculate_distance(res["path"])
        collision_risk = calculate_collision_risk(res["path"], radars, Z, x, y)
        final_avg = res["avg_fitness"][-1]
        fitness_std = np.std(fitness_data[name])

        metrics['Algorithm'].append(name)
        metrics['Final Best Fitness'].append(res["final_best_fitness"])
        metrics['Average Fitness (Final Gen)'].append(final_avg)
        metrics['Fitness Std'].append(fitness_std)
        metrics['Path Length (m)'].append(path_length)
        metrics['Collision Risk'].append(collision_risk)
        metrics['GD Value'].append(gd)
        metrics['Hypervolume (HV)'].append(hv)

    print("\n" + "=" * 50)
    print("Starting to plot comparison charts...")
    paths = {name: res["path"] for name, res in results.items()}
    plot_3d_comparison(X, Y, Z, paths, start, target, radars, save_dir)

    avg_data = {name: res["avg_fitness"] for name, res in results.items()}
    best_data = {name: res["best_fitness"] for name, res in results.items()}
    plot_average_fitness_convergence(avg_data, save_dir)
    plot_best_fitness_convergence(best_data, save_dir)

    plot_fitness_distribution_comparison(fitness_data, save_dir)
    plot_metrics_comparison(metrics, save_dir)

    import pandas as pd

    os.makedirs('excel_multi', exist_ok=True)
    df = pd.DataFrame(metrics)
    df.to_excel('excel_multi/algorithm_metrics.xlsx', index=False)

    print("\n" + "=" * 50)
    print("Key Metrics Comparison Across Multiple Algorithms:")
    print(f"| Algorithm Name | Final Best Fitness | Average Fitness | Path Length (m) | Collision Risk | GD Value | HV Value |")
    print(f"|----------------|--------------------|-----------------|-----------------|----------------|----------|----------|")
    for i in range(len(metrics['Algorithm'])):
        print(
            f"| {metrics['Algorithm'][i]:<14} | {metrics['Final Best Fitness'][i]:<18.4f} | {metrics['Average Fitness (Final Gen)'][i]:<15.4f} | {metrics['Path Length (m)'][i]:<15.4f} | {metrics['Collision Risk'][i]:<14.4f} | {metrics['GD Value'][i]:<8.4f} | {metrics['Hypervolume (HV)'][i]:<8.4f} |")
    print("\nAll comparison charts have been saved to:", os.path.abspath(save_dir))
    print("Metrics data has been saved to:", os.path.abspath("excel_multi/algorithm_metrics.xlsx"))