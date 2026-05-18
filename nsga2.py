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

hill_count = 4
hill_amplitude = [110, 120, 120, 100]
hill_width = [400, 200, 300, 200]
hill_center_x = [500, 1500, 2800, 2000]
hill_center_y = [500, 800, 2200, 2800]
noise_intensity = 15
base_height = 20

radar_count = 3
radars = [
    {"pos": (1800, 1500), "radius": 200, "rings": 3, "height": 120},
    {"pos": (2800, 1200), "radius": 200, "rings": 3, "height": 120},
    {"pos": (800, 2200), "radius": 200, "rings": 3, "height": 120}
]

ADAPTIVE_PC_INIT = 0.75
ADAPTIVE_PC_FINAL = 0.65
ADAPTIVE_PM_BASE = 0.25
ADAPTIVE_PM_MAX = 0.3
LOCAL_SEARCH_RATIO = 0.2
DIVERSITY_THRESHOLD = 0.15

x = np.linspace(0, 3500, 100)
y = np.linspace(0, 3500, 100)
X, Y = np.meshgrid(x, y)

Z = base_height + noise_intensity * np.random.rand(*X.shape)

for i in range(hill_count):
    Z += hill_amplitude[i] * np.exp(-(((X - hill_center_x[i]) ** 2) / (2 * hill_width[i] ** 2) +
                                      ((Y - hill_center_y[i]) ** 2) / (2 * hill_width[i] ** 2)))

Z = np.clip(Z, 0, 120)

start_xy = (200, 3200)
start_idx_x = np.argmin(np.abs(x - start_xy[0]))
start_idx_y = np.argmin(np.abs(y - start_xy[1]))
start_z = Z[start_idx_y, start_idx_x] + 50
start = (start_xy[0], start_xy[1], start_z)

target_xy = (3200, 500)
target_idx_x = np.argmin(np.abs(x - target_xy[0]))
target_idx_y = np.argmin(np.abs(y - target_xy[1]))
target_z = Z[target_idx_y, target_idx_x] + 50
target = (target_xy[0], target_xy[1], target_z)

class Individual:
    def __init__(self, chromosome):
        self.chromosome = chromosome
        self.objectives = []
        self.rank = None
        self.crowding_distance = 0

def generate_individual(num_control_points, x_range, y_range, z_range):
    chromosome = []
    for _ in range(num_control_points):
        chromosome.append(random.uniform(x_range[0], x_range[1]))
        chromosome.append(random.uniform(y_range[0], y_range[1]))
        chromosome.append(random.uniform(z_range[0], z_range[1]))
    return Individual(chromosome)

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
    path_z = np.clip(path_z, 0, 120)

    return np.column_stack((path_x, path_y, path_z))

def calculate_distance(path):
    return np.sum(np.sqrt(np.sum(np.diff(path, axis=0) ** 2, axis=1)))

def calculate_collision_risk(path, radars, terrain, x_grid, y_grid, safe_distance=200):
    risk = 0

    for point in path:
        idx_x = np.argmin(np.abs(x_grid - point[0]))
        idx_y = np.argmin(np.abs(y_grid - point[1]))
        terrain_z = terrain[idx_y, idx_x]
        if point[2] < terrain_z + 50:
            risk += 100 * (terrain_z + 50 - point[2]) / 50

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
                risk += 1000
            elif dist_horiz <= radius + safe_distance:
                proximity = (radius + safe_distance - dist_horiz) / safe_distance
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
        distance[sorted_indices[0][0]] = float('inf')
        distance[sorted_indices[-1][0]] = float('inf')
        min_val = population[sorted_indices[0][1]].objectives[m]
        max_val = population[sorted_indices[-1][1]].objectives[m]
        if max_val - min_val < 1e-6:
            continue
        for k in range(1, len(sorted_indices) - 1):
            prev_idx = sorted_indices[k-1][1]
            next_idx = sorted_indices[k+1][1]
            distance[sorted_indices[k][0]] += (population[next_idx].objectives[m] - population[prev_idx].objectives[m]) / (max_val - min_val)

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

def crossover(parent1, parent2, crossover_rate=0.8):
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

def mutate(individual, mutation_rate=0.1, mutation_range=(-50, 50)):
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

def nsga2(start, target, radars, terrain, x_grid, y_grid, pop_size=50, generations=100, num_control_points=3):
    def calculate_fitness(individual):
        return 0.5 * individual.objectives[0] + 0.3 * individual.objectives[1] + 0.2 * individual.objectives[2]

    population = []
    for _ in range(pop_size):
        individual = generate_individual(
            num_control_points,
            (min(start[0], target[0]) - 500, max(start[0], target[0]) + 500),
            (min(start[1], target[1]) - 500, max(start[1], target[1]) + 500),
            (max(min(start[2], target[2]) - 10, 0), 120)
        )
        population.append(evaluate_individual(individual, start, target, radars, terrain, x_grid, y_grid))

    gen_avg_fitness = []
    gen_best_fitness = []
    initial_fitness = [calculate_fitness(ind) for ind in population]
    gen_avg_fitness.append(np.mean(initial_fitness))
    initial_fronts = fast_non_dominated_sort(population)
    initial_front_fitness = [calculate_fitness(population[i]) for i in initial_fronts[0]]
    gen_best_fitness.append(min(initial_front_fitness))

    for gen in range(generations):
        if (gen + 1) % (generations // 10) == 0 or (gen + 1) == generations:
            progress = ((gen + 1) / generations) * 100
            print(f"Iteration progress: {progress:.0f}%")

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
            child1, child2 = crossover(parent1, parent2)
            child1 = mutate(child1)
            child2 = mutate(child2)
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

        current_fitness = [calculate_fitness(ind) for ind in population]
        gen_avg_fitness.append(np.mean(current_fitness))
        current_fronts = fast_non_dominated_sort(population)
        current_front_fitness = [calculate_fitness(population[i]) for i in current_fronts[0]]
        gen_best_fitness.append(min(current_front_fitness))

    final_fronts = fast_non_dominated_sort(population)
    best_front = [population[i] for i in final_fronts[0]]
    best_individual = min(best_front, key=lambda ind: calculate_fitness(ind))
    final_best_fitness = calculate_fitness(best_individual)

    return decode_chromosome(best_individual.chromosome, start, target), gen_avg_fitness, gen_best_fitness, population, final_best_fitness

if __name__ == "__main__":
    save_dir = "./picture"
    os.makedirs(save_dir, exist_ok=True)

    pop_size = 50
    generations = 30
    path, gen_avg_fitness, gen_best_fitness, final_population, final_best_fitness = nsga2(
        start, target, radars, Z, x, y,
        pop_size=pop_size, generations=generations, num_control_points=3
    )

    fig = plt.figure(figsize=(12, 9))
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

    ax.plot(path[:, 0], path[:, 1], path[:, 2], color="blue", linewidth=2, label="NSGA2 Path")
    ax.scatter(start[0], start[1], start[2], color="red", s=100, marker="o", label="Start Point")
    ax.scatter(target[0], target[1], target[2], color="orange", s=100, marker="o", label="Target Point")

    ax.set_xlabel("X /m",fontsize=16, labelpad=25)
    ax.set_ylabel("Y /m",fontsize=16, labelpad=15)
    ax.set_zlabel("Z /m",fontsize=16, labelpad=5)
    ax.tick_params(axis='x',labelsize=14, pad=5)
    ax.tick_params(axis='y',labelsize=14, pad=5)
    ax.tick_params(axis='z', labelsize=14, pad=0)

    ax.zaxis.set_ticks([0, 40, 80, 120])

    ax.set_title("3D UAV Path Planning with NSGA2 Algorithm",fontsize=16)
    ax.set_zlim(0, 140)
    ax.legend(fontsize=14, ncol=8, loc='upper center', bbox_to_anchor=(0.5, 0.14))

    ax.view_init(elev=22, azim=150)
    plt.savefig(os.path.join(save_dir, "nsga2_3d_view_current.png"), dpi=300, bbox_inches="tight")
    ax.view_init(elev=90, azim=180)

    if ax.elev == 90:
        ax.set_zticklabels([])
        for tick in ax.zaxis.get_major_ticks():
            tick.tick1line.set_visible(False)
            tick.tick2line.set_visible(False)
        ax.zaxis.line.set_visible(False)

    plt.savefig(os.path.join(save_dir, "nsga2_3d_view_top.png"), dpi=300, bbox_inches="tight")
    plt.close()

    plt.figure(figsize=(10, 6))
    gen_range = range(len(gen_avg_fitness))
    shadow_width = np.std(gen_avg_fitness) * 0.5

    plt.plot(gen_range, gen_avg_fitness, label="Average Fitness",
             color="#1f77b4", linewidth=2.5, marker='o', markersize=12, markevery=5)

    plt.fill_between(gen_range,
                     np.array(gen_avg_fitness) + shadow_width,
                     np.array(gen_avg_fitness) - shadow_width,
                     color="#1f77b4", alpha=0.2)

    plt.xlabel("Generation", fontsize=16)
    plt.ylabel("Average Fitness", fontsize=16)
    plt.tick_params(axis='x',labelsize=14)
    plt.tick_params(axis='y',labelsize=14)
    plt.title("Average Fitness Convergence Curve", fontsize=18)
    plt.legend(fontsize=15)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "avg_fitness_convergence_with_shadow.png"), dpi=300, bbox_inches="tight")
    plt.close()

    plt.figure(figsize=(10, 6))
    gen_range = range(len(gen_best_fitness))
    shadow_width_best = np.std(gen_best_fitness) * 0.5

    plt.plot(gen_range, gen_best_fitness, label="Best Fitness",
             color="#ff7f0e", linewidth=2.5, marker='s', markersize=12, markevery=5)

    plt.fill_between(gen_range,
                     np.array(gen_best_fitness) + shadow_width_best,
                     np.array(gen_best_fitness) - shadow_width_best,
                     color="#ff7f0e", alpha=0.2)

    plt.xlabel("Generation", fontsize=16)
    plt.ylabel("Best Fitness", fontsize=16)
    plt.tick_params(axis='x',labelsize=14)
    plt.tick_params(axis='y',labelsize=14)
    plt.title("Best Fitness Convergence Curve", fontsize=18)
    plt.legend(fontsize=15)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "best_fitness_convergence_with_shadow.png"), dpi=300, bbox_inches="tight")
    plt.close()

    def calculate_fitness(individual):
        return 0.5 * individual.objectives[0] + 0.3 * individual.objectives[1] + 0.2 * individual.objectives[2]

    nsga2_fitness = [calculate_fitness(ind) for ind in final_population]
    color = 'green'
    idx = 1

    fig, ax = plt.subplots(figsize=(6, 6))

    box_positions = [1.8]
    box = ax.boxplot(
        [nsga2_fitness],
        patch_artist=True,
        widths=0.2,
        showfliers=False,
        positions=box_positions
    )
    for patch in box['boxes']:
        patch.set_facecolor(color)

    x_scatter = np.random.normal(idx + 1, 0.05, len(nsga2_fitness))
    ax.scatter(x_scatter, nsga2_fitness, color=color, alpha=0.6, s=10)

    kde = stats.gaussian_kde(nsga2_fitness, bw_method='scott')
    y_vals = np.linspace(min(nsga2_fitness), max(nsga2_fitness), 200)
    density = kde(y_vals)
    density_norm = density / np.max(density)
    scale = 0.2
    x_density = (idx + 1) + density_norm * scale
    ax.plot(x_density, y_vals, color=color, alpha=0.8, linewidth=2)

    ax.set_xlabel('Algorithm Type',fontsize=16)
    ax.set_ylabel('Fitness Value',fontsize=16)
    ax.tick_params(axis='x',labelsize=14)
    ax.tick_params(axis='y',labelsize=14)
    ax.set_xticks([1.8, 2.0])
    ax.set_xticklabels(['NSGA2', ''])
    ax.set_title('Fitness Distribution of NSGA2 Algorithm',fontsize=16)

    from matplotlib.patches import Patch
    from matplotlib.lines import Line2D

    box_patch = Patch(facecolor=color, label='Box Plot')
    violin_line = Line2D([0], [0], color=color, linewidth=2, label='Violin Plot')
    scatter_proxy = Line2D([0], [0], marker='o', color=color, alpha=0.6, linestyle='None', markersize=5,
                           label='Population Points')

    ax.grid(True, alpha=0.3, linestyle='--')

    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "nsga2_fitness_violin_with_axes_labels.png"), dpi=300, bbox_inches="tight")
    plt.close()

    print(f"NSGA2 iterations: {generations}")
    print(f"Final best fitness value: {final_best_fitness:.2f}")
    print("All charts saved to:", os.path.abspath(save_dir))