import pandas as pd
import numpy as np
import random

# Step 1: Load data
orders = pd.read_csv('order_large.csv')
distances = pd.read_csv('distance.csv')

# Step 2: Clean & prepare data
orders = orders.dropna(subset=['Source', 'Destination', 'Weight'])
orders = orders.drop_duplicates(subset=['Source', 'Destination', 'Weight'])

# Step 3: Create distance lookup dictionary
distance_dict = {}
for _, row in distances.iterrows():
    key = (row['Source'], row['Destination'])
    distance_dict[key] = row['Distance(M)']

# Step 4: Filter orders with known distances
orders['Valid'] = orders.apply(lambda row: (row['Source'], row['Destination']) in distance_dict, axis=1)
orders = orders[orders['Valid']]

# Step 5: Parameters for ACO (fixed values)
MAX_WEIGHT = 10000000  # Max weight per truck
ANTS = 100
ITERS = 500
ALPHA = 1
BETA = 3
EVAPORATION = 0.1
Q = 100

# Step 6: Assign orders to trucks
truck_orders = []
current_truck = []
current_weight = 0
for _, order in orders.iterrows():
    if current_weight + order['Weight'] <= MAX_WEIGHT:
        current_truck.append(order)
        current_weight += order['Weight']
    else:
        truck_orders.append(current_truck)
        current_truck = [order]
        current_weight = order['Weight']
if current_truck:
    truck_orders.append(current_truck)

# Step 7: 2-opt local search
def total_distance(route, distance_matrix):
    return sum(distance_matrix[route[i]][route[i+1]] for i in range(len(route)-1))

def two_opt(route, distance_matrix):
    best = route
    improved = True
    while improved:
        improved = False
        for i in range(1, len(best) - 2):
            for j in range(i + 1, len(best)):
                if j - i == 1: continue
                new_route = best[:i] + best[i:j][::-1] + best[j:]
                if total_distance(new_route, distance_matrix) < total_distance(best, distance_matrix):
                    best = new_route
                    improved = True
    return best

# Step 8: ACO function
def aco_route(orders_subset, ants, iters, alpha, beta, evap, Q):
    locations = list(set([o['Source'] for o in orders_subset] + [o['Destination'] for o in orders_subset]))
    n = len(locations)
    location_index = {loc: idx for idx, loc in enumerate(locations)}

    distance_matrix = np.full((n, n), np.inf)
    for i in range(n):
        for j in range(n):
            if i != j:
                key = (locations[i], locations[j])
                if key in distance_dict:
                    distance_matrix[i][j] = distance_dict[key]

    pheromones = np.ones((n, n))
    best_route = None
    best_distance = np.inf

    for _ in range(iters):
        all_routes = []
        all_distances = []

        for _ in range(ants):
            route = [random.randint(0, n-1)]
            visited = set(route)
            total_dist = 0

            while len(visited) < n:
                current = route[-1]
                probabilities = []
                for j in range(n):
                    if j not in visited and distance_matrix[current][j] < np.inf:
                        tau = pheromones[current][j] ** alpha
                        eta = (1 / distance_matrix[current][j]) ** beta
                        probabilities.append((j, tau * eta))
                if not probabilities:
                    break
                total = sum(p for _, p in probabilities)
                probs = [p / total for _, p in probabilities]
                next_node = np.random.choice([j for j, _ in probabilities], p=probs)
                total_dist += distance_matrix[route[-1]][next_node]
                route.append(next_node)
                visited.add(next_node)

            if len(route) == n:
                all_routes.append(route)
                all_distances.append(total_dist)
                if total_dist < best_distance:
                    best_distance = total_dist
                    best_route = route

        pheromones *= (1 - evap)
        for route, dist in zip(all_routes, all_distances):
            for i in range(len(route) - 1):
                pheromones[route[i]][route[i + 1]] += Q / dist

    if best_route:
        best_route = two_opt(best_route, distance_matrix)
        best_distance = total_distance(best_route, distance_matrix)

    return [locations[i] for i in best_route], best_distance

# Step 9: Run ACO with fixed parameters on all trucks
with open('optimized_routes_op.txt', 'w') as f:
    for idx, truck in enumerate(truck_orders):
        truck_df = pd.DataFrame(truck)
        if truck_df.empty:
            continue
        try:
            route, total_dist = aco_route(truck_df.to_dict('records'), ANTS, ITERS, ALPHA, BETA, EVAPORATION, Q)
            f.write(f"Truck {idx+1}:\n")
            f.write(f"  Optimized Route: {' -> '.join(route)}\n")
            f.write(f"  Total Distance: {total_dist:.2f} meters\n\n")
        except Exception as e:
            f.write(f"Truck {idx+1}: Failed to compute route\n\n")
