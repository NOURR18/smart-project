import pandas as pd
import numpy as np
import random

# Step 1: Load data
orders = pd.read_csv('order_large.csv')
distances = pd.read_csv('distance.csv')

# Step 2: Clean & prepare data
orders = orders.dropna(subset=['Source', 'Destination', 'Weight'])

# Step 3: Create distance lookup dictionary
distance_dict = {}
for _, row in distances.iterrows():
    key = (row['Source'], row['Destination'])
    distance_dict[key] = row['Distance(M)']
    distance_dict[(row['Destination'], row['Source'])] = row['Distance(M)']  # Assuming symmetric distances

# Step 4: Filter orders with known distances
orders['Valid'] = orders.apply(lambda row: (row['Source'], row['Destination']) in distance_dict, axis=1)
orders = orders[orders['Valid']]

# Step 5: Parameters
MAX_WEIGHT = 10000000  # Max weight per truck (example: 10 tons)
NUM_ANTS = 10
NUM_ITERATIONS = 30
ALPHA = 1  # influence of pheromone
BETA = 2   # influence of distance
EVAPORATION = 0.5
Q = 100

# Step 6: Assign orders to trucks based on max weight
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

# Step 7: ACO for routing inside each truck's orders
def aco_route(orders_subset):
    locations = list(set([o['Source'] for o in orders_subset] + [o['Destination'] for o in orders_subset]))
    n = len(locations)
    location_index = {loc: idx for idx, loc in enumerate(locations)}
    
    # Distance matrix
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

    for _ in range(NUM_ITERATIONS):
        all_routes = []
        all_distances = []

        for _ in range(NUM_ANTS):
            route = [random.randint(0, n-1)]
            visited = set(route)
            total_distance = 0

            while len(visited) < n:
                current = route[-1]
                probabilities = []
                for j in range(n):
                    if j not in visited and distance_matrix[current][j] < np.inf:
                        tau = pheromones[current][j] ** ALPHA
                        eta = (1 / distance_matrix[current][j]) ** BETA
                        probabilities.append((j, tau * eta))
                if not probabilities:
                    break
                total = sum(p for _, p in probabilities)
                probs = [p / total for _, p in probabilities]
                next_node = np.random.choice([j for j, _ in probabilities], p=probs)
                total_distance += distance_matrix[route[-1]][next_node]
                route.append(next_node)
            visited.add(next_node)

            if len(route) == n:
                all_routes.append(route)
                all_distances.append(total_distance)
                if total_distance < best_distance:
                    best_distance = total_distance
                    best_route = route

       # Update pheromones
        pheromones *= (1 - EVAPORATION)
        for route, dist in zip(all_routes, all_distances):
            for i in range(len(route) - 1):
                pheromones[route[i]][route[i+1]] += Q / dist

    return [locations[i] for i in best_route], best_distance

# Step 8: Run ACO on each truck and print results
with open('optimized_routes.txt', 'w') as f:
    for idx, truck in enumerate(truck_orders):
        truck_df = pd.DataFrame(truck)
        if truck_df.empty:
            f.write(f"Truck {idx+1} has no orders. Skipping.\n\n")
            continue  #skip empty track

        route, total_dist = aco_route(truck_df.to_dict('records'))
        f.write(f"Truck {idx+1}:\n")
        f.write(f"  Optimized route: {route}\n")
        f.write(f"  Total distance: {total_dist}\n\n")