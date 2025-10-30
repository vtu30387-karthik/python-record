import numpy as np

# Distance matrix for 10 delivery locations
distances = np.array([
    [0, 29, 20, 21, 16, 31, 100, 12, 4, 31],
    [29, 0, 15, 29, 28, 40, 72, 21, 29, 41],
    [20, 15, 0, 15, 14, 25, 81, 9, 23, 27],
    [21, 29, 15, 0, 4, 12, 92, 12, 25, 13],
    [16, 28, 14, 4, 0, 16, 94, 9, 20, 16],
    [31, 40, 25, 12, 16, 0, 95, 24, 36, 3],
    [100, 72, 81, 92, 94, 95, 0, 90, 101, 99],
    [12, 21, 9, 12, 9, 24, 90, 0, 15, 25],
    [4, 29, 23, 25, 20, 36, 101, 15, 0, 35],
    [31, 41, 27, 13, 16, 3, 99, 25, 35, 0]
])

num_locations = len(distances)
num_ants = 20
num_iterations = 100
alpha = 1       # pheromone importance
beta = 5        # distance importance
evaporation = 0.5
Q = 100         # pheromone deposit factor

pheromone = np.ones((num_locations, num_locations))  # Initialize pheromones

def heuristic(i, j):
    return 1 / (distances[i][j] + 1e-10)  # inverse distance

def select_next_city(current_city, visited):
    probabilities = []
    for city in range(num_locations):
        if city not in visited:
            tau = pheromone[current_city][city] ** alpha
            eta = heuristic(current_city, city) ** beta
            probabilities.append(tau * eta)
        else:
            probabilities.append(0)
    probabilities = np.array(probabilities)
    probabilities /= probabilities.sum()
    return np.random.choice(range(num_locations), p=probabilities)

def total_distance(route):
    dist = 0
    for i in range(len(route) - 1):
        dist += distances[route[i]][route[i+1]]
    dist += distances[route[-1]][route[0]]  # return to start
    return dist

best_route = None
best_distance = float('inf')

for iteration in range(num_iterations):
    all_routes = []
    all_distances = []
    for ant in range(num_ants):
        route = [0]  # start at depot
        while len(route) < num_locations:
            next_city = select_next_city(route[-1], route)
            route.append(next_city)
        all_routes.append(route)
        dist = total_distance(route)
        all_distances.append(dist)
        if dist < best_distance:
            best_distance = dist
            best_route = route
    # Evaporate pheromone
    pheromone *= (1 - evaporation)
    # Deposit pheromone
    for route, dist in zip(all_routes, all_distances):
        deposit = Q / dist
        for i in range(len(route) - 1):
            pheromone[route[i]][route[i+1]] += deposit
            pheromone[route[i+1]][route[i]] += deposit
        pheromone[route[-1]][route[0]] += deposit
        pheromone[route[0]][route[-1]] += deposit

# Print best route with clean integer indices
print("Best route found:", [int(city) for city in best_route])
print("Total distance of best route:", best_distance)