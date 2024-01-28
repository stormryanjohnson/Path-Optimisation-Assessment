import pandas as pd
import numpy as np
from sklearn.svm import SVR
import matplotlib.pyplot as plt
import time
import heapq

# Start time counter
start = time.perf_counter()

# *------------------------STEP 1: Ingestion------------------------*

# Read altitude map and energy expenditure data
altitude_df = pd.read_csv('./data/altitude_map.csv', header=None)
expenditure_df = pd.read_csv('./data/energy_cost.csv')
alt_len = len(altitude_df) - 1


# *------------------------STEP 2: Modelling------------------------*

'''
NB:     For brevity, the chosen model is a Support Vector Machine (SVM). 
        SVM is appropriate because of the nonlinear nature of the data, and it generally works well with small datasets.
        Some alternative models to consider include: 
            - Any tree-based regressor,
            - Linear regression with polynomial feature transformation.
        The optimal hyperparameters for the SVM model were obtained through tuning in the model_training/model_training.ipynb notebook.
'''

# Build a model to predict energy expenditure based on gradient
X = expenditure_df['gradient'].values.reshape(-1, 1)
y = expenditure_df['energy_cost'].values
model = SVR(C=800, gamma=8)  # model hyperparameters were obtained through tuning in the model_training/model_training.ipynb notebook
model.fit(X, y)

# Generate a range of gradient values for prediction
X_pred = np.linspace(X.min(), X.max(), 100).reshape(-1, 1)

# Use the model to predict energy expenditure for these gradient values
y_pred = model.predict(X_pred)

# Plot and save original data points
plt.plot(X, y, '.', label='Original Data', color='blue')
plt.xlabel('Gradient')
plt.ylabel(r'Energy Expenditure  ($J.kg^{-1}.min^{-1}$)')
plt.legend()
plt.savefig('./data/energy_cost.png', dpi=1000, bbox_inches='tight')

# Convert the gradient DataFrames to a NumPy array for plotting
altitude_data = altitude_df.values
gradient_x = np.gradient(altitude_df.values, axis=1)  # Gradient along the columns (x-axis)
gradient_y = np.gradient(altitude_df.values, axis=0)  # Gradient along the rows (y-axis)

# Use model to predict energy expenditure from each gradient direction
energy_x = model.predict(gradient_x.reshape(-1,1)).reshape(altitude_df.shape)
energy_y = model.predict(gradient_y.reshape(-1,1)).reshape(altitude_df.shape)

energy_total = energy_x + energy_y

# Create a DataFrame to store the gradient and energy expenditure values
gradient_x_df = pd.DataFrame(gradient_x, columns=altitude_df.columns)
gradient_y_df = pd.DataFrame(gradient_y, columns=altitude_df.columns)
energy_df = pd.DataFrame(energy_total, columns=altitude_df.columns)

# Visualise altitude and energy expenditure through heatmaps
plt.figure(figsize=(10, 8))  
plt.imshow(altitude_data, cmap='viridis', origin='lower', aspect='auto')
plt.colorbar(label='Altitude (meters)')  
plt.xlabel(r'X Coordinate ($\times$10 meters)')
plt.ylabel(r'Y Coordinate ($\times$10 meters)')
plt.grid(False)  
plt.savefig('./data/altitude_map.png', dpi=1000, bbox_inches='tight')

plt.figure(figsize=(10, 8))  
plt.imshow(energy_df, cmap='plasma', origin='lower', aspect='auto')
plt.colorbar(label=r'Energy Expenditure ($J.kg^{-1}.min^{-1}$)')
plt.xlabel('X Coordinate (10 meters)')
plt.ylabel('Y Coordinate (10 meters)')
plt.grid(False)
plt.savefig('./output/energy_expenditure_map.png', dpi=1000, bbox_inches='tight')


# *------------------------STEP 3: Optimisation------------------------*

'''
NB:     Optimisation currently takes ~40 min on standard hardware. This can be improved by introducing multi-threading processes,
        or using more powerful hardware. Due to the time constraints of this assessment, this improvement was not included.
'''

# Create DataFrame which has energy expenditure corresponding to its (x,y) coordinates
energy_list = []
x_list = []
y_list = []
data = pd.DataFrame()
for i in range(energy_df.shape[0]):
    for j in range(energy_df.shape[1]):
        energy_list.append(energy_df.iloc[i, j])
        x_list.append(j)
        y_list.append(alt_len - i)  # Flip DataFrame about the x-axis to set coordinate system with (0,0) as the South-West corner
data['x'] = x_list
data['y'] = y_list
data['energy'] = energy_list

coordinates = data[['x', 'y']].values
energies = data['energy'].values

# Use Dijkstra's Algorithm to minimise energy expenditure path. (x,y) are used as nodes, and energy expenditure as weight edges
def optimise_path_using_dijkstra_algorithm(start_point, end_point):
    # Create a mapping from coordinates to indices
    coord_to_index = {tuple(coord): idx for idx, coord in enumerate(coordinates)}

    # Create a list of neighbors for each coordinate
    def get_neighbors(coord):
        x, y = coord
        neighbors = [(x, y + 1), (x, y - 1), (x + 1, y), (x - 1, y), (x - 1, y - 1), (x + 1, y + 1)]  # Degrees of freedom of path at point (x,y)
        return [neighbor for neighbor in neighbors if tuple(neighbor) in coord_to_index]

    # Initialize the distance dictionary
    distance = {tuple(coord): float('inf') for coord in coordinates}
    distance[start_point] = 0

    # Initialize a priority queue with the start point
    pq = [(0, start_point)]

    while pq:
        dist, current_coord = heapq.heappop(pq)
        if current_coord == end_point:
            break
        
        for neighbor_coord in get_neighbors(current_coord):
            neighbor_idx = coord_to_index[neighbor_coord]
            energy_cost = energies[neighbor_idx]  # Energy cost to move to the neighbor
            new_dist = dist + energy_cost
            
            if new_dist < distance[neighbor_coord]:
                distance[neighbor_coord] = new_dist
                heapq.heappush(pq, (new_dist, neighbor_coord))

    # Reconstruct the optimal path
    optimal_path = [end_point]
    current_coord = end_point

    while current_coord != start_point:
        neighbors = get_neighbors(current_coord)
        min_neighbor = min(neighbors, key=lambda coord: distance[coord])
        optimal_path.append(min_neighbor)
        current_coord = min_neighbor

    optimal_path.reverse()

    return optimal_path, distance[end_point]

# Loop through starting point along Southern border to find optimal path starting point
print('Optimisation progress:')
for i in range(energy_df.shape[1]):
    # Define the target coordinates
    start_point = (i, 0)
    end_point = (200, 559)

    path, energy = optimise_path_using_dijkstra_algorithm(start_point, end_point)
    if not i:
        optimal_path = path
        optimal_energy = energy
    else:
        if energy < optimal_energy:
            optimal_path = path
            optimal_energy = energy
    print(str(i) + ' / ' + str(energy_df.shape[1]))

print('Optimal energy expenditure: ', optimal_energy, 'J/(kg.min)')


# *------------------------STEP 4: Simple reporting------------------------*

# Write the optimal path coordinates to a CSV file
path_df = pd.DataFrame(optimal_path, columns=['x_coord', 'y_coord'])
path_df.to_csv('./output/optimal_path.csv', index=False)

# Create a visualization of the altitude map with the optimal path
plt.figure(figsize=(10, 8)) 
plt.imshow(altitude_data, cmap='viridis', origin='lower', aspect='auto')
plt.colorbar(label='Altitude (meters)')  
plt.xlabel(r'X Coordinate ($\times$10 meters)')
plt.ylabel(r'Y Coordinate ($\times$10 meters)')
plt.grid(False) 
for x, y in optimal_path:
    plt.scatter(x=x, y=y, c='r', s=5)  # Mark the optimal path in red
plt.savefig('./output/altitude_map_with_path.png', dpi=1000, bbox_inches='tight')


# End time counter
end = time.perf_counter()

print('Total execution time of script: ', end - start, ' seconds')