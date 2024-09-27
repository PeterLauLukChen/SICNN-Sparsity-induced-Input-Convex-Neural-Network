import numpy as np
import ot
import os

def smoothed_l0(T_x_minus_x, sigma):
    smoothed_l0_values = 1 - np.exp(-T_x_minus_x**2 / (2 * sigma**2))
    summed_smoothed_l0 = np.sum(smoothed_l0_values, axis=1)
    return summed_smoothed_l0

def compute_gradient(T, X, T_GT, l, gamma, beta, sigma):
    T_x_minus_x = T - X
    smooth_l0 = smoothed_l0(T_x_minus_x, sigma)
    #grad = 2 * (T_x_minus_x)
    grad = 0
    mask = (smooth_l0 > l).astype(float)
    grad += gamma * mask[:, None] * T_x_minus_x / sigma**2 * np.exp(-T_x_minus_x**2 / (2 * sigma**2))
    grad += 2 * beta * (T - T_GT)
    return grad

def gradient_descent(T, X, T_GT, l, gamma, beta, sigma, r=0.01, max_iter=1000, tol=1e-6):
    for i in range(max_iter):
        grad = compute_gradient(T, X, T_GT, l, gamma, beta, sigma)
        T_new = T - r * grad
        
        if i % 100 == 0:
            T_x_minus_x = T_new - X
            # Calculate the direct L0 norm for the first displacement vector
            direct_l0_norm = calculate_dimension(T_x_minus_x[0])
            print(f"Iteration {i}:")
            print(f"  Displacement vector 1 - Direct L0 Norm: {direct_l0_norm}")
        
        if np.linalg.norm(T_new - T) < tol:
            break
            
        T = T_new
    
    return T

def calculate_dimension(vector, threshold=1):
    thresholded_vector = np.where(np.abs(vector) < threshold, 0, vector)
    dimension = np.count_nonzero(thresholded_vector)
    return dimension

# Function to calculate statistics for a list of displacement vectors using direct L0 norm
def calculate_statistics(displacement_vectors, threshold=1):
    dimensions = [calculate_dimension(vec, threshold) for vec in displacement_vectors]
    
    # Calculate max, mean, and non-outlier range (using IQR)
    max_dimension = np.max(dimensions)
    mean_dimension = np.mean(dimensions)
    q1, q3 = np.percentile(dimensions, [25, 75])
    iqr = q3 - q1
    non_outlier_range = (q1 - 1.5 * iqr, q3 + 1.5 * iqr)
    return max_dimension, mean_dimension, non_outlier_range

# Parameters for gradient descent
gamma = 1.4
beta = 1
sigma = 1
l = 20
r = 0.01
max_iter = 10000
tol = 1e-6

current_folder = os.path.dirname(os.path.abspath(__file__))
data_folder_sicnn = os.path.join(current_folder, 'data')
control_data = np.load(os.path.join(data_folder_sicnn, 'control.npy'))
treated_data = np.load(os.path.join(data_folder_sicnn, 'treated.npy'))

# Solving GT map
num_cells = control_data.shape[0]
cost_matrix = ot.dist(control_data, treated_data, metric='euclidean')
a = np.ones((num_cells,)) / num_cells
b = np.ones((num_cells,)) / num_cells
transport_plan = ot.emd(a, b, cost_matrix)


max_value_index = np.unravel_index(np.argmax(transport_plan), transport_plan.shape)
control_cell_index, treated_cell_index = max_value_index
displacement_vector = treated_data[treated_cell_index] - control_data[control_cell_index]

# Initialize T as the ground truth (GT) map
T_GT = np.zeros_like(control_data)
for i in range(num_cells):
    T_GT[i] = control_data[i] + displacement_vector

T_initial = T_GT.copy()
T_optimal = gradient_descent(T_initial, control_data, T_GT, l, gamma, beta, sigma, r, max_iter, tol)
final_displacement_vector = T_optimal[0] - control_data[0]
final_displacement_formatted = [f"{item:.4f}" for item in final_displacement_vector]
print("Final displacement vector for the first element in X after optimization:")
print(final_displacement_formatted)

# Calculate displacement vectors for GT and optimal maps
gt_displacement_vectors = T_GT - control_data
optimal_displacement_vectors = T_optimal - control_data

gt_max_dim, gt_mean_dim, gt_non_outlier_range = calculate_statistics(gt_displacement_vectors)
print("GT Displacement Vectors - Max Dimension:", gt_max_dim)
print("GT Displacement Vectors - Mean Dimension:", f"{gt_mean_dim:.2f}")
print("GT Displacement Vectors - Non-Outlier Range:", gt_non_outlier_range)

opt_max_dim, opt_mean_dim, opt_non_outlier_range = calculate_statistics(optimal_displacement_vectors)
print("Optimal Displacement Vectors - Max Dimension:", opt_max_dim)
print("Optimal Displacement Vectors - Mean Dimension:", f"{opt_mean_dim:.2f}")
print("Optimal Displacement Vectors - Non-Outlier Range:", opt_non_outlier_range)
