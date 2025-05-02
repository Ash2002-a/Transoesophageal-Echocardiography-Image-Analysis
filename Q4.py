import scipy.io as sio
import numpy as np
import matplotlib.pyplot as plt
from skimage.metrics import structural_similarity
from scipy import stats
import cv2
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import pandas as pd

# Load the dataset
data = sio.loadmat('cw2.mat')

# Extract the necessary data from the loaded file
test_images = data['test_img']
gold_images = data['gold_img']
critical_percentage = data['crit_perc']
generated_impressions = data['gen_impr']

# Locations of missing image data
missing_data = [(8, 9), (12, 7), (13, 9), (14, 0), (15, 3)]

# Create a mask to handle missing data
valid_mask = np.ones((20, 10), dtype=bool)
for i, j in missing_data:
    valid_mask[i, j] = False

print("Data loaded successfully")

# Part i) Calculate rotation and translation using ECC algorithm
def calculate_rigid_transformation(img1, img2):
    """
    Calculate rigid transformation between two images using ECC algorithm
    Returns rotation (degrees) and translation (pixel units)
    """
    # Convert images to 8-bit format required by ECC algorithm
    img1_8bit = (img1 * 255).astype(np.uint8)
    img2_8bit = (img2 * 255).astype(np.uint8)
    
    # Define transformation type (rigid)
    warp_mode = cv2.MOTION_EUCLIDEAN
    
    # Define the warp matrix for rigid transformation (2x3)
    warp_matrix = np.eye(2, 3, dtype=np.float32)
    
    # Define termination criteria
    criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 1000, 1e-10)
    
    try:
        # Apply ECC algorithm
        _, warp_matrix = cv2.findTransformECC(
            img1_8bit, img2_8bit, warp_matrix, warp_mode, criteria, None, 1
        )
        
        # Extract rotation angle from warp matrix
        # For a 2x3 matrix [m11 m12 m13; m21 m22 m23], angle = atan2(m21, m11)
        rotation_rad = np.arctan2(warp_matrix[1, 0], warp_matrix[0, 0])
        rotation_deg = np.degrees(rotation_rad)
        
        # Extract translation components
        tx = warp_matrix[0, 2]  # m13
        ty = warp_matrix[1, 2]  # m23
        
        # Calculate total displacement
        displacement = np.sqrt(tx**2 + ty**2)
        
        return rotation_deg, tx, ty, displacement
    except cv2.error:
        # In case ECC algorithm fails
        return np.nan, np.nan, np.nan, np.nan

# Initialize arrays to store results
rotation_values = np.zeros((20, 10))
tx_values = np.zeros((20, 10))
ty_values = np.zeros((20, 10))
displacement_values = np.zeros((20, 10))

# Calculate transformation for all valid images
for row in range(20):
    for col in range(10):
        if valid_mask[row, col]:
            test_image = test_images[row][col]
            gold_standard_image = gold_images[0][col]
            
            # Calculate rigid transformation
            rot, tx, ty, disp = calculate_rigid_transformation(test_image, gold_standard_image)
            
            rotation_values[row, col] = rot
            tx_values[row, col] = tx
            ty_values[row, col] = ty
            displacement_values[row, col] = disp
        else:
            rotation_values[row, col] = np.nan
            tx_values[row, col] = np.nan
            ty_values[row, col] = np.nan
            displacement_values[row, col] = np.nan

# Part i) Create and display a table of the calculated values
data_dict = {
    'Participant': [f'Participant {i+1}' for i in range(20)],
    'Rotation (degrees)': [np.nanmean(rotation_values[i, :]) if np.any(valid_mask[i, :]) else np.nan for i in range(20)],
    'Tx (pixels)': [np.nanmean(tx_values[i, :]) if np.any(valid_mask[i, :]) else np.nan for i in range(20)],
    'Ty (pixels)': [np.nanmean(ty_values[i, :]) if np.any(valid_mask[i, :]) else np.nan for i in range(20)],
    'Displacement (pixels)': [np.nanmean(displacement_values[i, :]) if np.any(valid_mask[i, :]) else np.nan for i in range(20)],
}

# Creating a pandas DataFrame
df = pd.DataFrame(data_dict)

# Display the table
print("\nCalculated Rotation and Translation Values:")
print(df)

for row in range(20):
    for col in range(10):
        if valid_mask[row, col]:
            print(f"{row+1:<12} {col+1:<6} {rotation_values[row, col]:<15.2f} {tx_values[row, col]:<15.2f} "
                  f"{ty_values[row, col]:<15.2f} {displacement_values[row, col]:<20.2f}")

# Part ii) Statistical analysis between expert and novice groups
# Define expert and novice groups
expert_indices = list(range(7))
novice_indices = list(range(7, 20))

def perform_hypothesis_test(values, view, expert_indices, novice_indices):
    """Perform Mann-Whitney U test to compare expert and novice groups"""
    # Extract expert and novice values
    expert_vals = [values[i, view] for i in expert_indices if not np.isnan(values[i, view])]
    novice_vals = [values[i, view] for i in novice_indices if not np.isnan(values[i, view])]
    
    # Only perform test if we have enough data
    if len(expert_vals) > 0 and len(novice_vals) > 0:
        # Hypothesis: Experts will have better alignment (lower values for displacement and rotation)
        stat, p_val = stats.mannwhitneyu(expert_vals, novice_vals, alternative='less')
        return stat, p_val, np.mean(expert_vals), np.mean(novice_vals)
    else:
        return np.nan, np.nan, np.nan, np.nan

# Perform hypothesis tests for rotation and displacement for each view
# Perform hypothesis tests for rotation and displacement for each view
print("\nPart ii) Hypothesis Testing Results:")
print("=" * 100)
print(f"{'View':<6} {'Metric':<15} {'Expert Mean':<15} {'Novice Mean':<15} {'p-value':<10} {'Significant (p<0.05)':<20}")
print("=" * 100)

rotation_significant_views = []
displacement_significant_views = []

for view in range(10):
    # Test for rotation
    stat_rot, p_rot, expert_mean_rot, novice_mean_rot = perform_hypothesis_test(
        np.abs(rotation_values), view, expert_indices, novice_indices
    )
    
    # Test for displacement
    stat_disp, p_disp, expert_mean_disp, novice_mean_disp = perform_hypothesis_test(
        displacement_values, view, expert_indices, novice_indices
    )
    
    # Check if significant
    rot_significant = "Yes" if p_rot < 0.05 else "No"
    disp_significant = "Yes" if p_disp < 0.05 else "No"
    
    # Store view if significant
    if rot_significant == "Yes":
        rotation_significant_views.append(view)
    if disp_significant == "Yes":
        displacement_significant_views.append(view)
    
    # Print results for rotation
    print(f"{view+1:<6} {'Rotation (abs)':<15} {expert_mean_rot:<15.2f} {novice_mean_rot:<15.2f} {p_rot:<10.4f} {rot_significant:<20}")
    
    # Print results for displacement
    print(f"{view+1:<6} {'Displacement':<15} {expert_mean_disp:<15.2f} {novice_mean_disp:<15.2f} {p_disp:<10.4f} {disp_significant:<20}")

# Discussion of results for part ii)
print("\nDiscussion of Hypothesis Test Results:")
print("=" * 80)
print(f"Number of views with significant difference in rotation: {len(rotation_significant_views)}")
print(f"Views with significant difference in rotation: {[v+1 for v in rotation_significant_views]}")
print(f"Number of views with significant difference in displacement: {len(displacement_significant_views)}")
print(f"Views with significant difference in displacement: {[v+1 for v in displacement_significant_views]}")


# Part iii) Linear regression analysis
# Prepare data for regression
def perform_regression(X, y):
    """Perform linear regression and return RMSE and R² scores"""
    # Create and fit regression model
    model = LinearRegression()
    model.fit(X, y)
    
    # Make predictions
    y_pred = model.predict(X)
    
    # Calculate metrics
    rmse = np.sqrt(mean_squared_error(y, y_pred))
    r2 = r2_score(y, y_pred)
    
    return model, rmse, r2, y_pred

# Part iii) Linear regression analysis

# Initialize storage for regression results
regression_results = {}

# Perform regression for each view
for view in range(10):
    regression_results[view] = {
        'rot_vs_critperc': {'rmse': np.nan, 'r2': np.nan},
        'rot_vs_genimpr': {'rmse': np.nan, 'r2': np.nan},
        'disp_vs_critperc': {'rmse': np.nan, 'r2': np.nan},
        'disp_vs_genimpr': {'rmse': np.nan, 'r2': np.nan}
    }
    
    # Get valid participants for this view
    valid_participants = [i for i in range(20) if not np.isnan(rotation_values[i, view])]
    
    if len(valid_participants) > 2:  # Need at least 3 data points for meaningful regression
        # Prepare independent variables (rotation and displacement)
        rot_data = np.abs(rotation_values[valid_participants, view]).reshape(-1, 1)
        disp_data = displacement_values[valid_participants, view].reshape(-1, 1)
        
        # Prepare dependent variables (criteria percentage and general impression)
        crit_perc = critical_percentage[valid_participants, view]
        gen_impr = generated_impressions[valid_participants, view]
        
        # Perform regressions
        # 1. Rotation vs Criteria Percentage
        _, rmse_rot_crit, r2_rot_crit, _ = perform_regression(rot_data, crit_perc)
        regression_results[view]['rot_vs_critperc'] = {'rmse': rmse_rot_crit, 'r2': r2_rot_crit}
        
        # 2. Rotation vs General Impression
        _, rmse_rot_gen, r2_rot_gen, _ = perform_regression(rot_data, gen_impr)
        regression_results[view]['rot_vs_genimpr'] = {'rmse': rmse_rot_gen, 'r2': r2_rot_gen}
        
        # 3. Displacement vs Criteria Percentage
        _, rmse_disp_crit, r2_disp_crit, _ = perform_regression(disp_data, crit_perc)
        regression_results[view]['disp_vs_critperc'] = {'rmse': rmse_disp_crit, 'r2': r2_disp_crit}
        
        # 4. Displacement vs General Impression
        _, rmse_disp_gen, r2_disp_gen, _ = perform_regression(disp_data, gen_impr)
        regression_results[view]['disp_vs_genimpr'] = {'rmse': rmse_disp_gen, 'r2': r2_disp_gen}

# Display regression results
print("\nPart iii) Linear Regression Results:")
print("=" * 80)
print(f"{'View':<6} {'Regression':<20} {'RMSE':<10} {'R²':<10}")
print("-" * 80)

for view in range(10):
    print(f"View {view+1}:")
    for regression_type, metrics in regression_results[view].items():
        print(f"{'':<6} {regression_type:<20} {metrics['rmse']:<10.4f} {metrics['r2']:<10.4f}")

# Calculate combined performance score for each view
view_performance = {}
for view in range(10):
    # Sum of R² values (higher is better)
    r2_sum = (regression_results[view]['rot_vs_critperc']['r2'] + 
             regression_results[view]['rot_vs_genimpr']['r2'] +
             regression_results[view]['disp_vs_critperc']['r2'] +
             regression_results[view]['disp_vs_genimpr']['r2'])
    
    # Average RMSE (lower is better)
    rmse_avg = (regression_results[view]['rot_vs_critperc']['rmse'] + 
               regression_results[view]['rot_vs_genimpr']['rmse'] +
               regression_results[view]['disp_vs_critperc']['rmse'] +
               regression_results[view]['disp_vs_genimpr']['rmse']) / 4
    
    # Store combined score (higher R² and lower RMSE is better)
    view_performance[view] = {
        'r2_sum': r2_sum,
        'rmse_avg': rmse_avg,
        'combined_score': r2_sum - rmse_avg  # Simple combined metric
    }

# Find top 3 views based on combined score
top_views = sorted(view_performance.keys(), 
                   key=lambda v: view_performance[v]['combined_score'], 
                   reverse=True)[:3]

print("\nTop 3 Best Performing Views:")
print("-" * 80)
for rank, view in enumerate(top_views):
    print(f"Rank {rank+1}: View {view+1}")
    print(f"  Total R²: {view_performance[view]['r2_sum']:.4f}")
    print(f"  Average RMSE: {view_performance[view]['rmse_avg']:.4f}")
    print(f"  Combined Score: {view_performance[view]['combined_score']:.4f}")

# Plot regression results for the top 3 views
plt.figure(figsize=(18, 15))

regression_types = [
    ('rot_vs_critperc', 'Rotation vs Critical Percentage'),
    ('rot_vs_genimpr', 'Rotation vs General Impression'),
    ('disp_vs_critperc', 'Displacement vs Critical Percentage'),
    ('disp_vs_genimpr', 'Displacement vs General Impression')
]

for i, view in enumerate(top_views):
    valid_participants = [p for p in range(20) if not np.isnan(rotation_values[p, view])]
    
    rot_data = np.abs(rotation_values[valid_participants, view]).reshape(-1, 1)
    disp_data = displacement_values[valid_participants, view].reshape(-1, 1)
    crit_perc = critical_percentage[valid_participants, view]
    gen_impr = generated_impressions[valid_participants, view]
    
    for j, (reg_key, reg_title) in enumerate(regression_types):
        plt.subplot(3, 4, i * 4 + j + 1)
        
        # Determine X and Y based on regression type
        if 'rot_vs' in reg_key:
            X = rot_data
            y = crit_perc if 'critperc' in reg_key else gen_impr
        else:
            X = disp_data
            y = crit_perc if 'critperc' in reg_key else gen_impr
        
        # Plot the data and regression line
        plt.scatter(X, y, color='blue', label='Data')
        model, _, _, y_pred = perform_regression(X, y)
        
        plt.plot(X, y_pred, color='red', label='Regression Line')
        plt.title(f"{reg_title} (View {view+1})")
        plt.xlabel("Rotation" if 'rot' in reg_key else "Displacement")
        plt.ylabel("Criteria Percentage" if 'critperc' in reg_key else "General Impression")
        plt.legend()

plt.tight_layout()
plt.show()
