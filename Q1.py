import scipy.io as sio
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import math

# Loading the dataset
data = sio.loadmat('cw2.mat')

# Extract relevant variables
test_image_data = data['test_img']
gold_image_data = data['gold_img']
generated_impression_data = data['gen_impr']
criteria_percentage_data = data['crit_perc']

# Specify missing data indices
missing_data_indices = [(8, 9), (12, 7), (13, 9), (14, 0), (15, 3)]

# Initialize a mask to filter missing data
data_validity_mask = np.ones((20, 10), dtype=bool)
for idx in missing_data_indices:
    data_validity_mask[idx] = False

print("Dataset successfully loaded.")

# Metric combining RMSE and R² for overall performance evaluation
def combined_performance_metric(r2_value, rmse_value, max_rmse_value):
    """
    Combine R² and RMSE into a single performance score.
    
    Parameters:
    ----------
    r2_value : float
        The R-squared value
    rmse_value : float
        Root Mean Squared Error value
    max_rmse_value : float
        Maximum RMSE value to normalize
    
    Returns:
    -------
    float
        A combined performance score
    """
    weight_r2 = 0.5
    weight_rmse = 0.5
    
    # Normalize RMSE by reversing its scale (lower is better)
    normalized_rmse = 1 - (rmse_value / (max_rmse_value + 1e-10))
    
    # Combined performance score
    performance_score = weight_r2 * r2_value + weight_rmse * normalized_rmse
    
    return performance_score

# Part i: Pearson correlation analysis for each view
print("\nPart i - Pearson Correlation Analysis")
print("=" * 50)

pearson_corr_values = []
pearson_p_values = []

for view in range(10):
    # Filter valid data for the current view
    valid_data_indices = data_validity_mask[:, view]
    gen_impression_view = generated_impression_data[valid_data_indices, view]
    criteria_percentage_view = criteria_percentage_data[valid_data_indices, view]
    
    # Compute the Pearson correlation
    corr, p_val = stats.pearsonr(criteria_percentage_view, gen_impression_view)
    pearson_corr_values.append(corr)
    pearson_p_values.append(p_val)
    
    print(f"View {view + 1}: Pearson Correlation = {corr:.4f} (p-value = {p_val:.4f})")

# Identify the view with the highest correlation
best_correlated_view = np.argmax(pearson_corr_values)
print(f"\nBest view based on correlation: View {best_correlated_view + 1} (Correlation = {pearson_corr_values[best_correlated_view]:.4f})")

# Plotting the Pearson correlations
plt.figure(figsize=(10, 6))
plt.bar(range(1, 11), pearson_corr_values, color='seagreen')
plt.axhline(y=0, color='firebrick', linestyle='-', alpha=0.4)
plt.xlabel('View Number')
plt.ylabel('Pearson Correlation')
plt.title('Criteria Percentage vs. General Impression Correlation')
plt.xticks(range(1, 11))
plt.grid(axis='y', alpha=0.3)
plt.savefig('Q1_Pearson_Correlation_Plot.png', dpi=300)
plt.show()

# Part ii: Linear Regression Analysis
print("\nPart ii - Linear Regression Analysis")
print("=" * 50)

rmse_values = []
r2_values = []
regression_models = []

# Display table of results
print(f"{'View':<5} {'RMSE':<10} {'R²':<10} {'Regression Equation':<30}")
print("-" * 55)

for view in range(10):
    # Get valid data for the current view
    valid_data_indices = data_validity_mask[:, view]
    gen_impression_view = generated_impression_data[valid_data_indices, view]
    criteria_percentage_view = criteria_percentage_data[valid_data_indices, view]
    
    # Prepare data for regression (reshape for sklearn)
    X = criteria_percentage_view.reshape(-1, 1)
    y = gen_impression_view
    
    # Create and train the model
    model = LinearRegression()
    model.fit(X, y)
    regression_models.append(model)
    
    # Predictions
    y_pred = model.predict(X)
    
    # Compute RMSE and R²
    rmse = math.sqrt(mean_squared_error(y, y_pred))
    r_squared = r2_score(y, y_pred)
    
    rmse_values.append(rmse)
    r2_values.append(r_squared)
    
    regression_eq = f"y = {model.coef_[0]:.4f}x + {model.intercept_:.4f}"
    print(f"{view + 1:<5} {rmse:<10.4f} {r_squared:<10.4f} {regression_eq:<30}")

# Normalize RMSE by its maximum value
max_rmse = max(rmse_values)

# Combine performance metrics into a single score
combined_scores = []
for view in range(10):
    score = combined_performance_metric(r2_values[view], rmse_values[view], max_rmse)
    combined_scores.append(score)

# Identify top 3 views
top_three_views = np.argsort(combined_scores)[-3:][::-1]

# Display top 3 views based on combined metric
print("\nTop 3 Views Based on Combined Performance:")
print(f"{'View':<5} {'R²':<10} {'RMSE':<10} {'Combined Score':<15}")
print("-" * 40)
for idx in top_three_views:
    print(f"{idx + 1:<5} {r2_values[idx]:<10.4f} {rmse_values[idx]:<10.4f} {combined_scores[idx]:<15.4f}")

# Part iii: Detailed analysis for best performing views
print("\nPart iii - Analysis for Best Performing Views")
print("=" * 50)

plt.figure(figsize=(18, 6))
for idx, view_idx in enumerate(top_three_views):
    valid_data_indices = data_validity_mask[:, view_idx]
    gen_impression_view = generated_impression_data[valid_data_indices, view_idx]
    criteria_percentage_view = criteria_percentage_data[valid_data_indices, view_idx]
    
    # Prepare data for plotting
    X = criteria_percentage_view.reshape(-1, 1)
    y = gen_impression_view
    model = regression_models[view_idx]
    
    # Predictions
    y_pred = model.predict(X)
    
    # Plot regression results
    plt.subplot(1, 3, idx + 1)
    plt.scatter(criteria_percentage_view, gen_impression_view, color='mediumslateblue', alpha=0.7, label='Actual')
    plt.scatter(criteria_percentage_view, y_pred, color='tomato', alpha=0.7, label='Predicted')
    
    # Regression line
    x_line = np.linspace(0, 100, 100).reshape(-1, 1)
    y_line = model.predict(x_line)
    plt.plot(x_line, y_line, color='darkorange', lw=2, label='Regression Line')
    
    # Highlight ranges
    plt.axhspan(0, 2, alpha=0.1, color='lightsalmon', label='0-2 Range')
    plt.axhspan(2, 4, alpha=0.1, color='lightseagreen', label='2-4 Range')
    
    # Display model stats
    plt.text(5, 3.5, 
             f"View {view_idx + 1}\n"
             f"RMSE = {rmse_values[view_idx]:.4f}\n"
             f"R² = {r2_values[view_idx]:.4f}\n"
             f"Combined Score = {combined_scores[view_idx]:.4f}", 
             bbox=dict(facecolor='white', alpha=0.7))
    
    plt.title(f"View {view_idx + 1}")
    plt.xlabel("Criteria Percentage")
    plt.ylabel("General Impression")
    plt.xlim(0, 100)
    plt.ylim(0, 4)
    plt.grid(True, alpha=0.3)
    if idx == 0:
        plt.legend(loc='lower right')

plt.tight_layout()
plt.savefig('Q1_Top_Views_Analysis.png', dpi=300)
plt.show()

# Evaluate model performance within specific score ranges
print("Evaluating model performance within specific score ranges:")
print(f"{'View':<6} {'Range':<10} {'RMSE':<10} {'R²':<10} {'Samples':<10}")
print("-" * 46)

for view_idx in top_three_views:
    valid_data_indices = data_validity_mask[:, view_idx]
    gen_impression_view = generated_impression_data[valid_data_indices, view_idx]
    criteria_percentage_view = criteria_percentage_data[valid_data_indices, view_idx]
    
    # Predictions
    X = criteria_percentage_view.reshape(-1, 1)
    y = gen_impression_view
    model = regression_models[view_idx]
    y_pred = model.predict(X)
    
    # For low score range (0-2)
    low_range_mask = y < 2
    low_count = np.sum(low_range_mask)
    if low_count > 1:
        low_rmse = np.sqrt(mean_squared_error(y[low_range_mask], y_pred[low_range_mask]))
        low_r2 = r2_score(y[low_range_mask], y_pred[low_range_mask])
        print(f"{view_idx + 1:<6} {'0-2':<10} {low_rmse:<10.4f} {low_r2:<10.4f} {low_count:<10}")
    else:
        print(f"{view_idx + 1:<6} {'0-2':<10} {'N/A':<10} {'N/A':<10} {low_count:<10}")
    
    # For high score range (2-4)
    high_range_mask = y >= 2
    high_count = np.sum(high_range_mask)
    if high_count > 1:
        high_rmse = np.sqrt(mean_squared_error(y[high_range_mask], y_pred[high_range_mask]))
        high_r2 = r2_score(y[high_range_mask], y_pred[high_range_mask])
        print(f"{view_idx + 1:<6} {'2-4':<10} {high_rmse:<10.4f} {high_r2:<10.4f} {high_count:<10}")
    else:
        print(f"{view_idx + 1:<6} {'2-4':<10} {'N/A':<10} {'N/A':<10} {high_count:<10}")

# Full regression analysis for all views
plt.figure(figsize=(20, 15))
for view_idx in range(10):
    valid_data_indices = data_validity_mask[:, view_idx]
    gen_impression_view = generated_impression_data[valid_data_indices, view_idx]
    criteria_percentage_view = criteria_percentage_data[valid_data_indices, view_idx]
    
    # Prepare data for plotting
    X = criteria_percentage_view.reshape(-1, 1)
    y = gen_impression_view
    model = regression_models[view_idx]
    
    # Predictions
    y_pred = model.predict(X)
    
    # Plotting
    plt.subplot(4, 3, view_idx + 1)
    plt.scatter(criteria_percentage_view, gen_impression_view, color='slateblue', alpha=0.7, label='Actual')
    x_line = np.linspace(0, 100, 100).reshape(-1, 1)
    y_line = model.predict(x_line)
    plt.plot(x_line, y_line, color='tomato', lw=2, label='Regression Line')
    
    # Model stats
    rmse = math.sqrt(mean_squared_error(y, y_pred))
    r2 = r2_score(y, y_pred)
    plt.text(5, 3.5, 
             f"View {view_idx + 1}\n"
             f"RMSE = {rmse:.4f}\n"
             f"R² = {r2:.4f}\n"
             f"Combined Score = {combined_scores[view_idx]:.4f}", 
             bbox=dict(facecolor='white', alpha=0.8))
    
    plt.title(f"View {view_idx + 1}")
    plt.xlabel("Criteria Percentage")
    plt.ylabel("General Impression")
    plt.xlim(0, 100)
    plt.ylim(0, 4)
    plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('Q1_Full_Regression_Analysis_All_Views.png', dpi=300)
plt.show()
