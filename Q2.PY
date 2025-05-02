import scipy.io as sio
import numpy as np
import matplotlib.pyplot as plt
from skimage.metrics import structural_similarity
from scipy import stats
from sklearn.metrics.pairwise import cosine_similarity

# Load the dataset (make sure the path to 'cw2.mat' is correct)
data = sio.loadmat('cw2.mat')

# Extract the necessary data from the loaded file
test_images = data['test_img']
gold_images = data['gold_img']
generated_images = data['gen_impr']
critical_percentage = data['crit_perc']

# Locations of missing image data
missing_data = [(8, 9), (12, 7), (13, 9), (14, 0), (15, 3)]

# Create a mask to handle missing data
valid_mask = np.ones((20, 10), dtype=bool)
for i, j in missing_data:
    valid_mask[i, j] = False

print("Data loaded successfully")

# Helper function to compute mutual information
def calculate_mutual_information(img1, img2, bins=256):
    hist_2d, _, _ = np.histogram2d(img1.flatten(), img2.flatten(), bins=bins)
    pxy = hist_2d / np.sum(hist_2d)
    px = np.sum(pxy, axis=1)
    py = np.sum(pxy, axis=0)
    
    # Entropy of each image
    entropy_x = -np.sum(px * np.log(px + np.finfo(float).eps))
    entropy_y = -np.sum(py * np.log(py + np.finfo(float).eps))
    
    # Joint entropy
    joint_entropy = -np.sum(pxy * np.log(pxy + np.finfo(float).eps))
    
    return entropy_x + entropy_y - joint_entropy

# Function to compute cosine similarity between two images
def cosine_sim(image1, image2):
    flattened1 = image1.flatten().reshape(1, -1)
    flattened2 = image2.flatten().reshape(1, -1)
    return cosine_similarity(flattened1, flattened2)[0][0]

# Metric calculations and analysis
ssi_scores = np.zeros((20, 10))
mi_scores = np.zeros((20, 10))
cs_scores = np.zeros((20, 10))

for row in range(20):
    for col in range(10):
        if valid_mask[row, col]:
            test_image = test_images[row][col]
            gold_standard_image = gold_images[0][col]
            
            # Calculate Structural Similarity Index (SSI)
            ssi_scores[row, col] = structural_similarity(test_image, gold_standard_image)
            
            # Calculate Mutual Information (MI)
            mi_scores[row, col] = calculate_mutual_information(test_image, gold_standard_image)
            
            # Calculate Cosine Similarity (CS)
            cs_scores[row, col] = cosine_sim(test_image, gold_standard_image)
        else:
            ssi_scores[row, col] = np.nan
            mi_scores[row, col] = np.nan
            cs_scores[row, col] = np.nan

# Extract the top 3 participants for each view
def top_three_participants(similarity_scores, view):
    valid_scores = similarity_scores[:, view]
    valid_indices = ~np.isnan(valid_scores)
    
    # Sorting and selecting top 3
    top_indices = np.argsort(valid_scores[valid_indices])[-3:][::-1]
    return np.where(valid_indices)[0][top_indices], valid_scores[valid_indices][top_indices]

# Reporting the top 3 participants for each view for SSI, MI, and CS
def print_top3(metric_scores, metric_name):
    print(f"\nTop 3 Participants by {metric_name}:")
    print(f"{'View':<5} {'Top 1':<15} {'Top 2':<15} {'Top 3':<15}")
    print("-" * 50)
    
    for view in range(10):
        participants, values = top_three_participants(metric_scores, view)
        print(f"{view+1:<5} {participants[0]+1:<5} ({values[0]:.4f}) {participants[1]+1:<5} ({values[1]:.4f}) {participants[2]+1:<5} ({values[2]:.4f})")

# Display the top 3 participants for SSI, MI, and CS
print_top3(ssi_scores, 'SSI')
print_top3(mi_scores, 'MI')
print_top3(cs_scores, 'CS')

# Normalize the similarity values
def normalize_metrics(metrics):
    min_val = np.nanmin(metrics)
    max_val = np.nanmax(metrics)
    return (metrics - min_val) / (max_val - min_val)

normalized_ssi = normalize_metrics(ssi_scores)
normalized_mi = normalize_metrics(mi_scores)
normalized_cs = normalize_metrics(cs_scores)

# Composite score calculation with normalized values
def compute_composite_score(ssi, mi, cs, ssi_weight=0.4, mi_weight=0.3, cs_weight=0.3):
    return ssi_weight * ssi + mi_weight * mi + cs_weight * cs

composite_scores = np.zeros((20, 10))
for row in range(20):
    for col in range(10):
        if valid_mask[row, col]:
            composite_scores[row, col] = compute_composite_score(
                normalized_ssi[row, col], normalized_mi[row, col], normalized_cs[row, col]
            )

# Reporting the top 3 participants based on composite score
print_top3(composite_scores, 'Composite Score')

# Expert vs Novice analysis
def expert_vs_novice_comparison(similarity_scores, view, group_indices):
    expert_vals = [similarity_scores[i, view] for i in group_indices['expert'] if not np.isnan(similarity_scores[i, view])]
    novice_vals = [similarity_scores[i, view] for i in group_indices['novice'] if not np.isnan(similarity_scores[i, view])]
    stat, p_val = stats.mannwhitneyu(expert_vals, novice_vals, alternative='greater')
    return stat, p_val, np.mean(expert_vals), np.mean(novice_vals)

group_indices = {
    'expert': list(range(7)),
    'novice': list(range(7, 20))
}

# Statistical analysis for each view and metric
print("\nStatistical Test Results (Mann-Whitney U Test, Expert > Novice):")
print(f"{'View':<5} {'Metric':<8} {'Expert Mean':<15} {'Novice Mean':<15} {'p-value':<10} {'Significant':<10}")
print("-" * 65)

for view in range(10):
    for metric, metric_scores in zip(['SSI', 'MI', 'CS'], [ssi_scores, mi_scores, cs_scores]):
        stat, p_value, expert_mean, novice_mean = expert_vs_novice_comparison(metric_scores, view, group_indices)
        significant = "Yes" if p_value < 0.05 else "No"
        print(f"{view+1:<5} {metric:<8} {expert_mean:<15.4f} {novice_mean:<15.4f} {p_value:<10.4f} {significant:<10}")

# Boxplots for expert vs novice group comparison
plt.figure(figsize=(20, 15))
for view in range(10):
    plt.subplot(4, 3, view+1)
    
    # Prepare data for boxplot comparison
    expert_ssi = [ssi_scores[i, view] for i in group_indices['expert'] if not np.isnan(ssi_scores[i, view])]
    novice_ssi = [ssi_scores[i, view] for i in group_indices['novice'] if not np.isnan(ssi_scores[i, view])]
    
    box_data = [expert_ssi, novice_ssi]
    bp = plt.boxplot(box_data, positions=[1, 2], patch_artist=True)
    
    # Customizing plot colors
    bp['boxes'][0].set_facecolor('lightgreen')
    bp['boxes'][1].set_facecolor('lightpink')
    
    # Add titles and labels
    plt.xticks([1, 2], ['Expert', 'Novice'])
    plt.title(f'View {view+1} - SSI Comparison')
    plt.ylabel('SSI Value')
    
    # Annotate p-value
    _, p_value, _, _ = expert_vs_novice_comparison(ssi_scores, view, group_indices)
    plt.annotate(f'p = {p_value:.4f}', xy=(1.5, min(min(expert_ssi), min(novice_ssi))), 
                 ha='center', va='center', bbox=dict(boxstyle='round', fc='white', alpha=0.7))

plt.tight_layout()
plt.savefig('comparison_boxplots.png', dpi=300)
plt.show()
