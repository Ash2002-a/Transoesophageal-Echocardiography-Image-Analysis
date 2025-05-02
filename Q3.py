import scipy.io as sio
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from skimage.metrics import structural_similarity
from scipy import stats
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.linear_model import Ridge, ElasticNet
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.pipeline import Pipeline
import warnings

warnings.filterwarnings('ignore')

# Load dataset
print("Loading data...")
data = sio.loadmat('cw2.mat')

# Extract necessary data
test_images = data['test_img']
gold_images = data['gold_img']
gen_impr = data['gen_impr']
crit_perc = data['crit_perc']

# Missing data locations
missing_data = [(8, 9), (12, 7), (13, 9), (14, 0), (15, 3)]
valid_mask = np.ones((20, 10), dtype=bool)
for i, j in missing_data:
    valid_mask[i, j] = False

# Function to compute Structural Similarity Index (SSI)
def calculate_mutual_information(img1, img2, bins=256):
    hist_2d, _, _ = np.histogram2d(img1.flatten(), img2.flatten(), bins=bins)
    pxy = hist_2d / np.sum(hist_2d)
    px = np.sum(pxy, axis=1)
    py = np.sum(pxy, axis=0)
    entropy_x = -np.sum(px * np.log(px + np.finfo(float).eps))
    entropy_y = -np.sum(py * np.log(py + np.finfo(float).eps))
    joint_entropy = -np.sum(pxy * np.log(pxy + np.finfo(float).eps))
    return entropy_x + entropy_y - joint_entropy

def cosine_sim(image1, image2):
    flattened1 = image1.flatten().reshape(1, -1)
    flattened2 = image2.flatten().reshape(1, -1)
    return cosine_similarity(flattened1, flattened2)[0][0]

# Compute similarity metrics
ssi_scores = np.full((20, 10), np.nan)
mi_scores = np.full((20, 10), np.nan)
cs_scores = np.full((20, 10), np.nan)

print("Computing similarity metrics...")
for row in range(20):
    for col in range(10):
        if valid_mask[row, col]:
            test_image = test_images[row][col]
            gold_image = gold_images[0][col]
            ssi_scores[row, col] = structural_similarity(test_image, gold_image)
            mi_scores[row, col] = calculate_mutual_information(test_image, gold_image)
            cs_scores[row, col] = cosine_sim(test_image, gold_image)

# Compute correlation coefficients
corr_results = []
for view in range(10):
    valid_indices = valid_mask[:, view]
    ssi_view, mi_view, cs_view = ssi_scores[valid_indices, view], mi_scores[valid_indices, view], cs_scores[valid_indices, view]
    
    corr_ssi_mi, _ = stats.pearsonr(ssi_view, mi_view)
    corr_ssi_cs, _ = stats.pearsonr(ssi_view, cs_view)
    corr_mi_cs, _ = stats.pearsonr(mi_view, cs_view)
    
    corr_results.append((view+1, corr_ssi_mi, corr_ssi_cs, corr_mi_cs))

# Print correlation results
print("\nCorrelation Coefficients for each View:")
print("View | SSI-MI | SSI-CS | MI-CS")
for view, ssi_mi, ssi_cs, mi_cs in corr_results:
    print(f"{view:4d} | {ssi_mi:.3f} | {ssi_cs:.3f} | {mi_cs:.3f}")

# Polynomial Regression with Regularization
def find_optimal_params(X, y, max_degree=7, cv=5):
    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('poly', PolynomialFeatures()),
        ('model', ElasticNet())
    ])
    
    param_grid = {
        'poly__degree': range(1, max_degree + 1),
        'model__alpha': [0.001, 0.01, 0.1, 1.0, 10.0],
        'model__l1_ratio': [0.1, 0.5, 0.7, 0.9, 0.95, 0.99, 1.0]
    }
    
    grid_search = GridSearchCV(pipeline, param_grid, cv=cv, scoring='neg_mean_squared_error')
    grid_search.fit(X, y)
    
    return grid_search.best_params_

best_models = {}
for view in range(10):
    valid_indices = valid_mask[:, view]
    for metric_name, metric_scores in {'SSI': ssi_scores, 'MI': mi_scores, 'CS': cs_scores}.items():
        for target_name, target_scores in {'General Impression': gen_impr, 'Criteria Percentage': crit_perc}.items():
            X = metric_scores[valid_indices, view].reshape(-1, 1)
            y = target_scores[valid_indices, view]
            best_params = find_optimal_params(X, y)
            optimal_degree, alpha, l1_ratio = best_params['poly__degree'], best_params['model__alpha'], best_params['model__l1_ratio']

            pipeline = Pipeline([
                ('scaler', StandardScaler()),
                ('poly', PolynomialFeatures(degree=optimal_degree)),
                ('model', ElasticNet(alpha=alpha, l1_ratio=l1_ratio))
            ])
            
            pipeline.fit(X, y)
            y_pred = pipeline.predict(X)
            rmse, r2 = np.sqrt(mean_squared_error(y, y_pred)), r2_score(y, y_pred)
            
            model_key = f"{metric_name}_{target_name}_View{view+1}"
            best_models[model_key] = {'rmse': rmse, 'r2': r2, 'X': X, 'y': y, 'y_pred': y_pred}

# Select the three best-performing models
top_models = sorted(best_models.items(), key=lambda x: x[1]['rmse'])[:3]

# Plot the best regression models
for model_name, model_data in top_models:
    plt.figure(figsize=(6, 4))
    plt.scatter(model_data['X'], model_data['y'], label="Actual")
    plt.plot(model_data['X'], model_data['y_pred'], color='red', label="Predicted")
    plt.title(f"Polynomial Regression - {model_name}")
    plt.xlabel("Metric Score")
    plt.ylabel("Target Score")
    plt.legend()
    plt.show()

# Gaussian Basis Regression
def gaussian_basis(x, mu, sigma):
    return np.exp(-(x - mu)**2 / (2 * sigma**2))

def create_gaussian_features(X, n_basis, sigma=None):
    sigma = (X.max() - X.min()) / (2 * n_basis) if sigma is None else sigma
    centers = np.linspace(X.min(), X.max(), n_basis)
    return np.array([gaussian_basis(X, mu, sigma) for mu in centers]).T

best_gaussian_models = {}
for view in range(10):
    valid_indices = valid_mask[:, view]
    X, y = ssi_scores[valid_indices, view], gen_impr[valid_indices, view]
    X_gauss = create_gaussian_features(X, n_basis=10)
    
    model = Ridge(alpha=0.1)
    model.fit(X_gauss, y)
    y_pred = model.predict(X_gauss)
    
    rmse, r2 = np.sqrt(mean_squared_error(y, y_pred)), r2_score(y, y_pred)
    best_gaussian_models[view+1] = {'rmse': rmse, 'r2': r2, 'X': X, 'y': y, 'y_pred': y_pred}

# Print Gaussian Regression Results
print("\nGaussian Basis Regression RMSE & R2 Scores:")
for view, model_data in best_gaussian_models.items():
    print(f"View {view}: RMSE={model_data['rmse']:.3f}, R²={model_data['r2']:.3f}")
