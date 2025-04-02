
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.stats as ss
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
from sklearn.linear_model import LinearRegression
from scipy.spatial.distance import cdist
#import kneed  # Remove this import

def plot_relational_plot(df):
    """
    Plots mileage vs. price as a scatter plot.
    """
    # selects the resolution of the plot
    fig, ax = plt.subplots(dpi=144)
    # plots the scatterplot using seaborn
    sns.scatterplot(x=df['Mileage'], y=df['Price'], color='red')
    # formatting the x and y labels
    ax.set_xlabel('Mileage')
    ax.set_ylabel('Price')
    # title of the plot
    ax.set_title('Price vs Mileage')
    plt.savefig('relational_plot.png')
    plt.show()
    return

def plot_categorical_plot(df):
    """
    Plot the categorical distribution of car prices.
    """
    # sets the resolution of the plot
    fig, ax = plt.subplots(dpi=144)
    sns.histplot(df['Price'], bins=50, kde=True)
    # formatting
    ax.set_xlabel('Price')
    ax.set_ylabel('Frequency')
    # sets the title of the plot
    ax.set_title('Distribution of Car Prices')
    plt.savefig('statistical_plot.png')
    plt.show()
    return

def plot_statistical_plot(df):
    """
    Plot a categorical plot showing the distribution of car prices
    by fuel type.
    """
    # selects the resolution of the plot
    fig, ax = plt.subplots(dpi=144)
    # plots the boxplot with seaborn
    sns.boxplot(x=df['Fuel_Type'], y=df['Price'], hue=df['Fuel_Type'],
                palette='Set2', legend=False)
    # formatting the x and y label
    ax.set_xlabel('Fuel_Type')
    ax.set_ylabel('Price')
    # Sets the plot title
    ax.set_title('Distribution of Price by Fuel Type')
    plt.savefig('categorical_plot.png')
    plt.show()
    return

def plot_correlation_heatmap(df):
    """
    Plot a correlation heatmap to visualize correlation between features.
    """
    # Select only numeric columns
    numeric_df = df.select_dtypes(include=[np.number])
    
    # Calculate correlation matrix
    corr_matrix = numeric_df.corr()
    
    # Create heatmap
    plt.figure(figsize=(10, 8), dpi=144)
    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', vmin=-1, vmax=1, 
                linewidths=0.5, fmt='.2f')
    plt.title('Correlation Heatmap of Numeric Features')
    plt.tight_layout()
    plt.savefig('correlation_heatmap.png')
    plt.show()
    return

def statistical_analysis(df, col: str):
    """
    Compute statistical moments: mean, standard deviation, skewness,
    and excess kurtosis.
    """
    mean = df[col].mean()
    stddev = df[col].std()
    skew = ss.skew(df[col])
    excess_kurtosis = ss.kurtosis(df[col])
    return mean, stddev, skew, excess_kurtosis

def preprocessing(df):
    """
    Preprocess the data by checking for missing values,
    describing data, and checking correlation.
    """
    # basic summary statistics
    df.describe()
    numeric_df = df.select_dtypes(include=[np.number])
    numeric_df.corr()
    df.head()
    print('\n Basic Summary of the data\n', df.head())
    print('\n Data Description:\n', df.describe())
    # Select only numeric columns before calculating correlation
    # Compute correlation only on numeric data
    print('\nCorrelation:\n', numeric_df.corr())
    
    # Drop missing values as in the second code
    df = df.dropna()
    return df

def writing(moments, col):
    """
    Print statistical moments analysis.
    """
    print(f'For the attribute {col}:')
    print(f'Mean = {moments[0]:.2f}, '
          f'Standard Deviation = {moments[1]:.2f}, '
          f'Skewness = {moments[2]:.2f}, and '
          f'Excess Kurtosis = {moments[3]:.2f}.')
    
    # Determine skewness type
    skew_type = "not"
    if moments[2] > 0.5:
        skew_type = "right"
    elif moments[2] < -0.5:
        skew_type = "left"
    
    # Determine kurtosis type
    kurtosis_type = "mesokurtic"
    if moments[3] > 0.5:
        kurtosis_type = "leptokurtic"
    elif moments[3] < -0.5:
        kurtosis_type = "platykurtic"
    
    print(f'The data was {skew_type} skewed and {kurtosis_type}.')
    return

def find_optimal_k(scaled_data, k_range=range(2, 10)):
    """
    Find the optimal number of clusters using multiple methods:
    1. Elbow method using inertia
    2. Elbow method using distortion (average distance to centers)
    3. Silhouette scores
    
    Parameters:
    scaled_data (ndarray): Standardized data for clustering
    k_range (range): Range of k values to test
    
    Returns:
    tuple: Contains optimal k and metrics for each k value
    """
    inertias = []
    distortions = []
    silhouette_scores = []
    
    # For elbow method and silhouette scores
    for k in k_range:
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        kmeans.fit(scaled_data)
        
        # Inertia: Sum of squared distances to closest centroid
        inertias.append(kmeans.inertia_)
        
        # Distortion: Mean Euclidean distance to closest centroid
        distortion = np.mean(np.min(cdist(scaled_data, kmeans.cluster_centers_, 'euclidean'), axis=1))
        distortions.append(distortion)
        
        # Silhouette scores (only if k > 1)
        labels = kmeans.labels_
        if k > 1:  # Silhouette score requires at least 2 clusters
            score = silhouette_score(scaled_data, labels)
            silhouette_scores.append(score)
        else:
            silhouette_scores.append(0)  # Placeholder for k=1
    
    # Improved method for finding elbow point without kneed
    # Calculate the rate of change (first derivative)
    inertia_derivative = np.diff(inertias)
    # Calculate the second derivative (rate of change of the rate of change)
    inertia_second_derivative = np.diff(inertia_derivative)
    
    # Find the point where the second derivative is maximum (the elbow point)
    # Add 2 to account for the k_range starting at 2 and the double differentiation
    optimal_k_inertia = np.argmax(np.abs(inertia_second_derivative)) + 2 + 1
    
    # Find the optimal k using silhouette score (the highest score)
    optimal_k_silhouette = k_range[np.argmax(silhouette_scores)]
    
    # Calculate the rate of change in distortion
    distortion_derivative = np.diff(distortions)
    distortion_second_derivative = np.diff(distortion_derivative)
    optimal_k_distortion = np.argmax(np.abs(distortion_second_derivative)) + 2 + 1
    
    # Combine methods using weighted voting
    # Give higher weight to silhouette score as it's generally more reliable
    candidates = [optimal_k_inertia, optimal_k_distortion, optimal_k_silhouette]
    weights = [1, 1, 2]  # Higher weight for silhouette score
    
    # Count occurrences with weights
    from collections import Counter
    weighted_votes = Counter()
    for k, weight in zip(candidates, weights):
        weighted_votes[k] += weight
    
    # Get the k with highest weighted votes
    optimal_k = weighted_votes.most_common(1)[0][0]
    
    # If all methods disagree significantly, prefer silhouette score
    if len(weighted_votes) == 3 and max(weighted_votes.values()) == 2:
        optimal_k = optimal_k_silhouette
    
    return optimal_k, (list(k_range), inertias, distortions, silhouette_scores)

def perform_clustering(df, col1, col2):
    """
    Perform K-means clustering on two columns of the dataframe.
    Uses the elbow method to automatically determine the optimal number of clusters.
    
    Parameters:
    df (pandas.DataFrame): Input dataframe
    col1 (str): First column name for clustering
    col2 (str): Second column name for clustering
    
    Returns:
    tuple: Contains labels, standardized data, cluster centers (original scale),
           silhouette score, and inertia value
    """
    # Extract data for clustering
    data = df[[col1, col2]].values
    
    # Scaling
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(data)
    
    # Find optimal number of clusters using multiple methods
    best_k, metrics = find_optimal_k(scaled_data)
    k_range, inertias, distortions, silhouette_scores = metrics
    
    # Plot elbow method - Inertia
    plt.figure(dpi=144)
    plt.plot(k_range, inertias, marker='o')
    plt.xlabel('Number of Clusters (k)')
    plt.ylabel('Inertia')
    plt.title(f'Elbow Method (Inertia) - Optimal k = {best_k}')
    plt.axvline(x=best_k, color='r', linestyle='--', 
               label=f'Optimal k = {best_k}')
    plt.legend()
    plt.savefig('elbow_plot_inertia.png')
    plt.show()
    
    # Plot elbow method - Distortion
    plt.figure(dpi=144)
    plt.plot(k_range, distortions, marker='o')
    plt.xlabel('Number of Clusters (k)')
    plt.ylabel('Average Distance to Centroid')
    plt.title('Elbow Method (Distortion)')
    plt.axvline(x=best_k, color='r', linestyle='--', 
               label=f'Optimal k = {best_k}')
    plt.legend()
    plt.savefig('elbow_plot_distortion.png')
    plt.show()
    
    # Plot silhouette scores
    plt.figure(dpi=144)
    plt.plot(k_range, silhouette_scores, marker='o')
    plt.xlabel('Number of Clusters (k)')
    plt.ylabel('Silhouette Score')
    plt.title('Silhouette Score Method')
    plt.axvline(x=best_k, color='r', linestyle='--', 
               label=f'Selected k = {best_k}')
    plt.legend()
    plt.savefig('silhouette_scores.png')
    plt.show()
    
    # Apply k-means with the optimal number of clusters
    kmeans = KMeans(n_clusters=best_k, random_state=42, n_init=10)
    labels = kmeans.fit_predict(scaled_data)
    
    # Get cluster centers and inertia
    scaled_centers = kmeans.cluster_centers_
    inertia = kmeans.inertia_
    
    # Inverse transform to get centers in original scale
    original_centers = scaler.inverse_transform(scaled_centers)
    
    # Calculate silhouette score
    score = silhouette_score(scaled_data, labels)
    
    print(f"Clustering Results:")
    print(f"Optimal number of clusters: {best_k}")
    print(f"Silhouette score: {score:.4f}")
    print(f"Inertia: {inertia:.4f}")
    print("Cluster centers (original scale):")
    for i, center in enumerate(original_centers):
        print(f"  Cluster {i+1}: {col1}={center[0]:.2f}, {col2}={center[1]:.2f}")
    
    return labels, scaled_data, original_centers, score, inertia, best_k

def plot_clustered_data(labels, scaled_data, centers, score, col1, col2, best_k):
    """
    Plot the clustered data along with cluster centers.
    
    Parameters:
    labels (ndarray): Cluster labels for each data point
    scaled_data (ndarray): Standardized data used for clustering
    centers (ndarray): Cluster centers in original scale
    score (float): Silhouette score
    col1 (str): Name of x-axis column
    col2 (str): Name of y-axis column
    best_k (int): Optimal number of clusters
    """
    # Create figure for scaled data visualization
    plt.figure(dpi=144, figsize=(10, 6))
    
    # Get a color map with distinct colors for each cluster
    colors = plt.cm.viridis(np.linspace(0, 1, len(np.unique(labels))))
    
    # Plot each cluster with its own color
    for i, cluster_id in enumerate(np.unique(labels)):
        cluster_data = scaled_data[labels == cluster_id]
        plt.scatter(cluster_data[:, 0], cluster_data[:, 1], 
                   color=colors[i], label=f'Cluster {cluster_id+1}',
                   alpha=0.7)
    
    # Add cluster centers
    for i in range(best_k):
        plt.scatter(scaled_data[labels == i, 0].mean(), 
                   scaled_data[labels == i, 1].mean(), 
                   color='red', marker='X', s=100, 
                   edgecolor='black', linewidth=1,
                   label='Cluster centers' if i == 0 else "")
        plt.annotate(f'Cluster {i+1}', 
                     xy=(scaled_data[labels == i, 0].mean(), 
                         scaled_data[labels == i, 1].mean()),
                     xytext=(5, 5), textcoords='offset points',
                     fontweight='bold')
    
    plt.title(f'K-Means Clustering Results (k={best_k})\nSilhouette Score: {score:.4f}')
    plt.xlabel(f'{col1} (standardized)')
    plt.ylabel(f'{col2} (standardized)')
    plt.legend(loc='upper right')
    plt.grid(alpha=0.3)
    plt.savefig('clustering.png')
    plt.show()
    
    # Create silhouette plot to visualize cluster quality
    from matplotlib import cm
    
    plt.figure(figsize=(7, 5), dpi=144)
    
    # Create a subplot with 1 row and 2 columns
    from sklearn.metrics import silhouette_samples
    
    # Compute the silhouette scores for each sample
    silhouette_vals = silhouette_samples(scaled_data, labels)
    
    y_ticks = []
    y_lower, y_upper = 0, 0
    
    for i, cluster in enumerate(np.unique(labels)):
        # Aggregate the silhouette scores for samples belonging to cluster i
        cluster_silhouette_vals = silhouette_vals[labels == cluster]
        cluster_silhouette_vals.sort()
        
        y_upper += len(cluster_silhouette_vals)
        
        plt.barh(range(y_lower, y_upper), cluster_silhouette_vals, 
                height=1.0, edgecolor='none')
        
        # Add horizontal line for average silhouette score of all values
        plt.axvline(x=score, color='red', linestyle='--')
        
        # Compute average silhouette score for the current cluster
        avg_score = np.mean(cluster_silhouette_vals)
        plt.text(-0.05, (y_lower + y_upper) / 2, f'{cluster+1}')
        
        # Compute the new y_lower for next plot
        y_lower += len(cluster_silhouette_vals)
        
    
    plt.xlabel("Silhouette coefficient values")
    plt.ylabel("Cluster label")
    plt.title(f"Silhouette Plot for {best_k} Clusters")
    plt.axvline(x=0, color='gray', linestyle='--')  # Add vertical line at x=0
    plt.tight_layout()
    plt.savefig('silhouette_plot.png')
    plt.show()
    
    return

def perform_fitting(df, col1, col2):
    """
    Perform linear regression fitting between two columns with scaling.
    
    Parameters:
    df (pandas.DataFrame): Input dataframe
    col1 (str): Feature column name (X)
    col2 (str): Target column name (y)
    
    Returns:
    tuple: Contains original X, original y, predicted y, R² score, 
           coefficient, intercept, scaler_X, and scaler_y
    """
    # Prepare data
    X_original = df[[col1]].values
    y_original = df[col2].values
    
    # Scale X and y for better numerical stability
    scaler_X = StandardScaler()
    scaler_y = StandardScaler()
    
    X_scaled = scaler_X.fit_transform(X_original)
    y_scaled = scaler_y.fit_transform(y_original.reshape(-1, 1)).ravel()
    
    # Fit linear regression model on scaled data
    model = LinearRegression()
    model.fit(X_scaled, y_scaled)
    
    # Get predictions in scaled space
    y_pred_scaled = model.predict(X_scaled)
    
    # Transform predictions back to original scale
    y_pred = scaler_y.inverse_transform(y_pred_scaled.reshape(-1, 1)).ravel()
    
    # Calculate R² score on original data
    r2 = model.score(X_scaled, y_scaled)
    
    # Get scaled coefficient and intercept
    scaled_coef = model.coef_[0]
    scaled_intercept = model.intercept_
    
    # Convert coefficient and intercept to original scale
    # For a simple linear regression: y = mx + b
    # After scaling: (y-μy)/σy = m' * (x-μx)/σx + b'
    # So: y = (σy/σx * m') * x + (b'*σy + μy - m'*σy*μx/σx)
    std_y = scaler_y.scale_[0]
    std_x = scaler_X.scale_[0]
    mean_y = scaler_y.mean_[0]
    mean_x = scaler_X.mean_[0]
    
    coef = scaled_coef * (std_y / std_x)
    intercept = scaled_intercept * std_y + mean_y - scaled_coef * mean_x * (std_y / std_x)
    
    print(f"Linear Regression Results:")
    print(f"Scaled equation: y_scaled = {scaled_coef:.4f} * x_scaled + {scaled_intercept:.4f}")
    print(f"Original equation: {col2} = {coef:.4f} * {col1} + {intercept:.4f}")
    print(f"R² Score: {r2:.4f}")
    
    return X_original, y_original, y_pred, r2, coef, intercept, scaler_X, scaler_y

def plot_fitted_data(X, y, y_pred, r2, col1, col2, scaler_X, scaler_y):
    """
    Plot the original data points and the fitted regression line.
    
    Parameters:
    X (ndarray): Original X values
    y (ndarray): Original y values
    y_pred (ndarray): Predicted y values
    r2 (float): R² score
    col1 (str): X-axis column name
    col2 (str): Y-axis column name
    scaler_X (StandardScaler): Scaler used for X
    scaler_y (StandardScaler): Scaler used for y
    """
    # Create figure for original scale plot
    plt.figure(dpi=144, figsize=(10, 6))
    plt.scatter(X, y, label='Data', alpha=0.6)
    
    # Sort X and y_pred for proper line plotting
    sorted_indices = np.argsort(X.ravel())
    X_sorted = X[sorted_indices]
    y_pred_sorted = y_pred[sorted_indices]
    
    plt.plot(X_sorted, y_pred_sorted, color='red', linewidth=2, 
            label=f'Fitted Line (R² = {r2:.4f})')
    
    plt.xlabel(col1)
    plt.ylabel(col2)
    plt.title(f'Linear Regression: {col2} vs {col1}')
    plt.legend()
    plt.grid(alpha=0.3)
    plt.savefig('fitting.png')
    plt.show()
    
    # Create figure for scaled data plot
    plt.figure(dpi=144, figsize=(10, 6))
    X_scaled = scaler_X.transform(X)
    y_scaled = scaler_y.transform(y.reshape(-1, 1)).ravel()
    y_pred_scaled = scaler_y.transform(y_pred.reshape(-1, 1)).ravel()
    
    plt.scatter(X_scaled, y_scaled, label='Scaled Data', alpha=0.6)
    
    # Sort scaled X and y_pred for proper line plotting
    sorted_indices = np.argsort(X_scaled.ravel())
    X_scaled_sorted = X_scaled[sorted_indices]
    y_pred_scaled_sorted = y_pred_scaled[sorted_indices]
    
    plt.plot(X_scaled_sorted, y_pred_scaled_sorted, color='red', linewidth=2, 
            label=f'Fitted Line (R² = {r2:.4f})')
    
    plt.xlabel(f'{col1} (standardized)')
    plt.ylabel(f'{col2} (standardized)')
    plt.title(f'Linear Regression on Standardized Data')
    plt.legend()
    plt.grid(alpha=0.3)
    plt.savefig('fitting_scaled.png')
    plt.show()
    
    # Create diagnostic plots for regression
    plt.figure(figsize=(12, 10), dpi=144)
    
    # Residuals plot
    plt.subplot(2, 2, 1)
    residuals = y - y_pred
    plt.scatter(X, residuals, alpha=0.6)
    plt.axhline(y=0, color='r', linestyle='-')
    plt.xlabel(col1)
    plt.ylabel('Residuals')
    plt.title('Residuals vs Input Feature')
    plt.grid(alpha=0.3)
    
    # Histogram of residuals
    plt.subplot(2, 2, 2)
    plt.hist(residuals, bins=20, edgecolor='black')
    plt.xlabel('Residual Value')
    plt.ylabel('Frequency')
    plt.title('Histogram of Residuals')
    plt.grid(alpha=0.3)
    
    # Q-Q plot of residuals
    plt.subplot(2, 2, 3)
    from scipy import stats
    stats.probplot(residuals, dist="norm", plot=plt)
    plt.title('Q-Q Plot of Residuals')
    plt.grid(alpha=0.3)
    
    # Actual vs Predicted
    plt.subplot(2, 2, 4)
    plt.scatter(y, y_pred, alpha=0.6)
    plt.plot([y.min(), y.max()], [y.min(), y.max()], 'r--')
    plt.xlabel('Actual Values')
    plt.ylabel('Predicted Values')
    plt.title('Actual vs Predicted Values')
    plt.grid(alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('regression_diagnostics.png')
    plt.show()
    
    return

def main():
    """
    Function to process, analyze data set and plots charts
    """
    # import and read data
    df = pd.read_csv('data.csv')
    df = preprocessing(df)
    col = 'Price'
    
    # calls functions from first code
    plot_relational_plot(df)
    plot_categorical_plot(df)
    plot_statistical_plot(df)
    # Add correlation heatmap
    plot_correlation_heatmap(df)
    moments = statistical_analysis(df, col)
    writing(moments, col)
    
    # call additional functions from second code
    print("\n--- Performing Clustering Analysis ---")
    col1, col2 = 'Mileage', 'Price'
    labels, scaled_data, centers, score, inertia, best_k = perform_clustering(df, col1, col2)
    plot_clustered_data(labels, scaled_data, centers, score, col1, col2, best_k)
    
    print("\n--- Performing Linear Regression ---")
    fitting_results = perform_fitting(df, col1, col2)
    X, y, y_pred, r2, coef, intercept, scaler_X, scaler_y = fitting_results
    plot_fitted_data(X, y, y_pred, r2, col1, col2, scaler_X, scaler_y)
    
    return

if __name__ == '__main__':
    main()