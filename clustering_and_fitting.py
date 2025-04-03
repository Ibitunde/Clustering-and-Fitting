"""
This is the template file for the clustering and fitting assignment.
You will be expected to complete all the sections and
make this a fully working, documented file.
You should NOT change any function, file or variable names,
 if they are given to you here.
Make use of the functions presented in the lectures
and ensure your code is PEP-8 compliant, including docstrings.
Fitting should be done with only 1 target variable and 1 feature variable,
likewise, clustering should be done with only 2 variables.
"""
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.stats as ss
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from matplotlib.colors import ListedColormap
from numpy.polynomial import Polynomial as Poly


def plot_relational_plot(df):
    """
    Create a scatter plot to visualize the relationship between 
    petal length and width, colored by iris species.
    
    Args:
        df (pandas.DataFrame): DataFrame containing iris dataset.
    """
    # Create figure with higher resolution
    fig, ax = plt.subplots(dpi=144)
    
    # Set a different color palette
    sns.set_palette("viridis")
    
    # Generate scatter plot with species-based coloring
    sns.scatterplot(data=df, x='petal_length', y='petal_width', 
                    hue='species', ax=ax)
    
    # Add legend and format the plot
    ax.legend(title='Species', loc='upper left')
    ax.set_xlabel(df.columns[2])
    ax.set_ylabel(df.columns[3])
    ax.set_title("Relationship between Petal Length and Width")
    
    # Add grid
    ax.grid(True, linestyle='--', alpha=0.6)
    
    # Save and display the visualization
    plt.savefig('relational_plot.png')
    plt.show()
    return


def plot_categorical_plot(df):
    """
    Generate a pie chart showing the proportional distribution
    of iris species in the dataset.
    
    Args:
        df (pandas.DataFrame): DataFrame containing iris dataset.
    """
    # Calculate species frequency
    species_distribution = df.groupby('species').size()
    # Set up visualization environment
    fig, ax = plt.subplots(dpi=144)
    # Create pie chart
    species_distribution.plot(
        ax=ax, kind='pie', autopct='%1.1f%%', startangle=200,
        colors=['cornflowerblue', 'lightcoral', 'mediumseagreen']  # Updated color scheme
    )
    # Format chart appearance
    ax.set_title("Iris Species Distribution")
    ax.set_ylabel('')
    ax.axis('equal')
    # Save and display the chart
    plt.savefig('categorical_plot.png')
    plt.show()
    return


def plot_statistical_plot(df):
    """
    Generate a boxplot for the numerical features in the dataset.

    Args:
        df (pandas.DataFrame): DataFrame containing the dataset.
    """
    # Initialize plot area
    fig, ax = plt.subplots(dpi=150, figsize=(10, 6))

    # Create boxplot
    sns.boxplot(data=df, ax=ax, palette='plasma')

    # Format visualization
    ax.set_title('Feature Distribution - Boxplot', fontsize=12, pad=12)
    ax.set_ylabel('Value')
    ax.set_xlabel('Features')
    plt.xticks(rotation=45)

    # Save and display the boxplot
    plt.savefig('statistical_plot.png')
    plt.show()
    return
    

def statistical_analysis(df, col: str):
    """
    Calculate fundamental statistical measures for a specified column.
    
    Args:
        df (pandas.DataFrame): DataFrame containing the dataset.
        col (str): Name of the column to analyze.
        
    Returns:
        tuple: Contains mean, standard deviation, skewness, and excess 
        kurtosis.
    """
    # Compute central tendency
    avg = df[col].mean()
    # Compute dispersion
    std = df[col].std()
    # Compute distribution shape parameters
    asymmetry = ss.skew(df[col], nan_policy='omit')
    peakedness = ss.kurtosis(df[col], nan_policy='omit')
    return avg, std, asymmetry, peakedness


def preprocessing(df):
    """
    Clean and prepare the dataset by removing duplicates and missing values,
    then display summary information about the data.
    
    Args:
        df (pandas.DataFrame): Raw input DataFrame.
        
    Returns:
        pandas.DataFrame: Cleaned DataFrame.
    """
    # Remove duplicate entries
    cleaned_df = df.drop_duplicates()
    # Handle missing values
    cleaned_df = cleaned_df.dropna()
    # Display dataset overview
    print("Statistical summary:\n", cleaned_df.describe())
    print("\nSample data (first 5 rows):\n", cleaned_df.head())
    print("\nFeature correlations:\n", cleaned_df.corr(numeric_only=True))
    return cleaned_df


def writing(moments, col):
    """
    Provides written interpretation of statistical measures for a given attribute.
    
    Args:
        moments (list): Contains [mean, standard deviation, skewness, excess kurtosis].
        col (str): Name of the attribute being analyzed.
    """
    print(f'Analysis of {col}:')
    print(f'Mean = {moments[0]:.2f}, '
          f'Standard Deviation = {moments[1]:.2f}, '
          f'Skewness = {moments[2]:.2f}, and '
          f'Excess Kurtosis = {moments[3]:.2f}.')
    # Interpret skewness
    if moments[2] < -2:
        skew_interpretation = "left-skewed"
    elif moments[2] > 2:
        skew_interpretation = "right-skewed"
    else:
        skew_interpretation = "not skewed" 
    # Interpret kurtosis
    if moments[3] < -1:
        kurtosis_interpretation = "platykurtic"
    elif moments[3] > 1:
        kurtosis_interpretation = "leptokurtic"
    else:
        kurtosis_interpretation = "mesokurtic"
    print(f'The distribution is {skew_interpretation}and {kurtosis_interpretation}.')
    return


def perform_clustering(df, col1, col2):
    """
    Apply KMeans clustering on two selected features, determine the optimal
    number of clusters using silhouette score, and visualize the results.
    
    Args:
        df (DataFrame): Input data.
        col1 (str): First feature name.
        col2 (str): Second feature name.
        
    Returns:
        tuple: Cluster assignments, original data, cluster centers coordinates,
               and center labels.
    """

    
    def plot_elbow_method():
        """
        Create an elbow plot to visualize how WCSS (Within-Cluster Sum of Squares)
        changes with different numbers of clusters, highlighting the optimal choice.
        """
        k_values = list(range(2, 11))
        
        # Setup visualization
        fig, ax = plt.subplots(dpi=144, figsize=(8, 5))
        
        # Plot WCSS trend with deep blue
        ax.plot(k_values, inertia_values, marker='o', color='navy', label="WCSS")

        # Mark optimal k value with teal
        ax.axvline(x=optimal_k, color='teal', linestyle='--', 
                  label=f'Optimal k = {optimal_k}')
        
        # Format plot
        ax.set_title('Elbow Method for Optimal Clusters')
        ax.set_xlabel('Number of Clusters')
        ax.set_ylabel('WCSS (Inertia)')
        ax.legend()
        ax.grid(True, linestyle='--', alpha=0.6)  # Added dashed grid for clarity
        
        # Save and display the plot
        plt.savefig('elbow_plot.png')
        fig.tight_layout()
        plt.show()
        return


    def one_silhouette_inertia():
        """
        Compute evaluation metrics for a K-means clustering with n clusters.
        
        Returns:
            tuple: Silhouette score and inertia value for current clustering.
        """
        kmeans_model = KMeans(n_clusters=n, n_init=20)
        kmeans_model.fit(normalized_data)
        sil_score = silhouette_score(normalized_data, kmeans_model.labels_)
        inertia_val = kmeans_model.inertia_
        return sil_score, inertia_val


    # Extract and normalize features
    feature_subset = df[[col1, col2]].copy()
    scaler = StandardScaler()
    normalized_data = scaler.fit_transform(feature_subset)
    original_scale_data = scaler.inverse_transform(normalized_data)

    # Evaluate different cluster counts
    inertia_values = []
    optimal_k, best_score = None, -np.inf
    
    for n in range(2, 11):
        current_score, current_inertia = one_silhouette_inertia()
        inertia_values.append(current_inertia)
        print(f"{n} clusters silhouette score = {current_score:.2f}")
        
        if current_score > best_score:
            optimal_k = n
            best_score = current_score

    # Report best clustering configuration
    print(f"\nOptimal number of clusters = {optimal_k} with silhouette score = {best_score:.2f}")
    plot_elbow_method()

    # Perform final clustering with optimal parameters
    final_model = KMeans(n_clusters=optimal_k, n_init=20, random_state=42)
    final_model.fit(normalized_data)
    
    cluster_labels = final_model.labels_
    centers = scaler.inverse_transform(final_model.cluster_centers_)
    center_x = centers[:, 0]
    center_y = centers[:, 1]
    center_assignments = final_model.predict(final_model.cluster_centers_)
    
    return cluster_labels, original_scale_data, center_x, center_y, center_assignments



def plot_clustered_data(labels, data, xkmeans, ykmeans, centre_labels):
    """
    Visualize clustering results showing data points and cluster centers.
    
    Args:
        labels (array): Cluster assignment for each data point.
        data (array): Feature values in original scale.
        xkmeans (array): X-coordinates of cluster centers.
        ykmeans (array): Y-coordinates of cluster centers.
        centre_labels (array): Cluster assignments for the centers.
    """
    # Setup visualization
    fig, ax = plt.subplots(dpi=144)
    
    # Create color palette based on number of clusters (using viridis)
    palette = plt.cm.viridis(np.linspace(0, 1, len(np.unique(labels))))
    color_map = ListedColormap(palette)
    
    # Plot data points with cluster-based coloring
    scatter_plot = ax.scatter(data[:, 0], data[:, 1], c=labels, 
                             cmap=color_map, marker='o', label='Data')
    
    # Add cluster centers
    ax.scatter(xkmeans, ykmeans, c=centre_labels, cmap=color_map, 
              marker='x', s=100, label='Centroids', edgecolors='black')
    
    # Add color legend
    color_bar = fig.colorbar(scatter_plot, ax=ax)
    color_bar.set_ticks(np.unique(labels))
    
    # Format plot
    ax.legend()
    ax.set_xlabel('Petal Length')
    ax.set_ylabel('Sepal Width')
    ax.grid(True, linestyle='--', alpha=0.6)  # Added grid
    
    # Save and display visualization
    plt.savefig('clustering.png')
    plt.show()
    return



def perform_fitting(df, col1, col2):
    """
    Perform linear regression between two columns using numpy polynomial tools.

    Args:
        df (DataFrame): Input data.
        col1 (str): Independent variable column name.
        col2 (str): Dependent variable column name.
        Returns:
        tuple: Model parameters and fit data, plus original x and y values.
    """
    # Extract feature and target variables
    feature = df[col1].values
    target = df[col2].values
    # Fit linear model
    poly_model = Poly.fit(feature, target, 1)  # Linear fit (degree 1)
    covariance = np.polyfit(feature, target, 1, cov=True)[1]
    error_estimates = np.sqrt(np.diag(covariance))
    # Extract coefficients
    intercept, slope = poly_model.convert().coef
    # Report fit parameters
    print(f"slope = {slope:.2f} +/- {error_estimates[0]:.2f}")
    print(f"intercept = {intercept:.2f} +/- {error_estimates[1]:.2f}")
    # Generate fit line for plotting
    x_smooth = np.linspace(min(feature), max(feature), 100)
    y_predicted = slope * x_smooth + intercept
    # Package results
    model_results = {
        'params': [slope, intercept],
        'sigma': error_estimates,
        'xfit': x_smooth,
        'yfit': y_predicted
    }
    return model_results, feature, target


def plot_fitted_data(data, x, y):
    """
    Create a visualization of the original data with the fitted line
    and uncertainty band.
    Args:
        data (dict): Model parameters and fit data.
        x (array): Original x values.
        y (array): Original y values.
    """

    def linear_model(x_val, slope, intercept):
        """Calculate y-value using linear equation."""
        return slope * x_val + intercept

    
    # Setup visualization
    fig, ax = plt.subplots(dpi=144)
    
    # Plot original data points
    ax.plot(x, y, 'teal', marker='o', linestyle='', label='Original Data')  # Changed to teal
    
    # Extract model parameters
    x_fit_range = data['xfit']
    y_fit_values = data['yfit']
    slope, intercept = data['params']
    slope_error, intercept_error = data['sigma']
    
    # Plot fitted line
    ax.plot(x_fit_range, y_fit_values, 'darkorange', linestyle='-', label='Regression Line')  # Changed to dark orange
    
    # Add uncertainty band
    ax.fill_between(
        x_fit_range,
        linear_model(x_fit_range, slope - slope_error, intercept - intercept_error),
        linear_model(x_fit_range, slope + slope_error, intercept + intercept_error),
        color='gold',  # Changed to gold
        alpha=0.2,
        label='Confidence Band'
    )
    
    # Format plot
    ax.set_xlabel('Petal Length')
    ax.set_ylabel('Petal Width')
    ax.legend()
    
    # Save and display visualization
    plt.savefig('fitting.png')
    plt.show()
    return


def main():
    """
    Execute the complete data analysis workflow including data preparation,
    visualization, statistical analysis, clustering, and curve fitting.
    """
    # Load and preprocess data
    iris_data = pd.read_csv('data.csv')
    processed_data = preprocessing(iris_data)
    
    # Column selection for analysis
    target_column = 'petal_length'
    
    # Generate visualizations
    plot_relational_plot(processed_data)
    plot_statistical_plot(processed_data)
    plot_categorical_plot(processed_data)
    
    # Statistical analysis
    statistical_measures = statistical_analysis(processed_data, target_column)
    writing(statistical_measures, target_column)
    
    # Perform clustering analysis
    cluster_analysis = perform_clustering(processed_data, 'petal_length', 'sepal_width')
    plot_clustered_data(*cluster_analysis)
    
    # Perform regression analysis
    regression_analysis = perform_fitting(processed_data, 'petal_length', 'petal_width')
    plot_fitted_data(*regression_analysis)
    return


if __name__ == '__main__':
    main()