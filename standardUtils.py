import numpy as np
import matplotlib.pyplot as plt
import tqdm


def NormalizeData(data):
    """
    Normalizes the input data to a range between 0 and 1.

    Args:
    - data: Numpy array or pandas series to be normalized.

    Returns:
    - Normalized data.
    """
    return (data - np.min(data)) / (np.max(data) - np.min(data))


def get_popularity_score(shorts_df):
    """
    Calculates the normalized popularity score based on video view count and subscriber count.

    Args:
    - shorts_df: DataFrame containing 'videoViewCount' and 'subscriberCount' columns.

    Returns:
    - minmax_normalize_viewcnt: Min-max normalized popularity scores.
    """
    # Make popularity score
    normalize_viewcnt = (shorts_df['videoViewCount'] + 1) / shorts_df['subscriberCount']
    minmax_normalize_viewcnt = NormalizeData(normalize_viewcnt)
    return minmax_normalize_viewcnt


def visualize(df):
    """
    Visualizes the cumulative distribution function (CDF) and the popularity score.

    Args:
    - df: DataFrame containing 'popularity_score' and 'cdf' columns.

    Returns:
    - None. Shows a plot.
    """
    fig, ax = plt.subplots()

    # Draw a vertical line at the point where the gradient is less than 1.0
    line_idx = min(df[df['gradient'] < 1.0]['popularity_score'])
    ax.axvline(line_idx, 0, 1, color='red', linestyle='--', linewidth=1)

    # Scatter plot of popularity score vs. CDF
    df.plot.scatter(x='popularity_score', y='cdf', grid=True, ax=ax)

    # Set labels and show plot
    ax.set_xlabel('Normalized Popularity Score')
    ax.set_ylabel('Cumulative Distribution Function')
    plt.show()


def measure_popularity_score(shorts_df):
    """
    Measures the popularity score of videos and visualizes the cumulative distribution.

    Args:
    - shorts_df: DataFrame containing 'popularity_score'.

    Returns:
    - Updated shorts_df with a new 'popularity' column indicating popular videos.
    """
    # Filter and sort DataFrame based on popularity_score
    df = shorts_df[shorts_df['popularity_score'] < 0.2]
    df = df.sort_values('popularity_score')

    # Normalize popularity score and calculate CDF
    df['popularity_score'] = (df['popularity_score'] - df['popularity_score'].min()) / (
                df['popularity_score'].max() - df['popularity_score'].min())
    df['cdf'] = df['popularity_score'].rank(method='average', pct=True)

    # Calculate the slope (gradient) within a small window
    gradients = []
    c = 0.0005  # Smoothing constant for gradient calculation
    for i in tqdm.trange(len(df), desc="Calculating gradients"):
        x_point = df.iloc[i]

        x_upper = x_point['popularity_score'] + c
        x_lower = x_point['popularity_score'] - c

        window = df[df['popularity_score'] < x_upper]
        window = window[window['popularity_score'] > x_lower]

        if len(window) > 1:  # Ensure window has more than one point to avoid division by zero
            gradient = (window.iloc[-1]['cdf'] - window.iloc[0]['cdf']) / (x_upper - x_lower)
        else:
            gradient = 0

        gradients.append(gradient)

    df['gradient'] = gradients

    # Visualize the results
    visualize(df)

    # Determine popular videos based on gradient threshold
    popular_index = df[df['gradient'] < 1.0].index
    shorts_df = shorts_df[shorts_df['popularity_score'] < 0.2]
    shorts_df['popularity'] = 0
    shorts_df.loc[popular_index, 'popularity'] = 1  # Set popularity to 1 for popular videos

    return shorts_df
