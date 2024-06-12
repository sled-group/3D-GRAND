from sklearn.decomposition import PCA
import plotly.graph_objects as go
import torch


def visualize_features_PCA(
    xyz: torch.Tensor,
    features: torch.Tensor,
):
    """visualize the features using PCA

    Args:
        xyz (torch.Tensor): [num_points, 3]
        features (torch.Tensor): [num_points, feature_dim]
    """

    xyz = xyz.cpu().numpy()
    features = features.cpu().numpy()

    pca = PCA(n_components=3)
    pca_result = pca.fit_transform(features)  # (N, channel)

    # Normalize the results between 0 and 1 for color mapping
    pca_result -= pca_result.min(axis=0)
    pca_result /= pca_result.max(axis=0)

    # Scale to 0-255 for RGB
    pca_result = (pca_result * 255).astype(int)

    # Convert to a format Plotly understands
    color = ["rgb({},{},{})".format(r, g, b) for r, g, b in pca_result]

    # Create interactive 3D scatter plot
    fig = go.Figure(
        data=[
            go.Scatter3d(
                x=xyz[:, 0],
                y=xyz[:, 1],
                z=xyz[:, 2],
                mode="markers",
                marker=dict(
                    size=5,
                    color=color,  # set color to an array/list of desired values
                    opacity=0.8,
                ),
            )
        ]
    )

    fig.show()
