import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from matplotlib.colors import LinearSegmentedColormap
import matplotlib.gridspec as gridspec

def load_pkl_tensors(file_path):
    """Load tensors from a pickle file."""
    return [t.squeeze(0, 2) for t in torch.load(file_path)]

def vector_projection_3d(A, B, project=True):
    """
    Calculate vector projection of A onto B for every H & W position.
    A and B are 3D tensors of shape [C, H, W].
    Returns projection tensor of same shape.
    """
    # Reshape tensors to [H*W, C] to treat each spatial position as a separate vector
    H, W = A.shape[1], A.shape[2]
    A_reshaped = A.permute(1, 2, 0).reshape(-1, A.shape[0])  # [H*W, C]
    B_reshaped = B.permute(1, 2, 0).reshape(-1, B.shape[0])  # [H*W, C]
    
    # Calculate dot product for all positions: [H*W]
    dot_product = torch.sum(A_reshaped * B_reshaped, dim=1)
    
    # Calculate magnitude squared of B for all positions: [H*W]
    B_mag_squared = torch.sum(B_reshaped * B_reshaped, dim=1)
    A_mag_squared = torch.sum(A_reshaped * A_reshaped, dim=1)
    
    # Calculate projection scalar: [H*W]
    proj_scalar = dot_product / (B_mag_squared + 1e-8)  # add small epsilon to avoid division by zero
    
    # Calculate cosine similarity: [H*W]
    cos_sim = dot_product / (torch.sqrt(A_mag_squared * B_mag_squared) + 1e-8)
    
    return proj_scalar.reshape(H, W) if project else cos_sim.reshape(H, W)

def cumsum_tensor_lists(tlist):
    """
    Args:
        tlist (list): List of tensors
        
    Returns:
        list: List of cumulative sum tensors
    """
        
    result = []
    cumsum = torch.zeros_like(tlist[0])  # Initialize with zeros tensor of same shape/dtype
    
    for t in tlist:
        cumsum += t
        result.append(cumsum.clone())  # Clone to avoid reference issues
    return result

def compare_tensors(pkl_file1, pkl_file2, seq_stages = 10):
    """Compare tensors from two pickle files."""
    tensors1 = load_pkl_tensors(pkl_file1)
    tensors2 = load_pkl_tensors(pkl_file2)
    
    if len(tensors1) != len(tensors2):
        raise ValueError(f"Number of tensors mismatch: {len(tensors1)} vs {len(tensors2)}")
    
    cumsum1 = cumsum_tensor_lists(tensors1)
    cumsum2 = cumsum_tensor_lists(tensors2)
    torch.testing.assert_close(cumsum1[seq_stages], cumsum2[seq_stages])

    # last residual
    R_b = tensors1[seq_stages]
    # common base
    B = cumsum1[seq_stages]
    # ground truth
    G = cumsum1[-1]
    # ground truth residual
    R_g = G - B

    Rs_seq = tensors1[seq_stages - len(tensors1) - 1:]
    # residuals of parallel
    Rs_para = tensors2[seq_stages - len(tensors2) - 1:]

    names = ["B", "R_g"]
    for i in range(len(Rs_seq)):
        names.append(f"R[{i-len(Rs_seq)}]")
    n = len(names) # B & R_g & Rs
    m = 11
    fig = plt.figure(figsize=(n*5, m*4))  
    gs = gridspec.GridSpec(m, n, height_ratios=[1]*m, width_ratios=[1]* n)  
    
    # Find global min and max for consistent color scaling
    vmin = -1
    vmax = 1

    # Create a custom colormap from blue to white to red
    colors = [(0, 0, 1), (1, 1, 1), (1, 0, 0)]  # blue -> white -> red
    #colors = [(1, 1, 1), (1, 0, 0)]  # blue -> white -> red
    custom_cmap = LinearSegmentedColormap.from_list('custom', colors)

    r = 0
    for i, matrix in enumerate([B, R_g, *Rs_seq]):
        ax = plt.subplot(gs[r, i])
        sns.heatmap(vector_projection_3d(matrix, G).cpu(), ax=ax, 
                   cmap=custom_cmap,
                   vmin=vmin,
                   vmax=vmax,
                   # center=0,
                   cbar_kws={'label': 'Value'})
        ax.set_title(f'Seq {names[i]} contrib on G')
    
    r += 1
    for i, matrix in enumerate([B, R_g, *Rs_seq]):
        ax = plt.subplot(gs[r, i])
        m = vector_projection_3d(matrix, G).cpu()
        th_hi = np.percentile(m, 80)
        mask = m < th_hi
        m[mask] = 0
        sns.heatmap(m, ax=ax, 
                   cmap=custom_cmap,
                   vmin=vmin,
                   vmax=vmax,
                   # center=0,
                   cbar_kws={'label': 'Value'})
        ax.set_title(f'Seq {names[i]} contrib on G, top 20%')
    
    r += 1
    for i, matrix in enumerate([*Rs_para]):
        ax = plt.subplot(gs[r, i+2])
        sns.heatmap(vector_projection_3d(matrix, G).cpu(), ax=ax, 
                   cmap=custom_cmap,
                   vmin=vmin,
                   vmax=vmax,
                   center=0,
                   cbar_kws={'label': 'Value'})
        ax.set_title(f'Para {names[i+2]} contrib on G')
    
    r += 1
    para_sum = 0
    for i, matrix in enumerate([*Rs_para]):
        ax = plt.subplot(gs[r, i+2])
        m = vector_projection_3d(matrix, G).cpu()
        th_hi = np.percentile(m, 80)
        mask = m < th_hi
        m[mask] = 0
        para_sum += m
        sns.heatmap(m, ax=ax, 
                   cmap=custom_cmap,
                   vmin=vmin,
                   vmax=vmax,
                   # center=0,
                   cbar_kws={'label': 'Value'})
        ax.set_title(f'Seq {names[i]} contrib on G, top 20%')
    ax = plt.subplot(gs[r, 1]) 
    sns.heatmap(para_sum.cpu(), ax=ax, 
               cmap=custom_cmap,
               vmin=vmin,
               vmax=vmax,
               # center=0,
               cbar_kws={'label': 'Value'})
    ax.set_title(f'Para sum contrib on G, top 20%')   
    
    r += 1
    ax = plt.subplot(gs[r, 1])
    sns.heatmap(vector_projection_3d(sum(Rs_para), G).cpu(), ax=ax, 
                   cmap=custom_cmap,
                   vmin=vmin,
                   vmax=vmax,
                   center=0,
                   cbar_kws={'label': 'Value'})
    ax.set_title(f'R_g_para contrib on G')
    for i, matrix in enumerate([*Rs_seq]):
        ax = plt.subplot(gs[r, i+2])
        sns.heatmap(vector_projection_3d(matrix, R_g).cpu(), ax=ax, 
                   cmap=custom_cmap,
                   vmin=vmin,
                   vmax=vmax,
                   center=0,
                   cbar_kws={'label': 'Value'})
        ax.set_title(f'Seq {names[i+2]} contrib on R_g')
            
    r += 1
    for i, matrix in enumerate([*Rs_para]):
        ax = plt.subplot(gs[r, i+2])
        sns.heatmap(vector_projection_3d(matrix, R_g).cpu(), ax=ax, 
                   cmap=custom_cmap,
                   vmin=vmin,
                   vmax=vmax,
                   center=0,
                   cbar_kws={'label': 'Value'})
        ax.set_title(f'Para {names[i+2]} contrib on R_g')
    
    r += 1
    for i, (m1, m2) in enumerate(zip(Rs_para, Rs_seq)):
        ax = plt.subplot(gs[r, i+2])
        m1 = vector_projection_3d(m1, G).cpu()
        m2 = vector_projection_3d(m2, G).cpu()
        ax.scatter(m1, m2, alpha=0.5, marker='.')
        pall = pearson_correlation(m1, m2)
        ax.set_title(f'Para vs Seq {names[i+2]} pearson {pall:.4f}')
        ax.set_xlabel(f'Para {names[i+2]} over G')
        ax.set_ylabel(f'Seq {names[i+2]} over G')
    
    r += 1
    m2 = vector_projection_3d(R_g, G).cpu()
    for i, m in enumerate(Rs_para):
        ax = plt.subplot(gs[r, i+2])
        m1 = vector_projection_3d(m, G).cpu()
        ax.scatter(m1, m2, alpha=0.5, marker='.')
        pall = pearson_correlation(m1, m2)
        ax.set_title(f'Para {names[i+2]} vs R_g pearson {pall:.4f}')
        ax.set_xlabel(f'Para {names[i+2]} over G')
        ax.set_ylabel(f'R_g over G')
    
    r += 1
    m2 = get_freq1(B).cpu()
    ax = plt.subplot(gs[r, 0])
    sns.heatmap(m2, ax=ax, 
               cmap=custom_cmap,
               cbar_kws={'label': 'Value'})

    m_R_g = vector_projection_3d(R_g, G).cpu()
    ax = plt.subplot(gs[r, 1])
    th = np.percentile(m2, 20)
    mask = m2 > th
    ax.scatter(m2[mask], m_R_g[mask], alpha=0.5, marker='.')
    pall = pearson_correlation(m2, m_R_g)
    ax.set_title(f'R_g over G vs B_freq1 pearson {pall:.4f}')
    ax.set_xlabel(f'B_freq1')
    ax.set_ylabel(f'R_g over G')
    for i, m in enumerate(Rs_para):
        ax = plt.subplot(gs[r, i+2])
        m1 = vector_projection_3d(m, R_g, project=False).cpu()
        ax.scatter(m1, m2, alpha=0.5, marker='.')
        pall = pearson_correlation(m1, m2)
        ax.set_title(f'Para {names[i+2]} vs B_freq1 pearson {pall:.4f}')
        ax.set_xlabel(f'Para {names[i+2]} over R_g')
        ax.set_ylabel(f'B_freq1')
    
    r += 1
    m2 = get_freq2(B).cpu()
    ax = plt.subplot(gs[r, 0])
    sns.heatmap(m2, ax=ax, 
               cmap=custom_cmap,
               cbar_kws={'label': 'Value'})
    ax = plt.subplot(gs[r, 1])
    th = np.percentile(m2, 20)
    mask = m2 > th
    ax.scatter(m2[mask], m_R_g[mask], alpha=0.5, marker='.')
    pall = pearson_correlation(m2[mask], m_R_g[mask])
    ax.set_title(f'R_g over G vs B_freq2 pearson {pall:.4f}')
    ax.set_xlabel(f'B_freq2')
    ax.set_ylabel(f'R_g over G')
    for i, m in enumerate(Rs_para):
        ax = plt.subplot(gs[r, i+2])
        m1 = vector_projection_3d(m, R_g, project=False).cpu()
        ax.scatter(m1, m2, alpha=0.5, marker='.')
        pall = pearson_correlation(m1, m2)
        ax.set_title(f'Para {names[i+2]} vs B_freq2 pearson {pall:.4f}')
        ax.set_xlabel(f'Para {names[i+2]} over R_g')
        ax.set_ylabel(f'B_freq2')
    
    r += 1
    m2 = get_freq3(B, 3).cpu()
    ax = plt.subplot(gs[r, 0])
    sns.heatmap(m2, ax=ax, 
               cmap=custom_cmap,
               cbar_kws={'label': 'Value'})
    ax = plt.subplot(gs[r, 1])

    th = np.percentile(m2, 20)
    mask = m2 > 0.025
    ax.scatter(m2[mask], m_R_g[mask], alpha=0.5, marker='.')
    pall = pearson_correlation(m2[mask], m_R_g[mask])
    ax.set_title(f'R_g over G vs B_freq3 pearson {pall:.4f}')
    ax.set_xlabel(f'B_freq3')
    ax.set_ylabel(f'R_g over G')
    for i, m in enumerate(Rs_para):
        ax = plt.subplot(gs[r, i+2])
        m1 = vector_projection_3d(m, R_g, project=False).cpu()
        ax.scatter(m1, m2, alpha=0.5, marker='.')
        pall = pearson_correlation(m1, m2)
        ax.set_title(f'Para {names[i+2]} vs B_freq3 pearson {pall:.4f}')
        ax.set_xlabel(f'Para {names[i+2]} over R_g')
        ax.set_ylabel(f'B_freq3')

    plt.tight_layout()
    plt.savefig('heatmap_visualization.png', dpi=300)  # High DPI for better quality
    plt.close()
    # proj = [vector_projection_3d(R_b, B).cpu()]
    # for x in [B, R_g]:
    #     proj.append(vector_projection_3d(x, G).cpu())
    # for x in Rs_seq:
    #     proj.append(vector_projection_3d(x, R_g).cpu())
    
    # plot_heatmaps(B, proj, save_path='heatmap_visualization.png')


def plot_heatmaps(B, matrices, titles=None, save_path=None):
    """Plot a row of heatmaps for the given matrices with the same value range.
    
    Args:
        matrices: List of 2D numpy arrays to plot
        titles: Optional list of titles for each heatmap
        save_path: Optional path to save the figure
    """
    n = len(matrices)
    m = 4
    fig = plt.figure(figsize=(n*5, m*4))  
    gs = gridspec.GridSpec(m, n, height_ratios=[1]*m, width_ratios=[1]* n)  
    # fig, axes = plt.subplots(2, n, figsize=(5*n, 8))
    if n == 1:
        axes = [axes]
    
    # Find global min and max for consistent color scaling
    vmin = -1
    vmax = 1

    names = ["R_b", "B", "R_g"]
    for i in range(len(matrices) - 3):
        names.append(f"R[{i-3}]")
    
    # Create a custom colormap from blue to white to red
    colors = [(0, 0, 1), (1, 1, 1), (1, 0, 0)]  # blue -> white -> red
    custom_cmap = LinearSegmentedColormap.from_list('custom', colors)
        
    for i, matrix in enumerate(matrices):
        ax = plt.subplot(gs[0, i])
        sns.heatmap(matrix, ax=ax, 
                   cmap=custom_cmap,
                   vmin=vmin,
                   vmax=vmax,
                   center=0,
                   cbar_kws={'label': 'Value'})
        if titles and i < len(titles):
            ax.set_title(titles[i])
    

    def plot_row(basic_mat, basic_name, row_id):
        th_hi = np.percentile(basic_mat, 80)
        th_mi = np.percentile(basic_mat, 60)
        th_lo = np.percentile(basic_mat, 50)

        mask0 = basic_mat >= th_hi
        # mask1 = (basic_mat >= th_mi)
        # mask2 = (basic_mat >= th_lo)

        mask1 = (basic_mat < th_hi) & (basic_mat >= th_mi)
        mask2 = (basic_mat < th_mi) & (basic_mat >= th_lo)


        for i, matrix in enumerate(matrices):
            if i == 2:
                continue
            ax = plt.subplot(gs[row_id, i])
            #ax.scatter(basic_mat[mask], matrix[mask], alpha=0.5)
            ax.scatter(basic_mat, matrix, alpha=0.5, marker='.')
            # ax.set_ylim(-0.1, 0.3)
            pall = pearson_correlation(matrix, basic_mat)
            p0 = pearson_correlation(matrix[mask0], basic_mat[mask0])
            p1 = pearson_correlation(matrix[mask1], basic_mat[mask1])
            p2 = pearson_correlation(matrix[mask2], basic_mat[mask2])
            ax.set_title(f"{basic_name} vs {names[i]}: pearson {pall:.4f}|{p0:.4f}|{p1:.4f}|{p2:.4f}")
            ax.set_xlabel(f"{basic_name}")
            ax.set_ylabel(f"{names[i]}") 

    plot_row(matrices[2], "R_g", 1)
    plot_row(matrices[0], "R_b", 3)

    B_freq = get_freq3(B, 3).cpu()
    ax = plt.subplot(gs[1, 2])
    ax.scatter(matrices[2], B_freq, alpha=0.5, marker='.')
    ax.set_title(f"R_g vs B_freq")
    ax.set_xlabel(f"R_g")
    ax.set_ylabel(f"B_freq") 

    ax = plt.subplot(gs[2, 2])
    cc = np.percentile(B_freq, 80)
    sns.heatmap(B_freq, ax=ax, cmap=custom_cmap, center=cc)
    ax.set_title("B_freq heatmap")
    plot_row(B_freq, "B_freq", 2)
            
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300)  # High DPI for better quality
    plt.close()

def get_freq1(code):
    # codes: [d, pn, pn]
    C, H, W = code.shape
    code = code.reshape(C, -1)
    dc_component = torch.mean(code, dim=1, keepdim=True)
    dc_diff = torch.norm(code - dc_component, dim=0).reshape(H, W)
    return dc_diff


def get_freq2(tensor):
    """Calculate frequency using Sobel filters for [C, H, W] tensor.
    
    Args:
        tensor (torch.Tensor): Input tensor of shape [C, H, W]
    Returns:
        torch.Tensor: Gradient magnitude of shape [H, W]
    """
    # Add batch dimension for conv2d
    tensor = tensor.unsqueeze(0)  # [1, C, H, W]
    
    sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], device=tensor.device,
                          dtype=torch.float32).view(1, 1, 3, 3).repeat(tensor.shape[1], 1, 1, 1)
    sobel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], device=tensor.device,
                          dtype=torch.float32).view(1, 1, 3, 3).repeat(tensor.shape[1], 1, 1, 1)
    
    # Apply convolution across all channels
    grad_x = F.conv2d(tensor, sobel_x, padding='same', groups=tensor.shape[1])
    grad_y = F.conv2d(tensor, sobel_y, padding='same', groups=tensor.shape[1])
    
    # Calculate magnitude and average across channels
    gradient_magnitude = torch.sqrt(grad_x**2 + grad_y**2)
    gradient_magnitude = F.avg_pool2d(gradient_magnitude, kernel_size=3, stride=1, padding=1, count_include_pad=False)
    #print(gradient_magnitude.shape, high_freq_map.shape)
    return gradient_magnitude.mean(dim=1).squeeze(0)  # Return [H, W]


def get_freq3(tensor, window_size=3):
    x = tensor.unsqueeze(0)  # [1, C, H, W]
    mean = F.avg_pool2d(x, window_size, stride=1, padding=window_size//2, count_include_pad=False)
    mean_sq = F.avg_pool2d(x.pow(2), window_size, stride=1, padding=window_size//2, count_include_pad=False)
    #import pdb; pdb.set_trace()
    return (mean_sq - mean.pow(2)).mean(dim=1).squeeze(0)

def plot_3d_bars(matrix, title="3D Bar Graph", save_path="plot.pdf"):
    """
    Create a 3D bar graph from a 2D matrix and save it to a PDF file.
    
    Args:
        matrix (torch.Tensor or np.ndarray): Input matrix of shape (H, W)
        title (str): Title of the graph
        save_path (str): Path to save the PDF file
    """
    # Convert torch tensor to numpy if needed
    if isinstance(matrix, torch.Tensor):
        matrix = matrix.detach().cpu().numpy()
    
    # Create figure and 3D axes
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    # Create coordinates for each bar
    H, W = matrix.shape
    xpos, ypos = np.meshgrid(np.arange(W), np.arange(H))
    xpos = xpos.flatten()
    ypos = ypos.flatten()
    zpos = np.zeros_like(xpos)
    
    # Define bar dimensions
    dx = dy = 0.8
    dz = matrix.flatten()
    
    # Create the 3D bars
    ax.bar3d(xpos, ypos, zpos, dx, dy, dz)
    
    # Set labels and title
    ax.set_xlabel('Width')
    ax.set_ylabel('Height')
    ax.set_zlabel('Value')
    ax.set_title(title)
    
    # Adjust the viewing angle for better visualization
    ax.view_init(elev=30, azim=45)
    
    # Save the plot to PDF file
    plt.savefig(save_path, format='pdf', bbox_inches='tight')
    plt.close(fig)
    
    return fig


def plot_scatter(A, B, title="Scatter Plot", save_path="scatter.png"):
    """
    Create a scatter plot of two vectors A and B.
    
    Args:
        A (torch.Tensor): First vector
        B (torch.Tensor): Second vector
        title (str): Title of the plot
        save_path (str): Path to save the figure
    """
    # Convert torch tensors to numpy if needed
    if isinstance(A, torch.Tensor):
        A = A.detach().cpu().numpy()
    if isinstance(B, torch.Tensor):
        B = B.detach().cpu().numpy()
    
    # Create the scatter plot
    plt.figure(figsize=(8, 8))
    plt.scatter(A.flatten(), B.flatten(), alpha=0.5)
    
    # Add labels and title
    plt.xlabel('Vector A')
    plt.ylabel('Vector B')
    plt.title(title)
    
    # Add grid
    plt.grid(True, linestyle='--', alpha=0.7)
    
    # Save the plot
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()


def pearson_correlation(A, B):
    """
    Calculate Pearson correlation between two tensors A and B.
    
    Args:
        A (torch.Tensor): First tensor
        B (torch.Tensor): Second tensor
        
    Returns:
        torch.Tensor: Pearson correlation coefficient
    """
    # Ensure tensors are on the same device
    if A.device != B.device:
        B = B.to(A.device)
    
    # Calculate means
    A_mean = torch.mean(A)
    B_mean = torch.mean(B)
    
    # Center the variables
    A_centered = A - A_mean
    B_centered = B - B_mean
    
    # Calculate numerator (covariance)
    numerator = torch.sum(A_centered * B_centered)
    
    # Calculate denominator (product of standard deviations)
    denominator = torch.sqrt(torch.sum(A_centered ** 2) * torch.sum(B_centered ** 2) + 1e-8)
    
    # Calculate correlation
    correlation = numerator / denominator
    
    return correlation


if __name__ == "__main__":
    import sys
    if len(sys.argv) != 3:
        print("Usage: python tensor_diff_compare.py file1.pkl file2.pkl")
        sys.exit(1)
        
    compare_tensors(sys.argv[1], sys.argv[2])
