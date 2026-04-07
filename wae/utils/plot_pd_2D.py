#!/usr/bin/env python3

import numpy as np
from optparse import OptionParser
import matplotlib.pyplot as plt
from  matplotlib.colors import ListedColormap

def load_and_process_data(fout):
    """Load and process data from file"""
    datas = np.loadtxt(fout)

    epsilons, rs = datas[:, 0], datas[:, 1]
    latents = datas[:, 2:-2]
    labels = datas[:, -2].astype('int')
    labels_LS = datas[:, -1].astype('int')

    nbin = int(np.max(labels))
    eps = 0.00001
    hist, bin_edges = np.histogram(labels, bins=nbin+1, range=(-eps, nbin-eps+1))
    indices = np.argsort(-hist)
    l = len(indices)
    ihash = np.zeros(l)
    for i, x in enumerate(indices):
        ihash[x] = i

    labels = np.array([ihash[label] for label in labels])
    labels_LS = np.array([ihash[label] for label in labels_LS])

    return epsilons, rs, labels, labels_LS

def create_phase_diagram(rs, epsilons, labels, cmap):
    """Create phase diagram plot"""
    fig, axs1 = plt.subplots()
    axs1.scatter(rs, epsilons, c=labels, s=30, marker='s', cmap=cmap)
    axs1.set_xlim([1.1, 2.1])
    axs1.set_ylim([0.1, 5.0])
    axs1.set_xlabel('$r_0$', size=16)
    axs1.set_ylabel('$\epsilon$', size=16)
    axs1.set_title('Predicted phase diagram')
    return fig, axs1

def create_phase_diagram_with_ls(rs, epsilons, labels_LS, cmap):
    """Create phase diagram with LS plot"""
    fig, axs2 = plt.subplots()
    axs2.scatter(rs, epsilons, c=labels_LS, s=30, marker='s', cmap=cmap)
    axs2.set_xlim([1.1, 2.1])
    axs2.set_ylim([0.1, 5.0])
    axs2.set_xlabel('$r_0$', size=16)
    axs2.set_ylabel('$\epsilon$', size=16)
    axs2.set_title('Predicted phase diagram with LS')
    return fig, axs2

def plot_2d_atomic_structure(crystal_data, index=0, title="2D Atomic Structure"):
    """Plot 2D atomic structure from crystal data"""
    fig, ax = plt.subplots()
    
    # crystal_data shape is (5000, 1024, 3) - select specific structure
    structure = crystal_data[index]
    
    # Each row is [x, y, atom_type]
    x = structure[:, 0]
    y = structure[:, 1]
    atom_types = structure[:, 2]
    
    # Create a scatter plot with different colors for different atom types
    scatter = ax.scatter(x, y, c=atom_types, cmap='viridis', s=100, marker='o')

    ax.set_aspect('equal')
    
    return fig, ax

def plot_typical_structures_for_labels(crystal_data, ljgp_params, rs, epsilons, labels, cmap_str):
    """Plot typical atomic structures for each label class"""
    unique_labels = np.unique(labels)
    n_labels = len(unique_labels)
    
    # Create a two-row subplot grid
    n_cols = int(np.ceil(n_labels / 2))
    fig, axes = plt.subplots(2, n_cols, figsize=(5*n_cols, 10))
    axes = axes.flatten()  # Flatten for easier indexing
    
    # Get the label mapping from phase diagram
    nbin = int(np.max(labels))
    eps = 0.00001
    hist, bin_edges = np.histogram(labels, bins=nbin+1, range=(-eps, nbin-eps+1))
    indices = np.argsort(-hist)
    l = len(indices)
    ihash = np.zeros(l)
    for i, x in enumerate(indices):
        ihash[x] = i
    
    for i, label in enumerate(unique_labels):
        # Find structures with this label
        label_indices = np.where(labels == label)[0]
        
        if len(label_indices) > 0:
            # Randomly select a representative structure
            rep_index = np.random.choice(label_indices)
            structure = crystal_data[rep_index]
            
            # Get corresponding epsilon and rs values
            epsilon = ljgp_params[rep_index, 0]
            rs = ljgp_params[rep_index, 1]
            
            # Plot the structure with the same color as the label class
            x = structure[:, 0]
            y = structure[:, 1]
            atom_types = structure[:, 2]
            
            # Use the same color as the label class from phase diagram
            mapped_label = ihash[label]
            scatter = axes[i].scatter(x, y, c=atom_types, cmap='viridis', s=100, marker='o',
                                    edgecolor=cmap_str[int(mapped_label)], linewidth=0.5)
            axes[i].set_title(f'Label {label}\nε={epsilon:.3f}, r={rs:.3f}')
            axes[i].set_xlabel('X Position')
            axes[i].set_ylabel('Y Position')
            axes[i].set_aspect('equal')
    
    # Hide any unused subplots
    for j in range(n_labels, len(axes)):
        axes[j].axis('off')
    
    fig.suptitle("Typical Structures by Label")
    return fig, axes

def main():
    parser = OptionParser()
    parser.add_option("--output", dest="output", default=None,
                      help="Specify an output from evaluation.")
    parser.add_option("--crystal_data", dest="crystal_data", default=None,
                      help="Specify the crystal data file to plot.")
    parser.add_option("--ljgp_params", dest="ljgp_params", default=None,
                      help="Specify the LJGP parameters file (ljgp_params_2d.npy).")
    parser.add_option("--plot_labels", dest="plot_labels", action="store_true", default=False,
                      help="Plot typical structures for each label class.")

    (options, args) = parser.parse_args()
    fout = options.output
    crystal_file = options.crystal_data
    ljgp_file = options.ljgp_params
    plot_labels = options.plot_labels

    if fout is None and crystal_file is None and not plot_labels:
        print("Error: Please specify either an output file using --output, crystal data file using --crystal_data, or enable label plotting with --plot_labels")
        return

    cmap_str = ['blue', 'orange', 'green', 'red', 'purple', 'yellow', 'pink', 'olive', 'cyan',
                'darkblue', 'darkcyan', 'violet', 'darkred', 'darkgreen', 'chocolate', 'brown',
                'lime', 'dodgerblue', 'indigo', 'peru', 'darkorange', 'magenta']
    cmap = ListedColormap(cmap_str)

    if fout:
        epsilons, rs, labels, labels_LS = load_and_process_data(fout)
        fig1, axs1 = create_phase_diagram(rs, epsilons, labels, cmap)
        fig2, axs2 = create_phase_diagram_with_ls(rs, epsilons, labels_LS, cmap)

    if crystal_file and ljgp_file and plot_labels:
        crystal_data = np.load(crystal_file)
        ljgp_params = np.load(ljgp_file)
        plot_typical_structures_for_labels(crystal_data, ljgp_params, rs, epsilons, labels, cmap_str)


    plt.show()

if __name__ == "__main__":
    main()