import matplotlib.pyplot as plt

def plot_traces(posterior, pnames):
    its, chains, dims = posterior.shape
    fig, axes = plt.subplots(dims, 1, figsize=(8, 2.5*dims), sharex=True)
    for d in range(dims):
        for c in range(chains):
            axes[d].plot(posterior[:,c,d], alpha=0.6, label=f'chain {c+1}')
        axes[d].set_ylabel(pnames[d])
        axes[d].legend(loc='upper right')
    axes[-1].set_xlabel('Iteration')
    plt.tight_layout()
    plt.show()