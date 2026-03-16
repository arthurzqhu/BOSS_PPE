import matplotlib.pyplot as plt
import numpy as np

def plot_traces(posterior, pnames):
    its, chains, dims = posterior.shape
    fig, axes = plt.subplots(dims, 1, figsize=(8, 2.5*dims), sharex=True)
    if dims == 1:
        axes = [axes]
    for d in range(dims):
        for c in range(chains):
            axes[d].plot(posterior[:,c,d], alpha=0.6)
        if len(pnames) == dims:
            axes[d].set_ylabel(pnames[d])
    axes[-1].set_xlabel('Iteration')
    plt.tight_layout()
    plt.show()

