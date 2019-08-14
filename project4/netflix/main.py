import numpy as np
import kmeans
import common
import naive_em
import em

X = np.loadtxt("toy_data.txt")

# kMeans vs EM
K = [1, 2, 3, 4]

seeds = [0, 1, 2, 3, 4]

costs = [0, 0, 0, 0, 0]
best_seed = [0, 0, 0, 0]
mixtures = [0, 0, 0, 0, 0]
posts = [0, 0, 0, 0, 0]

for k in range(len(K)):
    for i in range(len(seeds)):
        mixtures[i], posts[i], costs[i] = kmeans.run(X, *common.init(X, K[k], seeds[i]))
    
    # Print lowest cost
    print("Lowest cost for cluster", K[k], "is:", np.min(costs))
    
    # Save best seed for plotting
    best_seed[k] = np.argmin(costs)
    
    # Plot for cluster
    common.plot(X, mixtures[best_seed[k]], posts[best_seed[k]], title="Cluster plot")
