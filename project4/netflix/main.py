import numpy as np
import kmeans
import common
import naive_em
import em

X = np.loadtxt("toy_data.txt")

# kMeans vs EM
K = [1, 2, 3, 4]    # Clusters to try
seeds = [0, 1, 2, 3, 4]     # Seeds to try

# Costs for diff. seeds
costs_kMeans = [0, 0, 0, 0, 0]
costs_EM = [0, 0, 0, 0, 0]

# Best seed for cluster based on lowest costs 
best_seed_kMeans = [0, 0, 0, 0]
best_seed_EM = [0, 0, 0, 0]

# Mixtures for best seeds
mixtures_kMeans = [0, 0, 0, 0, 0]
mixtures_EM = [0, 0, 0, 0, 0]

# Posterior probs. for best seeds
posts_kMeans = [0, 0, 0, 0, 0]
posts_EM = [0, 0, 0, 0, 0]

for k in range(len(K)):
    for i in range(len(seeds)):
        
        # Run kMeans
        mixtures_kMeans[i], posts_kMeans[i], costs_kMeans[i] = \
        kmeans.run(X, *common.init(X, K[k], seeds[i]))
        
        # Run Naive EM
        mixtures_EM[i], posts_EM[i], costs_EM[i] = \
        naive_em.run(X, *common.init(X, K[k], seeds[i]))
    
    # Print lowest cost
    print("Lowest cost for cluster using kMeans", K[k], "is:", np.min(costs_kMeans))
    
    # Save best seed for plotting
    best_seed_kMeans[k] = np.argmin(costs_kMeans)
    
    
    # Plot for cluster
    common.plot(X, mixtures[best_seed[k]], posts[best_seed[k]], title="Cluster plot")
