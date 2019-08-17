import numpy as np
import em
import common

X = np.loadtxt("netflix_incomplete.txt")
#X_gold = np.loadtxt("test_complete.txt")

n, d = X.shape

### Begin: Comparison of EM for matrix completion with K = 1 and 12 ###
K = [1, 12]    # Clusters to try
seeds = [0, 1, 2, 3, 4]     # Seeds to try

log_lh = [0, 0, 0, 0, 0]    # Log likelihoods for different seeds

# Best seed for cluster based on highest log likelihoods
best_seed = [0, 0]

# Mixtures for best seeds
mixtures = [0, 0, 0, 0, 0]

# Posterior probs. for best seeds
posts = [0, 0, 0, 0, 0]

for k in range(len(K)):
    for i in range(len(seeds)):
        
        # Run EM
        mixtures[i], posts[i], log_lh[i] = \
        em.run(X, *common.init(X, K[k], seeds[i]))
    
    # Print lowest cost
    print("=============== Clusters:", K[k], "======================")
    print("Highest log likelihood using EM is:", np.max(log_lh))
    
    # Save best seed for plotting
    best_seed[k] = np.argmax(log_lh)  