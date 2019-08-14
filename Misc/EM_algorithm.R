# Toy EM Algorithm: 1-D data

means <- c(6, 7)
vars <- c(1, 4)
probs <- c(0.5, 0.5)

x <- c(-1.0, 0.0, 4.0, 5.0, 6.0)

### E-step: Bayes calc ###
cl_p <- matrix(data = 0, nrow = length(x), ncol = length(means)) # Matrix to store cluster probs


### Main loop ###
loops = 10
var1 <- vector(mode = "list", length = loops)
mu1 <- vector(mode = "list", length = loops)
var2 <- vector(mode = "list", length = loops)
mu2 <- vector(mode = "list", length = loops)

for (l in c(1:loops)) { 
  # Normalizing constants
  norm <- c(0.0, 0.0, 0.0, 0.0, 0.0)
  for (i in c(1:length(x))) {
    for (j in c(1:length(means))) {
      norm[i] <- norm[i] + probs[j] * dnorm(x[i], mean = means[j], sd = sqrt(vars[j]))
    }
  }
  
  bayes <- function(x, means, vars, probs, i, j) {
    bayes <- (probs[j] * dnorm(x[i], mean = means[j], sd = sqrt(vars[j])))/norm[i]
    return(bayes)
  }
  
  # Calc cluster probs, p(j|i) for all x's
  for (i in c(1:length(x))) {
    for (j in c(1:length(means))) {
      cl_p[i,j] <- bayes(x, means, vars, probs, i, j)
    }
  }
  
  ### M-step: Now update the params. with the revised cluster probs ###
  n <- length(x)
  d <- 1
  nj = c(0, 0)
  
  for (j in c(1:length(means))) {
    for (i in c(1:length(x))) {
      nj[j] <- nj[j] + cl_p[i,j]
    }
  }
  
  probs <- nj/n
  
  m = c(0, 0)
  v = c(0, 0)
  for (j in c(1:length(means))) {
    for (i in c(1:n)) {
      m[j] <- m[j] + cl_p[i,j]*x[i]
    }
  }
  means <- (1/nj)*m
  
  for (j in c(1:length(means))) {
    for (i in c(1:n)) {
      v[j] <- v[j] + cl_p[i,j]*(x[i] - means[j])^2
    }
  }
  vars <- (1/(nj*d))*v
  
  print(paste0("--------------------- Iteration ", l, " ------------------------"))
  print("Revised prob. =")
  print(probs)
  print("Revised mean =")
  means
  print(means)
  print("Revised variance =")
  print(vars)
  mu1[[l]] <- means[1]
  mu2[[l]] <- means[2]
  var1[[l]] <- vars[1]
  var2[[l]] <- vars[2]
}

# Plot means and variances with iterations
par(mfrow = c(2, 2))
plot(c(1:loops), mu1, type="l", col="red", xlab = "Iterations through the dataset", ylab = "Cluster 1 Mean")
plot(c(1:loops), var1, type="l", col="green", xlab = "Iterations through the dataset", ylab = "Cluster 1 Variance")
plot(c(1:loops), mu2, type="l", col="red", xlab = "Iterations through the dataset", ylab = "Cluster 2 Mean")
plot(c(1:loops), var2, type="l", col="green", xlab = "Iterations through the dataset", ylab = "Cluster 2 Variance")


###################### HW5: EM algorithm calculations ############################

### Log-likelihood calc. with theta(0) ###
x <- c(-1,0,4,5,6)
mu <- c(6,7)
var <- c(1,4)
pis <- c(0.5,0.5)

log_lh = 0.0
for (i in c(1:length(x))) {
  log_lh = log_lh + log(pis[1]*dnorm(x[i], mean = mu[1], sd = sqrt(var[1])) + pis[2]*dnorm(x[i], mean = mu[2], sd = sqrt(var[2])))
}

### Mixture affiliations with initial params, theta(0) ###
cl = c(0,0,0,0,0) # Clusters
norm_const <- c(0,0,0,0,0)

# Calc. norm constants for each data point
for (i in c(1:length(x))) {
  norm_const[i] <- pis[1]*dnorm(x[i], mean = mu[1], sd = sqrt(var[1])) + pis[2]*dnorm(x[i], mean = mu[2], sd = sqrt(var[2]))
}

# Update loop for means and mixture affiliations
nj1 <- 0.0
mu1_sum <- 0.0
nj2 <- 0.0
mu2_sum <- 0.0
for (i in c(1:length(x))) {
  prob_cl1 <- pis[1]*dnorm(x[i], mean = mu[1], sd = sqrt(var[1]))/norm_const[i]
  nj1 <- nj1 + prob_cl1
  mu1_sum <- mu1_sum + prob_cl1*x[i]
  prob_cl2 <- pis[2]*dnorm(x[i], mean = mu[2], sd = sqrt(var[2]))/norm_const[i]
  nj2 <- nj2 + prob_cl2
  mu2_sum <- mu2_sum + prob_cl2*x[i]
  if (prob_cl1 > prob_cl2) {
    cl[i] = 1
  }
  else {
    cl[i] = 2
  }
}

# Updated means
means <- c(mu1_sum/nj1, mu2_sum/nj2)

# Update loop for variances
var1_sum <- 0.0
var2_sum <- 0.0
for (i in c(1:length(x))) {
  prob_cl1 <- pis[1]*dnorm(x[i], mean = mu[1], sd = sqrt(var[1]))/norm_const[i]
  var1_sum <- var1_sum + prob_cl1*(x[i] - means[1])^2
  prob_cl2 <- pis[2]*dnorm(x[i], mean = mu[2], sd = sqrt(var[2]))/norm_const[i]
  var2_sum <- var2_sum + prob_cl2*(x[i] - means[2])^2
}

# Updated variances
vars <- c(var1_sum/nj1, var2_sum/nj2)