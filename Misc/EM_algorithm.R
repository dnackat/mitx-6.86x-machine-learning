# Toy EM Algorithm: 1-D data

means <- c(-3, 2)
vars <- c(4, 4)
probs <- c(0.5, 0.5)

x <- c(0.2, -0.9, -1, 1.2, 1.8)

### E-step: Bayes calc ###
cl_p <- matrix(data = 0, nrow = length(x), ncol = length(means)) # Matrix to store cluster probs

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
    v[j] <- v[j] + cl_p[i,j]*(x[i] - means)^2
  }
}
vars <- (1/(nj*d))*v

print("Revised prob. =")
print(probs)
print("Revised mean =")
print(means)
print("Revised variance =")
print(vars)