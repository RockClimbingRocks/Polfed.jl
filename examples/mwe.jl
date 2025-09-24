# # Conditional Maximum Likelihood for the Rasch Model
#
#-
#md # !!! tip
#md #     This example is also available as a Jupyter notebook:
#md #     bla bla bla (here it was a reference to some notebook )
#-
using Polfed #hide
#
# The Rasch model is used in psychometrics as a model for
# assessment data such as student responses to a standardized
# test. Let $X_{pi}$ be the response accuracy of student $p$
# to item $i$ where $X_{pi}=1$ if the item was answered correctly
# and $X_{pi}=0$ otherwise for $p=1,\ldots,n$ and $i=1,\ldots,m$.
# The model for this accuracy is
# ```math
#   P(\mathbf{X}_{p}=\mathbf{x}_{p}|\xi_p, \mathbf\epsilon) = \prod_{i=1}^m \dfrac{(\xi_p \epsilon_j)^{x_{pi}}}{1 + \xi_p\epsilon_i}
# ```
# where $\xi_p > 0$ the latent ability of person $p$ and $\epsilon_i > 0$
# is the difficulty of item $i$.

# We simulate data from this model:

n = 1000
m = 5
theta = randn(n)
delta = randn(m)
r = zeros(n)
s = zeros(m)


f = [sum(r .== j) for j = 1:m];