# June 2nd, Diary 3

## Summary
In today's learning, we completed several complicated tasks:

* Understand mechanism of "Back Propagation"
* Learn how to construct some particular layers' forward and back propagation form
* Improve the TwoLayerNet we completed in last study
* Learn methods of checking accuracy of learning

## Back Propagation Comprehension

### Review of Forward Propagation

Let us make an assumption. We now have:

$$
X = 
\begin{bmatrix}
x_{11} & x_{12} & \cdots & x_{1D}\\
x_{21} & x_{22} & \cdots & x_{2D}\\
\vdots & \vdots & \ddots & \vdots\\
x_{N1} & x_{N2} & \cdots & x_{ND}
\end{bmatrix}
$$

$$
W = 
\begin{bmatrix}
w_{11} & w_{12} & \cdots & w_{1M}\\
w_{21} & w_{22} & \cdots & w_{2M}\\
\vdots & \vdots & \ddots & \vdots\\
w_{D1} & w_{D2} & \cdots & w_{DM}
\end{bmatrix}
$$

$$
b =
\begin{bmatrix}
b_1 & b_2 & \cdots & b_M
\end{bmatrix}
$$

The **Affine Transformation** of X is:

$$
A = XW + b
$$

in which:

$$
A_{ij} = \sum_{k=1}^{D} X_{ik} W_{kj} + b_j
$$

After the **Activation Layer** (e.g.: ReLU, Sigmoid, Softmax):

$$
Y = f(A)
$$

and **Objective (Loss) Layer** (e.g.: Cross Entropy, Least Square):

$$
L = Loss(Y)
$$

We get the objective "Loss" we wish to minimize.

### Goal of Back Propagation

We want to update parameters W and b using gradient descent:

$$
W_{new} = W - \eta \frac{\partial L}{\partial W}
$$
$$
b_{new} = b - \eta \frac{\partial L}{\partial b}
$$

Assume we have obtained the upstream gradient from the Loss layer:

$$
\frac{\partial L}{\partial Y} =
\begin{bmatrix}
g_{11} & g_{12} & \cdots & g_{1M}\\
g_{21} & g_{22} & \cdots & g_{2M}\\
\vdots & \vdots & \ddots & \vdots\\
g_{N1} & g_{N2} & \cdots & g_{NM}
\end{bmatrix}
$$

Using the Chain Rule, we can compute the gradients w.r.t W and b:

$$
\frac{\partial L}{\partial W} = \frac{\partial L}{\partial Y} \frac{\partial Y}{\partial A} \frac{\partial A}{\partial W}
$$
$$
\frac{\partial L}{\partial b} = \frac{\partial L}{\partial Y} \frac{\partial Y}{\partial A} \frac{\partial A}{\partial b}
$$

where:

- $\frac{\partial Y}{\partial A}$ corresponds to the Activation Layer.
- $\frac{\partial A}{\partial W}$ corresponds to the Multiplication Layer.
- $\frac{\partial A}{\partial b}$ corresponds to the Addition Layer.

### Back Propagation of Activation Layer

For A ∈ R(N×M), we have:

$$
Y_{ij} = f(A_{ij})
$$

which means that each element of output Y depends only on the corresponding element of input A.

Since we know the gradient of Loss to Y for each element:

$$
dY_{ij} = \frac{\partial L}{\partial Y_{ij}}
$$

and we want:

$$
dA_{ij} = \frac{\partial L}{\partial A_{ij}}
$$

Using Chain Rule:

$$
\frac{\partial L}{\partial A_{ij}} = \sum_{p=1}^{N} \sum_{q=1}^{M} \frac{\partial L}{\partial Y_{pq}} \frac{\partial Y_{pq}}{\partial A_{ij}}
$$

Since Y_{pq} depends only on A_{pq}, we have:

$$
\frac{\partial Y_{pq}}{\partial A_{ij}} = 0, \quad (p,q) \ne (i,j)
$$

and when (p,q) = (i,j):

$$
\frac{\partial Y_{ij}}{\partial A_{ij}} = f'(A_{ij})
$$

Thus:

$$
\frac{\partial L}{\partial A_{ij}} = \frac{\partial L}{\partial Y_{ij}} f'(A_{ij})
$$

In matrix form:

$$
dA = dY \odot f'(A)
$$

**Note:** \(\odot\) denotes element-wise multiplication (like NumPy `*` operator).

### Back Propagation of Multiplication Layer

Assume:

$$
A = X W
$$

$$
X \in R^{N\times D}, W \in R^{D\times M}
$$

$$
A \in R^{N\times M}
$$

Each element:

$$
A_{ij} = \sum_{k=1}^{D} X_{ik} W_{kj}
$$

Given upstream gradient:

$$
dA = \frac{\partial L}{\partial A}, \quad dA_{ij} = \frac{\partial L}{\partial A_{ij}}
$$

#### Gradient w.r.t W

For each W_{pq}:

$$
\frac{\partial L}{\partial W_{pq}} = \sum_{i=1}^{N} \sum_{j=1}^{M} dA_{ij} \frac{\partial A_{ij}}{\partial W_{pq}}
$$

Since A_{ij} depends on W_{pq} only when k=p, j=q:

$$
\frac{\partial A_{ij}}{\partial W_{pq}} = 
\begin{cases}
X_{ip}, & j=q \\
0, & j\ne q
\end{cases}
$$

Thus:

$$
\frac{\partial L}{\partial W_{pq}} = \sum_{i=1}^{N} X_{ip} dA_{iq}
$$

In matrix form:

$$
dW = X^T dA
$$

#### Gradient w.r.t X

Similarly:

$$
\frac{\partial L}{\partial X_{pq}} = \sum_{i=1}^{N} \sum_{j=1}^{M} dA_{ij} \frac{\partial A_{ij}}{\partial X_{pq}}
$$

Where:

$$
\frac{\partial A_{ij}}{\partial X_{pq}} = 
\begin{cases}
W_{qj}, & i=p \\
0, & i\ne p
\end{cases}
$$

Hence:

$$
\frac{\partial L}{\partial X_{pq}} = \sum_{j=1}^{M} dA_{pj} W_{qj}
$$

Matrix form:

$$
dX = dA W^T
$$

Combined with activation layer:

$$
dX = (dY \odot f'(A)) W^T
$$

### Back Propagation of Addition Layer

Assume:

$$
A = XW + b
$$
$$
X \in R^{N\times D}, W \in R^{D\times M}
$$

$$
b \in R^{1\times M}
$$

$$
A \in R^{N\times M}
$$

The gradient w.r.t bias:

$$
\frac{\partial L}{\partial b_j} = \sum_{i=1}^{N} dA_{ij}
$$

This gives a 1×M vector, broadcasting over the batch in forward propagation.