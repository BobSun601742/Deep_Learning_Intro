# June 2nd, Diary 3

## Summary:
In today's learning, we completed several complicated tasks:
* Understand mechanism of "Back Propagation"
* Learn how to construct some particular layers' forward and back propagation form
* Improve the TwoLayerNet we completed in last study
* Learn methods of checking accuracy of learning

## Back Propagation Comprehension:
### Review of Forward Propagation
Let us make an assumption. We now have:
$$
X = 
\begin{bmatrix}
    x_{11} & x_{12} & ... & x_{1D}\\
    x_{21} & x_{22} & ... & x_{2D}\\
    ... & ... & ... & ... \\
    x_{N1} & x_{N2} & ... & x_{ND}
\end{bmatrix}
$$

$$
W = 
\begin{bmatrix}
    w_{11} & w_{12} & ... & w_{1M}\\
    w_{21} & w_{22} & ... & w_{2M} \\
    ... & ... & ... & ... \\
    w_{D1} & w_{D2} & ... & w_{DM}
\end{bmatrix} 
$$
$$
b =
\begin{bmatrix}
    b_{11} & b_{12} & ... & b_{1M}\\
    b_{21} & b_{22} & ... & b_{2M}\\
    ... & ... & ... & ... \\
    b_{N1} & b_{N2} & ... & b_{NM} 
\end{bmatrix}
$$

The __Affine Tranformation__ of X is:
$$
A = 
XW+b
$$
in which:
$$
A_{ij} = \sum_{k=1}^{D} x_{ik} w_{kj} + b_{ij}
$$
and after the __Activation Layer__ (e.g.: Relu, Sigmoid, Softmax...):
$$
Y = f(A)
$$
and __Objective (Loss) Layer__ (e.g.: Cross Entropy Error, Least Square Error...):
$$
L = Loss(Y)
$$
We get the objective __"Loss"__ we wish to minimize.

### Goal of Back Propagation:

Now I'm going to show the workflow of __Back Propagation__:

We want to change parameters in __W__ and __b__, which will be implemented by __gradient descent__ learned in last chapter:
$$
W_{new} = W - \eta \frac{\partial L}{\partial W} \\
\\[10pt]
b_{new} = W - \eta \frac{\partial L}{\partial b}
$$

Imagine that we acquire the gradient matrix using __middle differentiation method__ we learned in last chapter:
$$
\frac{\partial L}{\partial Y} =
\begin{bmatrix}
    g_{11} & g_{12} & ... & g_{1M} \\
    g_{21} & g_{22} & ... & g_{2M} \\
    ... & ... & ... & ... \\
    g_{N1} & g_{N2} & ... & g_{NM} 
\end{bmatrix}
$$
Using __Chain Rule__, we get a path to compute the two gradient matrixs we need:
$$
\frac{\partial L}{\partial W} = \frac{\partial L}{\partial Y} \frac{\partial Y}{\partial A} \frac{\partial A}{\partial W}
\\[10pt]
\frac{\partial L}{\partial b} = \frac{\partial L}{\partial Y} \frac{\partial Y}{\partial A} \frac{\partial A}{\partial b}
$$
in which
$$
\frac{\partial Y}{\partial A}
$$
is __Activation Layer__;
$$
\frac{\partial A}{\partial W}
$$
is __Multiplcation Layer__;
$$
\frac{\partial A}{\partial b}
$$
is __Addition Layer__. 

### Back Propagation of Activation Layer:
For A ∈ R(N×M), we have:
$$
Y_{ij} = f(A_{ij})
$$
which means that each element of output Y is only dependent on the element of input A in the corresponding position. 

Since we know the gradient of Loss to Y for each element:
$$
dY_{ij} = \frac{\partial L}{\partial Y_{ij}}
$$
and we want
$$
dA_{ij} = \frac{\partial L}{\partial A_{ij}}
$$
, so we use __Chain Rule__:
$$
\frac{\partial L}{\partial A_{ij}} = \sum_{p=1}^{N} \sum_{q=1}^{M} \frac{\partial L}{\partial Y_{pq}} \frac{\partial Y_{pq}}{\partial A_{ij}}
$$
As we know, Y(pq) is only dependent on A(pq), so only when
$$
p = i, q = j
$$
, Y(pq) is dependent on A(ij), which means that
$$
\frac{\partial Y_{pq}}{\partial A_{ij}} = 0
\\[10pt]k
(p≠i, q≠j)
$$
. And, when (p,q) = (i, j):
$$
\frac{\partial Y_{ij}}{\partial A_{ij}} = f'(A_{ij})
$$
. In this way, we get:
$$
\frac{\partial L}{\partial A_{ij}} = \frac{\partial L}{\partial Y_{ij}} * f'(A_{ij})
$$
which is:
$$
dA_{ij} = dY_{ij}f'(A_{ij})
$$
. Written in matrix form, this is:
$$
dA = dY \odot f'(A)
$$
This form is really good! __"odot"__ is __Element-Wise Multiplication__, which is the default multiplication in numpy!

### Back Propagation of Multiplication Layer:
__Part I:__

Imagine that we have:
$$
A = XW
\\[10pt]
X∈R^{N×D}, W∈R^{D×M}
\\[10pt]
A∈R^{N×M}
$$
in which:
$$
A_{ij} = \sum_{k=1}^{D} X_{ik} W_{kj}
$$
. Now in __back propagation__, we have already had:
$$
dA = \frac{\partial L}{\partial A}
\\[10pt]
dA_{ij} = \frac{\partial L}{\partial A_{ij}}
$$
, and our goal is to calculate:
$$
dW = \frac{\partial L}{\partial W}
\\[10pt]
dX = \frac{\partial L}{\partial X}
$$
Let's first concentrate on dW. This means that we should calculate every element:
$$
\frac{\partial L}{\partial W_{pq}}
$$
Using __Chain Rule__, this is:
$$
\frac{\partial L}{\partial W_{pq}} = \sum_{i=1}^{N} \sum_{j=1}^{M} \frac{\partial L}{\partial A_{ij}} \frac{\partial A_{ij}}{\partial W_{pq}}
$$
As we know, 
$$
A_{ij} = \sum_{k=1}^{D} X_{ik} W_{kj}
$$
, so only when:
$$
k=p, j=q
$$
Wpq will appear. 

The corresponding multiplication becomes:
$$
X_{ip}W_{pq}
$$

This means that:
$$
\frac{\partial A_{ij}}{\partial W_{pq}}=
\begin{cases}
X_{ip}, & j=q \\
0, & j≠q
\end{cases}
$$
therefore, according to multiplication rule of constants:
$$
\frac{\partial L}{\partial W_{pq}} = \sum_{i=1}^{N} dA_{iq}X_{ip} = \sum_{i=1}^{N} X_{ip}dA_{iq}
$$
Notice that:
$$
\sum_{i=1}^{N} X_{ip}dA_{iq} = \sum_{i=1}^{N} X_{pi}^TdA_{iq} = (X^TdA)_{pq}
$$
we eventually get:
$$
dW = X^T dA
$$
so
$$
dW = X^T (dY \odot f'(A))
$$

__Part II:__

We want:
$$
\frac{\partial L}{\partial X_{pq}}
$$
so:
$$
\frac{\partial L}{\partial X_{pq}} = \sum_{i=1}^{N} \sum_{j=1}^{M} \frac{\partial L}{\partial A_{ij}} \frac{\partial A_{ij}}{\partial X_{pq}}
$$
Since:
$$
A_{ij} = \sum_{k=1}^{D} X_{ik}W_{kj}
$$
only when:
$$
i=p, k=q
$$
will they mutually influenced, so the corresponding multiplication becomes:
$$
X_{pq}W_{qj}
$$
so:
$$
\frac{\partial A_{ij}}{\partial X_{pq}}=
\begin{cases}
W_{qj}, & i=p \\
0, & i≠p
\end{cases}
$$
Take it back to original equation:
$$
\frac{\partial L}{\partial X_{pq}} = \sum_{j=1}^{M} dA_{pj}W_{qj}
$$

Notice that:
$$
\sum_{j=1}^{M} dA_{pj}W_{qj} = \sum_{j=1}^{M} dA_{pj}W_{jq}^T = (dAW^T)_{pq}
$$
so:
$$
dX=dAW^T
$$
which could be extended as:
$$
dX = (dY \odot f'(A))W^T
$$

__Part III: Conclusion__
$$
dW = X^T(dY \odot f'(A))
\\[10pt]
dX = (dY \odot f'(A))W^T
$$

### Back Propagation of Addition Layer:
We have:
$$
A = XW + b
\\[10pt]
X∈R^{N×D}, W∈R^{D×M}
\\[10pt]
b∈R^{1×M}
\\[10pt]
A∈R^{N×M}
$$
(Why b ∈ R(1×M)? This is because in order to __prevent over-fitting__, for every column, we have the same offset bj for every element in every row.)

According to __Chain Rule__:
$$
\frac{\partial L}{\partial b_j} = \sum_{i=1}^{N}\frac{\partial L}{\partial A_{ij}} = \sum_{i=1}^{N}dA_{ij}
$$