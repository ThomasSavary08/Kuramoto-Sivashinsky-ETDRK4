# One dimensional Kuramoto-Sivashinsky ETDRK4 solver

```math
\left\{
    \begin{array}{ll}
        \dot{x} = f(x,t) \\
        x_{n+1} = f(x_{n},n)
    \end{array}
\right. 
```
* the jacobian of $f$ with respect to $x$ or $x_{n}$:
```math
\left\{
    \begin{array}{ll}
        J(x,t) = \displaystyle \frac{\partial f}{\partial x}(x,t) \\
        ~ \\
        J(x_{n},n) = \displaystyle \frac{\partial f}{\partial x_{n}}(x_{n},n)
    \end{array}
\right. 
```
