# Math demo

This file demonstrates inline and display LaTeX math for Markdown viewers that support KaTeX or MathJax (for example, the VS Code extension "Markdown Preview Enhanced").

## Inline math

Euler's identity: $e^{i\pi} + 1 = 0$

## Display math

Here is a definite integral:

$$
\int_0^1 x^2 \, dx = \frac{1}{3}
$$

## More examples

- Summation:

$$
\sum_{n=1}^\infty \frac{1}{n^2} = \frac{\pi^2}{6}
$$

- A matrix:

$$
\begin{bmatrix}
1 & 0 \\
0 & 1
\end{bmatrix}
$$

## View instructions

- Install the VS Code extension:

```bash
code --install-extension shd101wyy.markdown-preview-enhanced
```

- Open this file and use the command `Markdown Preview Enhanced: Open Preview`.

If you want me to add KaTeX rendering to a documentation site (MkDocs, Hugo, or GitHub Pages), tell me which and I'll add the minimal configuration.
