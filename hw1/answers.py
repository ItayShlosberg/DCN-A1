r"""
Use this module to write your answers to the questions in the notebook.

Note: Inside the answer strings you can use Markdown format and also LaTeX
math (delimited with $$).
"""

# ==============
# Part 2 answers

part2_q1 = r"""
**Your answer:**

More neighbors smooths out local weirdness, which is what causes lack of generalization.
The more smoothing, the less local "hiccups" in the model, and the more generalization.

And,if we choose K = data set size, then our algorithm behaves as
underfitting and it gives a smooth decision surface and everything
becomes one class which is the majority class in our DataSet.

So, we should choose K wisely such that it should neither be overfitting nor be underfitting .

"""
# ==============

# ==============
# Part 3 answers

part3_q1 = r"""
**Your answer:**

The hyperparameters $Δ$ and $λ$ may seem like two different hyperparameters, but in fact they both control the same tradeoff: The tradeoff between the data loss and the regularization loss in the objective. Clearly, the magnitude of the weights $W$ has direct effect on the scores: As we shrink all values inside W the score differences will become lower, and as we scale up the weights the score differences will all become higher. 

Namely, the exact value of the margin $Δ$ is arbitrary because the weights can shrink or stretch the differences arbitrarily. Hence, the only real tradeoff is how large we allow the weights to grow, through the regularization strength $λ$. This gives us the power to arbitrarily select $Δ$.

"""

part3_q2 = r"""
**Your answer:**
We can interpret the reshaped image weights as patterns of the digits 0-9.
The matrix multiplication behaves as an inner product between the inputs X with those 
patterns, then extract the label of the pattern that produced the maximum value.
"""

part3_q3 = r"""
**Your answer:**
1. The learning rate is good, since the loss converges smoothly and relatively quickly. If the learning 
rate was too high, the graph would be noisier and perhaps wouldn't even converge, while a low learning rate would cause
a very slow convergence.

2. We can see there is a constant gap between the training and validation accuracy, the training one being higher.
This hints that the model is slightly overfitted over the training data. (The model has a bit of a hard time generalizing to the validation set - High variance)

"""

# ==============

# ==============
# Part 4 answers

part4_q1 = r"""
The ideal residual plot would have both dashed lines merge in the middle with the zero line.

THe best model did better on both validation and test set using cross validation, than the model
generated with the top 5 Boston features. This is easily visuallized by the fact the dashed lines are closer
to the zero lines and the data cloud's spread is closer to the zero line.

"""

part4_q2 = r"""

1. Using logspace lets us sample a wider range of values for $λ$, iterating over various orders of magnitude.
Had we used linspace we had to use many more samples, resulting in a much slower process,
or have the samples more sparse, resulting in a less representetive search for the smaller orders of magnitude.

2. The model was fitted #degrees x #lambdas x #folds times, so 3 x 20 x 3 = 180 times. 

"""

# ==============
