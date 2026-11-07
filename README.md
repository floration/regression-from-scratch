Linear regression from scratch, trained with hand derived gradient descent in raw numpy.

Formulas:
e = (wx + b - y)
change in loss per change in weights = 1/n * sum(2 * e * x)
change in loss per change in biases  = 1/n * sum(2 * e)

Results:
Using a dataset where y = wx + b - y; where w, and b are given variables 3 and 2, and y is a random float from 0 to 1
100 steps of gradient descent learned that w should be ~2.97 and b should be 1.92
this is because of the seed value for the random numbers, and will change if a different seed is used

Images:
[loss curve](loss.png)
[fit](fit.png)

How to Run:
$ uv run regression.py
$ uv run pytest

What I Learned:
Matplot graphing and images
Gradient exploding
Scalar noise adds a loss floor
