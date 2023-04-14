# Python cookbook

## Random Matrices

```
>>> import numpy as np
>>> np.random.randn(3,2)
array([[ 0.22913881,  1.07459504],
       [ 1.63070903, -0.89951504],
       [-0.32183631,  1.1027342 ]])
```

**Rescale**

```
>>> np.random.randn(3,2)*0.01
array([[0.0024564 , 0.02005188],
       [0.02137384, 0.01854684],
       [0.00374551, 0.01174697]])
```