import numpy as np
from matplotlib_inline import backend_inline
from d2l import torch as d2l
def f(x):
    return 3 * x ** 2 - 4 * x

x = d2l.arange(-10,10,0.1,requires_grad=True)
y = d2l.sin(x)
y.sum().backward()
d2l.plot(x.detach().numpy(),
         [y.detach().numpy(), x.grad.detach().numpy()],
         'x', 'y',
         legend=['sin(x)', "导数 cos(x)"])
d2l.plt.show()