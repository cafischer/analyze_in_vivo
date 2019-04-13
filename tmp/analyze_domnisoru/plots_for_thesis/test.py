import matplotlib.pyplot as pl
import numpy as np

fig, axs = pl.subplots(1,1)
h = axs.scatter(0, 0, marker='^', s=500, linewidths=0.8, hatch='+++', edgecolor='r', facecolor='None', label='red')
leg = pl.legend()
leg.legendHandles[0].set_edgecolor('r')
pl.show()

