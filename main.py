import numpy as np
import matplotlib.pyplot as plt
from eztao.carma import DRW_term
from eztao.ts import gpSimRand
amp = 0.2
tau = 100
DRW_kernel = DRW_term(np.log(amp), np.log(tau))
t, y, yerr = gpSimRand(DRW_kernel, 10, 365*10, 200)
fig, ax = plt.subplots(1, 1, dpi=150, figsize=(8, 3))
ax.errorbar(t, y, yerr, fmt='.')
#plot.show()