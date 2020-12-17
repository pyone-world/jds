import numpy as np
import scipy.stats

x = np.array([15.0, 12.0, 8.0, 8.0, 7.0, 7.0, 7.0, 6.0, 5.0, 3.0])
y = np.array([10.0, 25.0, 17.0, 11.0, 13.0, 17.0, 20.0, 13.0, 9.0, 15.0])

result = scipy.stats.pearsonr(x, y)[0]
print(round(result, 3))
