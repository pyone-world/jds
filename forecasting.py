import numpy as np
from sklearn import svm

months = []
npgrs = []

X_test = [61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71, 72]

input_case = open('forecasting-passenger-traffic-testcases/input/input00.txt').read().splitlines()

num = int(input_case[0])
for line in input_case[1:]:
    m, npg = line.split('\t')
    mo, i = m.split('_')
    months.append(int(i))
    npgrs.append(int(npg))


clf = svm.SVR(kernel='rbf', C=1000, gamma=0.1)

X_m = np.array(months)
Y_n = np.array(npgrs)

X_m = X_m.reshape(X_m.shape[0], 1)

clf.fit(X_m, Y_n)

X_test = np.array(X_test)
X_test = X_test.reshape(X_test.shape[0], 1)
Y_pred = clf.predict(X_test)
for yps in Y_pred:
    print(int(yps))
