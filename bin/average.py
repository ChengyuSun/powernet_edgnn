import numpy as np
accs=[]
accs= np.loadtxt('./acc_file.txt', dtype=np.float, delimiter=',')
print(len(accs))
print(sum(accs)/len(accs))