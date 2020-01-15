import matplotlib.pyplot as plt
import numpy as np


CM = [3856.8972569057287, 509.44264298281587, 253.58362807398674, 102.24166779382695, 50.82365534658731]
SLATM = [541.7630338973415, 14.546501865038607, 7.007622737762912, 3.495302292965314, 2.2828617615041113]
Traning = [100, 1000, 2000, 5000, 10000]
StdCM = [0.15217299649321853, 0.01623470576872112, 0.005203141096532805 , 0.0014784175028615585, 0]
StdSLATM = [0.27832101126536674, 0.0019484438815888065, 0.000803575696966312, 0.000033652891644, 0]
StdCM = [i * 630 for i in StdCM]
StdSLATM = [i * 630 for i in StdSLATM]
Atm = [0.40411114798679526 * 630]



plt.xlabel("Training set size", fontsize=14)
plt.ylabel("MAE[kcal/mol]", fontsize=14)
plt.tick_params(axis='x', labelsize=16)
plt.tick_params(axis='y', labelsize=16)
plt.loglog([10000], Atm, 'g-o', label = "AtomCloudNet")
plt.loglog(Traning, CM, 'r-o', label = "CM (Base line)")
plt.loglog(Traning, SLATM, 'k-o', label = "SLATM (State of the art)")
plt.title("machine learning atomization energies of QM9", fontsize=18)
plt.legend(loc='best')
plt.show()
