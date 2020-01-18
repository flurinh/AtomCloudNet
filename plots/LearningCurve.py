import matplotlib.pyplot as plt
import numpy as np

"""
AtomCloudModels were run and evaluated in configuration settings (corresponds to config_file id):
To get the values below follow instructions on github to train and evaluate a model with "sandbox.py" and "results.py",
they were manually added here.
 
cloud radius        10k samples         1k samples      100 samples

3                   13003               15005           15005
4                   15002               15009           15007
5                   15003               15010           15008

"""

CM = [3856.8972569057287, 509.44264298281587, 253.58362807398674, 102.24166779382695, 50.8236553465873]
SLATM = [541.7630338973415, 14.546501865038607, 7.007622737762912, 2.495302292965314, 1.2828617615041113]
Traning = [100, 1000, 2000, 5000, 10000]
ACN3 = [0.04549915399112006, 0.017281286622418325, 0.0005205715925826579]
s3 = [x * (630 * 400) for x in ACN3]
ACN4 = [0.062323461389267865, 0.01897340647880035, 0.0029883508606623993]
s4 = [x * (630 * 400) for x in ACN4]
ACN5 = [0.06332461197562213,  0.06448201962395173, 0.004551845370080491]
s5 = [x * (630 * 400) for x in ACN5]

plt.figure(figsize=(15, 15))
plt.xlabel("Training set size", fontsize=24)
plt.ylabel("MAE[kcal/mol]", fontsize=24)
plt.tick_params(axis='x', labelsize=26)
plt.tick_params(axis='y', labelsize=26)
plt.loglog([100, 1000, 10000], s3, 'b-o', label = "AtomCloudNet (Ours), r = 3")
plt.loglog([100, 1000, 10000], s4, 'g-o', label = "AtomCloudNet (Ours), r = 4")
plt.loglog([100, 1000, 10000], s5, 'c-o', label = "AtomCloudNet (Ours), r = 5")
plt.loglog([100, 1000, 2000, 5000, 10000], CM, 'r-o', label = "CM (Base line)")
plt.loglog(Traning, SLATM, 'k-o', label = "SLATM (State of the art)")
plt.legend(loc='best', fontsize=24)
plt.savefig('Learning_Curve_Comparision.png')