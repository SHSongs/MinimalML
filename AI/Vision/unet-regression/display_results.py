##
import numpy as np
import matplotlib.pyplot as plt
import os
##
result_dir = './results/numpy'

lst_data = os.listdir(result_dir)
 
lst_label = [f for f in lst_data if f.startswith('label')]
lst_input = [f for f in lst_data if f.startswith('input')]
lst_output = [f for f in lst_data if f.startswith('output')]


lst_label.sort()
lst_input.sort()
lst_output.sort()

##
id = 0

label = np.load(os.path.join(result_dir, lst_label[id]))
input = np.load(os.path.join(result_dir, lst_input[id]))
output = np.load(os.path.join(result_dir, lst_output[id]))

##
plt.subplot(131)
plt.imshow(label)
plt.title('label')

plt.subplot(132)
plt.imshow(input)
plt.title('input')

plt.subplot(133)
plt.imshow(output)
plt.title('output')

plt.show()
