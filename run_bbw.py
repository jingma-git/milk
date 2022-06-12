import numpy as np
import os

model_list = np.loadtxt("/mnt/f/Dataset/RigNetv1/test_final.txt", dtype=np.int)
for model_id in model_list:
    print("model_id", model_id)
    os.system("./build/run_bbw {}".format(model_id))