import os, json, glob, subprocess
from collections import defaultdict
import numpy as np


passed = []
not_passed = []

for x in glob.glob("./prob*-*"):
	if os.path.exists(os.path.join(x, "error_localize_edit.txt")):
		unique_id = os.path.basename(x)
		if os.path.exists(os.path.join(x, 'compiled.txt')):
			passed.append(x)
		else:
			not_passed.append(x)

total_num_examples = len(passed) + len(not_passed)
print ("#passed: {}  #failed: {}".format(len(passed), len(not_passed)))
print ("acc:\t{:.3f}".format((len(passed) + 0.) / total_num_examples))
