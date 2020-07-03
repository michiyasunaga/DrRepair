import os, json, random
from collections import defaultdict
random.seed(619341)


total_dpoints = 2000
shuffled_dev_path = "err-dev-compiler--from-SPoC/err-dev.all.shuffled.jsonl"
out_path = "err-dev-compiler--from-SPoC/err-dev.{}.jsonl".format(total_dpoints)



probid_dict = defaultdict(int)
d_points = []
with open(shuffled_dev_path, "r") as inf:
    for line in inf:
        d_point = json.loads(line)
        d_points.append(d_point)
        probid = d_point["info"]["probid"]
        probid_dict[probid] += 1

num_probids = len(probid_dict)
print ("# of distinct probids", num_probids)

subset_data = {}
for num_mod in range(1,5): #1,2,3,4
    subset_data[num_mod] = {probid: 0 for probid in probid_dict}
cap = (total_dpoints // (4 * num_probids)) + 1

to_dump = []
with open(out_path, "w") as outf:
    count = 0
    for d_point in d_points:
        probid = d_point["info"]["probid"]
        num_mod = sum(d_point["gold_linenos"])
        if subset_data[num_mod][probid] < cap:
            subset_data[num_mod][probid] += 1
            x = json.dumps(d_point)
            to_dump.append(x)
        count += 1
        if count % 500 == 0:
            print ([sum(subset_data[num_mod].values()) for num_mod in subset_data])
    random.shuffle(to_dump)
    for x in to_dump:
        print(x, file=outf)
