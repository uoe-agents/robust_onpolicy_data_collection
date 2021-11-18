import os
import re
from collections import defaultdict

cmd = os.popen('qstat')
# cmd = os.popen('squeue')

content = cmd.read()
cmd.close()

total = 0
jobs = defaultdict(lambda: defaultdict(int))
for line in content.split('\n')[1:-1]:
    line = re.split(r'[ ]+', line)
    job_name = line[3]
    state = line[5]
    jobs[job_name][state] += 1
    total += 1

[print(i) for i in jobs.items()]
print(total)
