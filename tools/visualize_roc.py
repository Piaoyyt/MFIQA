import  matplotlib.pyplot as plt


file_name = r"record of tp and fp.txt"
tp_list = []
fp_list = []
with open(file_name, 'r') as f:
    next(f)
    for line in f.readlines():
        tp_list.append(float(line.split(' ')[0]))
        fp_list.append(float(line.split(' ')[-1]))
print(fp_list)
print((tp_list))
plt.plot(fp_list,tp_list)
plt.show()