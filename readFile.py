import pickle

name = "102906-b64-e2000-train.p"
with open('portfolio_val/' + name, 'rb') as file:
    portfol_val = pickle.load(file)
    #print(portfol_val)

# Find Maximum Value and Index
max_v = portfol_val[0][1]
index = 0
i = 0
for vals in portfol_val:
    if vals[1] > max_v:
        max_v = vals[1]
        index = i
    i += 1
    #print(vals[0], vals[1])

print("Maximum Action & Value & index")
print(portfol_val[index][0], max_v, index)
'''
이전 데이터 저장 형식 (value만 저장)
max_v = portfol_val[0]
index = 0
for i in range(len(portfol_val)) :
    if portfol_val[i] > max_v :
         max_v = portfol_val[i]
         index = i
'''

# import h5py

# filename = 'weights/202010150814-dqn.h5'

# h5 = h5py.File(filename, 'r')

# print(h5)

# h5.close()