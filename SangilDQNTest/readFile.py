import pickle

name = "202010281557-test.p"
with open('portfolio_val/' + name, 'rb') as file:
    portfol_val = pickle.load(file)
    #print(portfol_val)

#for vals in portfol_val:
#    print(vals[0], vals[1])
max_v = portfol_val[0]
index = 0
for i in range(len(portfol_val)) :
    if portfol_val[i] > max_v :
         max_v = portfol_val[i]
         index = i

max_val = max(portfol_val)
min_val = min(portfol_val)
print("The Best Earned: ", max_val, max_v, index)
print("The Worst Earned: ", min_val) 

# import h5py

# filename = 'weights/202010150814-dqn.h5'

# h5 = h5py.File(filename, 'r')

# print(h5)

# h5.close()