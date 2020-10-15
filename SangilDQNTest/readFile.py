import pickle

with open('portfolio_val/202010150814-train.p', 'rb') as file:
    portfolo_val = pickle.load(file)
    print(portfolo_val)

# import h5py

# filename = 'weights/202010150814-dqn.h5'

# h5 = h5py.File(filename, 'r')

# print(h5)

# h5.close()