def CapLabels():
    # Simple transformation to reduce the range of the resid because we only 
    # care about ordinal labels at the end
    # if heat_resid < 0 , then heat_resid = 0
    # if heat_resid > 300 , then heat_resid = 300
    for i in range (y_train.shape[0]):
        if y_train[i] < 0:
            y_train[i] = 0
        elif y_train[i] > 300:
            y_train[i] = 300

    print (y_train.max(), y_train.min())

def Normalize (y):
    # normalize binned_y
    min_val = y.min()
    max_val = y.max()
    range_val = max_val - min_val

    for i in range(y.shape[0]):
        norm_y[i] = (y[i] - min_val) / range_val
    
    return norm_y

#print (norm_y)
# initialize norm_y
norm_y = np.zeros_like(y_train, dtype=np.float32)
norm_y = norm_y.astype(np.float16)

def Denorm (y, max, min):

    denorm_y = np.zeros_like(y)
    range_val = max - min

    for i in range(y.shape[0]):
        denorm_y[i] = y[i] * range_val + min
    
    return denorm_y[i]

#---------- Get Data---------------------------------------
# Open training data
file = h5py.File("../mp02files/DEM_train.h5", "r+")
X_train = np.array(file["/images"])
y_train = np.array(file["/meta"])
file.close()

# Open test data
file = h5py.File("../mp02files/DEM_test_features.h5", "r+")
X_test = np.array(file["/images"])
file.close()

print (X_train.shape, y_train.shape, X_test.shape)
print (f'y_train_max {y_train.max()}, y_train_min {y_train.min()}')

# resize binned_y to make it a 2D matrix
y_train = y_train.reshape(-1,1)
print (y_train.shape)

#-------------------------------------------------------------

#----------Split Training Data-------------------------------
train_in, val_in, train_y, val_y = train_test_split(X_train, norm_y, test_size=0.20)

print (train_in.shape, train_y.shape, val_in.shape, val_y.shape)
#------------------------------------------------------------