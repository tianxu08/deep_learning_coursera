import h5py as h5

if __name__ == "__main__":
    f = h5.File('./train_signs_1.h5', 'r')
    
    print(list(f.keys()))
