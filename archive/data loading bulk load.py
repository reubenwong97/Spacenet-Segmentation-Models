def generate_train_test():
    paths = data_paths()
    data = [[], [], [], []]

    for index, path in tqdm(enumerate(paths), total=len(paths)):
        fnames = get_fnames(path)
        
        # for fname in tqdm(fnames[:16], total=len(fnames[:16])):
        for fname in tqdm(fnames, total=len(fnames)):
            npy = rebuild_npy(path / fname)
            data[index].append(npy)

        data[index] = np.array(data[index])
    
    X_train, Y_train, X_test, Y_test = data[0], data[1], data[2], data[3]
    
    return (X_train, Y_train, X_test, Y_test)