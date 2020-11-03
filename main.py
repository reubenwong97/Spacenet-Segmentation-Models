import utils.helper as helper

X_train, Y_train, X_test, Y_test = helper.generate_train_test()

index = 8
helper.plot_img_mask(index, X_train[index], Y_train[index])