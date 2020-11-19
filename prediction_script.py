
# Load in the test set
PATH_TRAIN_IMG, PATH_TRAIN_MASK, PATH_TEST_IMG, PATH_TEST_MASK = helper.data_paths()

X_test_fnames = helper.get_fnames(PATH_TEST_IMG)
Y_test_fnames = helper.get_fnames(PATH_TEST_MASK)

# slice it for speed reasons 
X_test_fnames = X_test_fnames[:2]
Y_test_fnames = Y_test_fnames[:2]

X_test, Y_test = [], []

for fname in X_test_fnames:
    X_test.append(helper.rebuild_npy(PATH_TEST_IMG/fname))
    
for fname in Y_test_fnames:
    Y_test.append(helper.rebuild_npy(PATH_TEST_MASK / fname))
    
X_test = np.array(X_test)
Y_test = np.array(Y_test)

X_test = tf.dtypes.cast(X_test, tf.dtypes.float32)
Y_test = tf.dtypes.cast(Y_test, tf.dtypes.float32)



# Recreate the model here - copy the desired model code

model = sm.Unet(BACKBONE, encoder_weights='imagenet', input_shape=(None, None, 3))
model.compile(
    optimizer='adam',
    loss=sm.losses.BinaryFocalLoss(alpha=0.75, gamma=0.25),
    metrics=[sm.metrics.IOUScore()],
)


# Load in weights for the recreated model
model.load_weights(str(PATH_CHECKPOINTS / 'architecture_trial_resnet50_old.hdf5'))




# Predict on the test set
preds_test = model.predict(X_test, verbose=1)
preds_test_t = (preds_test > 0.5).astype(np.uint8)




# Plot some predictions 
for index in range(2):
    helper.plot_img_mask(index, X_test[index], Y_test[index], pred=preds_test_t[index])

