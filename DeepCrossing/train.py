from DeepCrossing.deepcrossing import deep_crossing
from Datasets.criteo.criteo_loader import criteo_reader
import tensorflow as tf
from sklearn.metrics import accuracy_score

if __name__ == '__main__':
    feature_columns, (X_train, y_train), (X_test, y_test) = criteo_reader()

    k = 32
    hidden_units = [256, 256]
    res_layer_num = 4

    model = deep_crossing(feature_columns, k,  hidden_units, res_layer_num)
    optimizer = tf.keras.optimizers.SGD(0.01)

    train_dataset = tf.data.Dataset.from_tensor_slices((X_train, y_train))
    train_dataset = train_dataset.batch(32).prefetch(tf.data.experimental.AUTOTUNE)

    model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy'])
    model.fit(train_dataset, epochs=5)
    logloss, auc = model.evaluate(X_test, y_test)
    print('logloss {}\nAUC {}'.format(round(logloss,2), round(auc,2)))

    # summary = tf.summary.create_file_writer("/Users/leehom/Downloads/tensorboard_test")
    # for i in range(5):
    #     with tf.GradientTape() as tape:
    #         pre = model(X_train)
    #         loss = tf.reduce_mean(tf.keras.losses.binary_crossentropy(y_train.to_numpy().reshape(-1, 1), pre))
    #         print(loss.numpy())
    #     with summary.as_default():
    #         tf.summary.scalar('loss', loss, i)
    #     grad = tape.gradient(loss, model.variables)
    #     optimizer.apply_gradients(grads_and_vars=zip(grad, model.variables))

    #评估
    pre = model(X_test)
    pre = [1 if x>0.5 else 0 for x in pre]
    print("Accuracy: ", accuracy_score(y_test.to_numpy().reshape(-1, 1), pre))