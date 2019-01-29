from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import numpy as np

#MNIST Data
import mnist_data
mnist = mnist_data.read_data_sets("F:\ASU\pro_auto_encoder", one_hot=True)
X_train, _  = mnist.train.next_batch(256)

pca = PCA(n_components=32)
pca_result = pca.fit_transform(X_train)

print(X_train[0].shape)
print("pca",pca_result[0].shape)

n = 4
ori = np.empty((28*n, 28*n))
gen = np.empty((28*n, 28*n))

for i in range(n):
    for j in range(n):
        ori[i * 28:(i + 1) * 28, j * 28:(j + 1) * 28] = \
            X_train[j].reshape([28, 28])
    for j in range(n):
        print(j)
        gen[i * 28:(i + 1) * 28, j * 28:(j + 1) * 28] = \
            pca_result[j].reshape([28, 28])
            
fig = plt.figure(figsize=(n, n))
ax1 = fig.add_subplot(2,2,1)
ax1.imshow(ori, origin="upper", cmap="gray")
ax2 = fig.add_subplot(2,2,2)
ax2.imshow(gen, origin="upper", cmap="gray")
plt.show()


