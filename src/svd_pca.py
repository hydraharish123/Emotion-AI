## Load necessary data for SVD and PCA
import numpy as np
from matplotlib import pyplot as plt

X_resized = np.load('X_resized_labelled.npy')
flattened_unlabelled_data = np.load('X_flattened_unlabelled.npy')


## Perform singular value decomposition
U, S, Vt = np.linalg.svd(flattened_unlabelled_data, full_matrices=False)

print(f"The dimensions of S is {S.shape}") ## shape=(4096,)

## Analyse the S matrix (scaling matrix -> diagonal)
plt.figure(figsize=(20,10))
plt.semilogy(S)
plt.title('Singular Values')
plt.show()

plt.figure(figsize=(20,10))
plt.plot(np.cumsum(S)/np.sum(S))
plt.title('Singular Values: Cumulative Sum')
plt.show() 

## Figure out the best r value from the SVD result (which captures the most variance)
## Using elbow plot (Knee locater)
from kneed import KneeLocator

S = np.array(S)  

plt.figure(figsize=(20,10))
plt.semilogy(S)
plt.title("Singular Values")

x = np.arange(1, len(S)+1)
knee = KneeLocator(x, S, curve='convex', direction='decreasing')
r = knee.knee

print(f"Optimal r (elbow point) = {r}")
plt.axvline(x=r, color='r', linestyle='--', label=f'r = {r}')
plt.legend()
plt.show()

## Optimal r (elbow point) = 68


## Perform PCA using the best r value (or you can use a value lesser than it)
from sklearn.decomposition import PCA

val = int(input('Enter number of dimensions to reduce to for PCA (recommended value : r or less than than): '))
pca = PCA(n_components=val)    
pca.fit(flattened_unlabelled_data) 
X_labeled_pca = pca.transform(X_resized)

## Analyse how much variance it captured

explained_variance = pca.explained_variance_ratio_  
plt.figure(figsize=(20, 10))
plt.bar(range(1, len(explained_variance) + 1), explained_variance, alpha=0.7)
plt.xlabel('Principal Component')
plt.ylabel('Explained Variance Ratio')
plt.title('Variance Captured by Each Principal Component')
plt.xticks(range(1, len(explained_variance) + 1))
plt.show()


## Save X_labelled_PCA for next script
np.save('X_labelled_pca.npy', X_labeled_pca)