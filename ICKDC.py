#following the sklearn convention
from math import gamma

import numpy as np
from sklearn.base import BaseEstimator, ClusterMixin
from sklearn.utils.validation import check_array, check_is_fitted
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics.pairwise import pairwise_distances

class ICKDC(BaseEstimator, ClusterMixin):

    def __init__(
            self,
            gamma=2.0
    ):
        self.gamma = gamma

    def _local_density_estimation(self, X, K):
        n = X.shape[0] #number of data points
        NN = NearestNeighbors(n_neighbors=K+1) #init NN object with our own K (+1 because the point itself counts)
        NN.fit(X) #train the NN
        rho = np.zeros(shape=(n, 1)) #init the local denisty array
        distances, indices = NN.kneighbors(X) #get the distances and indices of the nearest neighbors
        # for each datapoint
        rho = np.sum(np.exp(-np.square(distances)), axis=1) #apply the local density formula
        #np.array with n local density values

        return rho, distances, indices

    def _core_points_identification(self, X, rho, indices):
        n = X.shape[0] #number of datapoints
        #rho[indices] gets the densities of the K+1 neighbors for each point
        local_max_idx = np.argmax(rho[indices], axis=1) #np.argmax finds the column index (0 to K) of the neighbor with the max density
        CP = indices[np.arange(n), local_max_idx] #map the local column index back to the global index of the actual data point
        CP_minus, inv_indices = np.unique(CP, return_inverse=True)#since multiple core points might be repeated we apply unique

        return CP, CP_minus, inv_indices

    def _core_points_integration(self, K, X, CP_minus):
        n = X.shape[0]  # number of datapoints
        n_core = CP_minus.shape[0] #number of core points
        alpha = 2*(n_core/n + 0.1) #calculate alpha constant used in computing a new K value
        K_alpha = max(1, min(int(alpha * K), n - 1)) #modified K alpha for NN for core points

        NN_alpha = NearestNeighbors(n_neighbors=K_alpha+1, metric="euclidean") #init nearest neighbors with new k
        NN_alpha.fit(X)
        distances, indices = NN_alpha.kneighbors(X[CP_minus])

        #now we will classify core points into clusters

        core_labels = np.full(n_core, -1)  #create array of dimensions equal to core points where -1 means unclassified
        neighborhoods = [set(row) for row in indices] #convert each row of indices into a Python set for O(1) intersection lookups

        current_cluster_id = 0

        while -1 in core_labels: #while there are unclassified core points do:

            unclassified_indices = np.where(core_labels == -1)[0] #select a unclassified core points
            seed_idx = unclassified_indices[0]  #pick the first one
            core_labels[seed_idx] = current_cluster_id #assign it to the new cluster
            queue = [seed_idx] #start the queue with this seed point (core points inside this point's neighborhood will be classified to the same cluster).

            while queue:
                current_idx = queue.pop(0) #pop the first item off the queue to check its neighbors
                unclassified = np.where(core_labels == -1)[0] #evaluate who is still unclassified at this exact moment

                for candidate_idx in unclassified:
                    if not neighborhoods[current_idx].isdisjoint(neighborhoods[candidate_idx]): #test if the two core-points share at least one point in their neighborhoods
                        core_labels[candidate_idx] = current_cluster_id #if they do (intersection) --> assign the candidate to the current cluster
                        queue.append(candidate_idx)  #add this new member to the queue to check its neighbors next

            #once the queue is empty, this cluster has stopped growing (no core points in the neighborhoods)
            current_cluster_id += 1 #increment the cluster ID for the next isolated core point


        return core_labels #maps exactly 1-to-1 with the CP_minus array

    def _rest_integration(self, core_labels, inv_indices):
        return core_labels[inv_indices]

    def fit(self, X, y=None):
        X = check_array(X) #validate the input array (ensures it's 2D, no NaNs, etc.)
        n_data = X.shape[0] #extract n (number of data points) to calculate K if using gamma
        K = max(1, min(int((self.gamma / 100.0) * n_data), n_data - 1))

        #1: local density estimation
        rho, distances, indices = self._local_density_estimation(X=X, K=K)

        #2: core points identification
        CP, CP_minus, inv_indices = self._core_points_identification(X, rho, indices)

        #3: core points integration
        core_labels = self._core_points_integration(K, X, CP_minus)

        self.labels_ = self._rest_integration(core_labels, inv_indices) #final labels (clusters) following the sklearn convention
        self.core_points_ = CP_minus

        return self #fit method must always return self