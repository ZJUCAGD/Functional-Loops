import numpy as np
from gtda.homology import VietorisRipsPersistence


def computing_PD_based_on_cor(cor,  homology_dimensions=[0, 1]):
    # Construct functional network from correlation coefficient matrix cor and calculate its PD

    # Calculate the functional connectivity matrix
    cor = abs(cor)
    cor = np.nan_to_num(cor)

    # For lower-star filtration
    cor = 1 - cor
    # To ensure that this matrix is a distance matrix
    for ii in range(cor.shape[0]):
        for jj in range(ii):
            cor[ii, jj] = cor[jj, ii]
    # The diagonal is set to 0
    row, col = np.diag_indices_from(cor)
    cor[row, col] = 0

    # lower-star filtration
    VR = VietorisRipsPersistence(
        metric="precomputed", homology_dimensions=homology_dimensions)
    dgms = VR.fit_transform([cor])

    return dgms[0]

# Compute persistence features, such as mean or p-norm, for PDs with a single dimension


def compute_persistence(net_dgms, order=2, homology_dimension=0.0):

    net_persistence = np.zeros(len(net_dgms))
    for i, dgm in enumerate(net_dgms):
        index = np.where(dgm[:, 2] == homology_dimension)

        # calculate persistence for PD
        persistence = dgm[index, 1]-dgm[index, 0]
        if order == 'Mean':
            net_persistence[i] = np.mean(persistence)
        elif order == 'Number':
            net_persistence[i] = len(index[0])
        else:
            net_persistence[i] = np.linalg.norm(persistence, ord=order)

    return net_persistence

# Calculate features of persistence


def get_persistence_features(dgms):
    features = {'1Number': None, '1L2': None, '1Mean': None}

    for method in features.keys():
        if method[1:] == 'L2':
            features[method] = compute_persistence(
                dgms, order=2, homology_dimension=eval(method[0]))
        elif method[1:] == 'Mean':
            features[method] = compute_persistence(
                dgms, order='Mean', homology_dimension=eval(method[0]))
        elif method[1:] == 'Number':
            features[method] = compute_persistence(
                dgms, order='Number', homology_dimension=eval(method[0]))

    return features
