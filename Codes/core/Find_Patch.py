import numpy as np
from numba import njit

def Find_Patch(Edge, Num_TBF, J_col):
    """
    Find patches from IRES solution.
    Edge: (n_edges, 4) array, columns 3 and 4 are used for connectivity
    Num_TBF: number of patches to find
    J_col: amplitude vector (n_sources,)
    Returns: IND (n_sources, Num_TBF), Num_TBF (possibly reduced)
    """
    Edge_mod = Edge[:, [2, 3]]  # MATLAB 1-based, Python 0-based
    Number_src = int(np.max(Edge_mod))
    IND = np.zeros((Number_src, Num_TBF), dtype=int)
    J_amp = J_col.copy()

    for i_clst in range(Num_TBF):
        ind_max = np.argmax(J_amp)
        IND[ind_max, i_clst] = 1
        ind_neigh_local = [ind_max]
        ind_neigh_global = np.zeros(Number_src, dtype=int)
        pntInd_neigh_global = 0

        while True:
            nNeighLocal = len(ind_neigh_local)
            ind_neigh = []
            for i_ind in range(nNeighLocal):
                # Find immediate neighbours of this node
                node = ind_neigh_local[i_ind]
                neigh1 = Edge_mod[Edge_mod[:, 0] == node, 1]
                neigh2 = Edge_mod[Edge_mod[:, 1] == node, 0]
                ind_neigh.extend(neigh1.tolist())
                ind_neigh.extend(neigh2.tolist())
            if len(ind_neigh) == 0:
                break
            ind_neigh = np.unique(ind_neigh)
            # Discard locations where solution amplitude is zero
            ind_neigh = [idx for idx in ind_neigh if J_amp[idx] != 0]
            # Remove repeated indices, comparing with the global index
            keep = []
            for idx in ind_neigh:
                if idx not in ind_neigh_global[:pntInd_neigh_global]:
                    keep.append(idx)
            ind_neigh = keep
            nNeigh = len(ind_neigh)
            # Store local neighbour results to global index/registry
            if nNeigh > 0:
                ind_neigh_global[pntInd_neigh_global:pntInd_neigh_global+nNeigh] = ind_neigh
                pntInd_neigh_global += nNeigh
            # Set boundary nodes to current local neighbours
            ind_neigh_local = ind_neigh
            # If no new neighbors were found, break
            if len(ind_neigh) == 0:
                break
            else:
                IND[ind_neigh, i_clst] = 1
        J_amp[IND[:, i_clst] > 0] = 0
        # Return if no more large patches of activity were detected
        if not np.any(J_amp) and i_clst < Num_TBF - 1:
            Num_TBF = i_clst + 1
            IND = IND[:, :Num_TBF]
            break
    return IND, Num_TBF
