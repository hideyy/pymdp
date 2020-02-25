#!/usr/bin/env python
# -*- coding: utf-8 -*-
# pylint: disable=no-member

""" Functions
__author__: Conor Heins, Alexander Tschantz, Brennan Klein
"""

import itertools
import numpy as np
import torch
from scipy import special
from inferactively.distributions import Categorical
from inferactively.distributions import Dirichlet


def softmax(distrib, return_numpy = True):
    """ Computes the softmax function on a set of values
    """
    if isinstance(distrib, Categorical):
        if distrib.IS_AOA:
            output = Categorical(dims=[list(el.shape) for el in distrib])
            for i in range(len(distrib.values)):
                output[i] = softmax(distrib.values[i], return_numpy=True)
            output = Categorical(dims = [list(el.shape) for el in distrib])
        else:
            distrib = np.copy(distrib.values)
    output = distrib - distrib.max(axis=0)
    output = np.exp(output)
    output = output / np.sum(output, axis=0)
    if return_numpy:
        return output
    else:
        return Categorical(values=output)


# def generate_policies(n_actions, policy_len):
#     """ Generate of possible combinations of N actions for policy length T
#     Returns
#     -------
#     `policies` [list]
#         A list of tuples, each specifying a list of actions [int]
#     """

#     x = [n_actions] * policy_len
#     return list(itertools.product(*[list(range(i)) for i in x]))


def constructNu(Ns,Nf,cntrl_fac_idx,policy_len):
    '''Generate list of possible combinations of Ns[f_i] actions for Nf hidden state factors,
    where Nu[i] gives the number of actions available along hidden state factor f_i. Assumes that for each controllable hidden
    state factor, the number of possible actions == Ns[f_i]
    Arguments:
    -------
    Ns: list of dimensionalities of hidden state factors
    Nf: number of hidden state factors total
    cntrl_fac_idx: indices of the hidden state factors that are controllable (i.e. those whose Nu[i] > 1)
    policy_len: length of each policy
    Returns:
    -------
    Nu: list of dimensionalities of actions along each hidden state factor
    possible_policies: list of arrays, where each array within the list corresponds to a policy, and each row
                        within a given policy (array) corresponds to a list of actions for each the several hidden state factor
                        for a given timestep (policy_len x Nf)
    '''

    Nu = []

    for f_i in range(Nf):
        if f_i in cntrl_fac_idx:
            Nu.append(Ns[f_i]) 
        else:
            Nu.append(1)

    x = Nu * policy_len
    
    possible_policies = list(itertools.product(*[list(range(i)) for i in x]))

    if policy_len > 1:
        for pol_i in range(len(possible_policies)):
            possible_policies[pol_i] = np.array(possible_policies[pol_i]).reshape(policy_len,Nf)

    Nu = np.array(Nu).astype(int)
    return Nu, possible_policies


def kl_divergence(q, p):
    """
    TODO: make this work for multi-dimensional arrays
    """
    if not isinstance(type(q), type(Categorical)) or not isinstance(
        type(p), type(Categorical)
    ):
        raise ValueError("`kl_divergence` function takes `Categorical` objects")
    q.remove_zeros()
    p.remove_zeros()
    q = np.copy(q.values)
    p = np.copy(p.values)
    kl = np.sum(q * np.log(q / p), axis=0)[0]
    return kl


def spm_dot(X, x, dims_to_omit=None):
    """ Dot product of a multidimensional array with `x`
    The dimensions in `dims_to_omit` will not be summed across during the dot product
    Parameters
    ----------
    `x` [1D numpy.ndarray] - either vector or array of arrays
        The alternative array to perform the dot product with
    `dims_to_omit` [list :: int] (optional)
        Which dimensions to omit
    """

    if x.dtype == object:
        dims = (np.arange(0, len(x)) + X.ndim - len(x)).astype(int)
    else:
        if x.shape[0] != X.shape[1]:
            """
            Case when the first dimension of `x` is likely the same as the first dimension of `A`
            e.g. inverting the generative model using observations.
            Equivalent to something like self.values[np.where(x),:]
            when `x` is a discrete 'one-hot' observation vector
            """
            dims = np.array([0], dtype=int)
        else:
            """
            Case when `x` leading dimension matches the lagging dimension of `values`
            E.g. a more 'classical' dot product of a likelihood with hidden states
            """
            dims = np.array([1], dtype=int)
        x_new = np.empty(1, dtype=object)
        x_new[0] = x.squeeze()
        x = x_new

    if dims_to_omit is not None:
        if not isinstance(dims_to_omit, list):
            raise ValueError("dims_to_omit must be a `list`")
        dims = np.delete(dims, dims_to_omit)
        if len(x) == 1:
            x = np.empty([0], dtype=object)
        else:
            x = np.delete(x, dims_to_omit)

    Y = X
    for d in range(len(x)):
        s = np.ones(np.ndim(Y), dtype=int)
        s[dims[d]] = np.shape(x[d])[0]
        Y = Y * x[d].reshape(tuple(s))
        Y = np.sum(Y, axis=dims[d], keepdims=True)
    Y = np.squeeze(Y)

    # perform check to see if `y` is a number
    if np.prod(Y.shape) <= 1.0:
        Y = Y.item()
        Y = np.array([Y]).astype("float64")

    return Y

def spm_dot_torch(X, x, dims_to_omit=None):
    """ Dot product of a multidimensional array with `x` -- Pytorch version, using Tensor instances
    @TODO: Instead of a separate function, this should be integrated with spm_dot so that it can either take torch.Tensors or nd.arrays

    The dimensions in `dims_to_omit` will not be summed across during the dot product

    Parameters
    ----------
    'X' [torch.Tensor]
    `x` [1D torch.Tensor or nnumpy object array containing 1D torch.Tensors]
        The array(s) to dot X with
    `dims_to_omit` [list :: int] (optional)
        Which dimensions to omit from summing across
    """

    if x.dtype == object:
        dims = (np.arange(0, len(x)) + X.ndim - len(x)).astype(int)
    else:
        if x.shape[0] != X.shape[1]:
            """
            Case when the first dimension of `x` is likely the same as the first dimension of `A`
            e.g. inverting the generative model using observations.
            Equivalent to something like self.values[np.where(x),:]
            when `x` is a discrete 'one-hot' observation vector
            """
            dims = np.array([0], dtype=int)
        else:
            """
            Case when `x` leading dimension matches the lagging dimension of `values`
            E.g. a more 'classical' dot product of a likelihood with hidden states
            """
            dims = np.array([1], dtype=int)
        x_new = np.empty(1, dtype=object)
        x_new[0] = x.squeeze()
        x = x_new

    if dims_to_omit is not None:
        if not isinstance(dims_to_omit, list):
            raise ValueError("dims_to_omit must be a `list`")
        dims = np.delete(dims, dims_to_omit)
        if len(x) == 1:
            x = np.empty([0], dtype=object)
        else:
            x = np.delete(x, dims_to_omit)

    Y = X
    for d in range(len(x)):
        s = np.ones(Y.ndim, dtype=int)
        s[dims[d]] = max(x[d].shape)
        Y = Y * x[d].view(tuple(s))
        Y = Y.sum(dim=int(dims[d]), keepdim=True)
    Y = Y.squeeze()

    # perform check to see if `y` is a number
    if Y.numel() <= 1:
        Y = np.asscalar(Y)
        Y = torch.Tensor([Y])

    return Y


def spm_cross(X, x=None, *args):
    """ Multi-dimensional outer product
    If no `x` argument is passed, the function returns the "auto-outer product" of self
    Otherwise, the function will recursively take the outer product of the initial entry
    of `x` with `self` until it has depleted the possible entries of `x` that it can outer-product
    Parameters
    ----------
    `x` [np.ndarray] || [Categorical] (optional)
        The values to perfrom the outer-product with
    `args` [np.ndarray] || [Categorical] (optional)
        Perform the outer product of the `args` with self
    
    Returns
    -------
    `y` [np.ndarray] || [Categorical]
        The result of the outer-product
    """

    if len(args) == 0 and x is None:
        if X.dtype == "object":
            Y = spm_cross(*list(X))

        elif np.issubdtype(X.dtype, np.number):
            Y = X

        return Y

    if X.dtype == "object":
        X = spm_cross(*list(X))

    if x is not None and x.dtype == "object":
        x = spm_cross(*list(x))

    reshape_dims = tuple(list(X.shape) + list(np.ones(x.ndim, dtype=int)))
    A = X.reshape(reshape_dims)

    reshape_dims = tuple(list(np.ones(X.ndim, dtype=int)) + list(x.shape))
    B = x.reshape(reshape_dims)

    Y = np.squeeze(A * B)

    for x in args:
        Y = spm_cross(Y, x)

    return Y

def spm_wnorm(A):
    """
    Normalization of a prior over Dirichlet parameters, used in updates for information gain
    """
    
    A = A + 1e-16
    
    norm = np.divide(1.0, np.sum(A,axis=0))
    
    avg = np.divide(1.0, A)
    
    wA = norm - avg
   
    return wA


def spm_betaln(z):
    """
    Returns the log of the multivariate beta function of a vector.
    FORMAT y = spm_betaln(z)
     y = spm_betaln(z) computes the natural logarithm of the beta function
     for corresponding elements of the vector z. if concerned is a matrix,
     the logarithm is taken over the columns of the matrix z.
    """

    y = np.sum(special.gammaln(z), axis=0) - special.gammaln(np.sum(z, axis=0))

    return y


def update_posterior_states(A, observation, prior, return_numpy=True, method="FPI", **kwargs):
    """ 
    Update marginal posterior qx using variational inference, with optional selection of a message-passing algorithm
    Parameters
    ----------
    'A' [numpy nd.array (matrix or tensor or array-of-arrays) or Categorical]:
        Observation likelihood of the generative model, mapping from hidden states to observations. 
        Used to invert generative model to obtain marginal likelihood over hidden states, given the observation
    'observation' [numpy 1D array, array of arrays (with 1D numpy array entries), int or tuple]:
        The observation (generated by the environment). If single modality, this can be a 1D array 
        (one-hot vector representation) or an int (observation index)
        If multi-modality, this can be an array of arrays (whose entries are 1D one-hot vectors) or a tuple (of observation indices)
        The observation (generated by the environment). If single modality, this can be a 1D array (one-hot vector representation) or an int (observation index)
        If multi-mopdality, this can be an aray of arrays (whose entries are 1D one-hot vectors) or a tuple (of observation indices)
    'prior' [numpy 1D array, array of arrays (with 1D numpy array entries), or Categorical]:
        Prior beliefs of the agent, to be integrated with the marginal likelihood to obtain posterior
    'return_numpy' [Boolean]:
        True/False flag to determine whether the posterior is returned as a numpy array or a Categorical
    'method' [str]:
        Algorithm used to perform the variational inference. 
        Options: 'FPI' - Fixed point iteration 
                - http://www.cs.cmu.edu/~guestrin/Class/10708/recitations/r9/VI-view.pdf, slides 13- 18
                - http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.137.221&rep=rep1&type=pdf, slides 24 - 38
                 'VMP  - Variational message passing
                 'MMP' - Marginal message passing
                 'BP'  - Belief propagation
                 'EP'  - Expectation propagation
                 'CV'  - CLuster variation method
    **kwargs: List of keyword/parameter arguments corresponding to parameter values for the respective variational inference algorithm

    Returns
    ----------
    'qx' [numpy 1D array, array of arrays (with 1D numpy array entries), or Categorical]:
        Marginal posterior beliefs over hidden states (single- or multi-factor) achieved via variational approximation.
    """

    if isinstance(A, Categorical):
        A = A.values

    if A.dtype == "object":
        Nf = A[0].ndim - 1
        Ns = list(A[0].shape[1:])
        Ng = len(A)
        No = []
        for g in range(Ng):
            No.append(A[g].shape[0])
    else:
        Nf = A.ndim - 1
        Ns = list(A.shape[1:])
        Ng = 1
        No = [A.shape[0]]

    if isinstance(observation, Categorical):
        observation = observation.values
        if Ng == 1:
            observation = observation.squeeze()
        else:
            for g in range(Ng):
                observation[g] = observation[g].squeeze()

    if isinstance(observation, (int, np.integer)):
        observation = np.eye(No[0])[observation]

    if isinstance(observation, tuple):
        observation_AoA = np.empty(Ng, dtype=object)
        for g in range(Ng):
            observation_AoA[g] = np.eye(No[g])[observation[g]]
        
        observation = observation_AoA

    if isinstance(prior, Categorical):

        prior_new = np.empty(Nf, dtype=object)
        
        if prior.IS_AOA:
            for f in range(Nf):
                prior_new[f] = prior[f].values.squeeze()
        else:
            prior_new[0] = prior.values.squeeze()

        prior = prior_new

    elif prior.dtype != "object":

        prior_new = np.empty(Nf, dtype=object)
        prior_new[0] = prior
        prior = prior_new

    if method == "FPI":
        qx = run_FPI(A, observation, prior, No, Ns, **kwargs)
    if method == "VMP":
        raise NotImplementedError("VMP is not implemented")
    if method == "MMP":
        raise NotImplementedError("MMP is not implemented")
    if method == "BP":
        raise NotImplementedError("BP is not implemented")
    if method == "EP":
        raise NotImplementedError("EP is not implemented")
    if method == "CV":
        raise NotImplementedError("CV is not implemented")

    if return_numpy:
        return qx
    else:
        return Categorical(values=qx)


def run_FPI(A, observation, prior, No, Ns, num_iter=10, dF=1.0, dF_tol=0.001):
    """
    Update marginal posterior beliefs about hidden states
    using variational fixed point iteration (FPI)
    Parameters
    ----------
    'A' [numpy nd.array (matrix or tensor or array-of-arrays)]:
        Observation likelihood of the generative model, mapping from hidden states to observations. 
        Used to invert generative model to obtain marginal likelihood over hidden states, given the observation
    'observation' [numpy 1D array or array of arrays (with 1D numpy array entries)]:
        The observation (generated by the environment). If single modality, this can be a 1D array (one-hot vector representation).
        If multi-modality, this can be an array of arrays (whose entries are 1D one-hot vectors).
    'prior' [numpy 1D array, array of arrays (with 1D numpy array entries)]:
        Prior beliefs of the agent, to be integrated with the marginal likelihood to obtain posterior
    'num_iter' [int]:
        Number of variational fixed-point iterations to run.
    'dF' [float]:
        Starting free energy gradient (dF/dQx) before updating in the course of gradient descent.
    'dF_tol' [float]:
        Threshold value of the gradient of the variational free energy (dF/dQx), to be checked at each iteration. If 
        dF <= dF_tol, the iterations are halted pre-emptively and the final marginal posterior belief(s) is(are) returned
    Returns
    ----------
    'qx' [numpy 1D array or array of arrays (with 1D numpy array entries):
        Marginal posterior beliefs over hidden states (single- or multi-factor) achieved via variational fixed point iteration (mean-field)
    """

    # Code should be changed to this, once you've defined the gradient of the free energy:
    # dF = 1
    # while iterNum < numIter or dF > dF_tol:
    #       [DO ITERATIONS]
    # until then, use the following code:

    Ng = len(No)
    Nf = len(Ns)

    L = np.ones(tuple(Ns))

    # loop over observation modalities and use mean-field assumption to multiply 'induced posterior' onto
    # a single joint likelihood over hidden factors - of size Ns
    if Ng == 1:
        L *= spm_dot(A, observation)
    else:
        for g in range(Ng):
            L *= spm_dot(A[g], observation[g])

    # initialize posterior to flat distribution
    qx = np.empty(Nf, dtype=object)
    for f in range(Nf):
        qx[f] = np.ones(Ns[f]) / Ns[f]

    # in the trivial case of one hidden state factor, inference doesn't require FPI
    if Nf == 1:
        qL = spm_dot(L, qx, [0])
        qx[0] = softmax(np.log(qL + 1e-16) + np.log(prior[0] + 1e-16))
        return qx[0]

    else:
        iter_i = 0
        while iter_i < num_iter:

            for f in range(Nf):

                # get the marginal for hidden state factor f by marginalizing out
                # other factors (summing them, weighted by their posterior expectation)
                qL = spm_dot(L, qx, [f])

                # this math is wrong, but anyway in theory we should add this in at
                # some point -- calculate the free energy and update the derivative
                # accordingly:
                # lnP = spm_dot(A_gm[g1,g2],O)
                # dF += np.sum(np.log(qL + 1e-16)-np.log(prior + 1e-16) + spm_dot(lnL, Qs, [f]))

                qx[f] = softmax(np.log(qL + 1e-16) + np.log(prior[f] + 1e-16))

            iter_i += 1

        return qx

def update_posterior_policies(Qs, A, pA, B, pB, C, possiblePolicies, gamma = 16.0, return_numpy=True):
    '''
    Updates the posterior beliefs about policies using the expected free energy approach (where belief in a policy is proportional to the free energy expected under its pursuit)
    @TODO: Needs to be amended for use with multi-step policies (where possiblePolicies is a list of np.arrays, not a list of tuples)
    Parameters
    ----------
    Qs [1D numpy array, array-of-arrays, or Categorical (either single- or multi-factor)]:
        current marginal beliefs about hidden state factors
    A [numpy ndarray, array-of-arrays (in case of multiple modalities), or Categorical (both single and multi-modality)]:
        Observation likelihood model (beliefs about the likelihood mapping entertained by the agent)
    pA [numpy ndarray, array-of-arrays (in case of multiple modalities), or Dirichlet (both single and multi-modality)]:
        Prior dirichlet parameters for A
    B [numpy ndarray, array-of-arrays (in case of multiple hidden state factors), or Categorical (both single and multi-factor)]:
        Transition likelihood model (beliefs about the likelihood mapping entertained by the agent)
    pB [numpy ndarray, array-of-arrays (in case of multiple hidden state factors), or Dirichlet (both single and multi-factor)]:
        Prior dirichlet parameters for B
    C [numpy 1D-array, array-of-arrays (in case of multiple modalities), or Categorical (both single and multi-modality)]:
        Prior beliefs about outcomes (prior preferences)
    possiblePolicies [list of tuples]:
        a list of all the possible policies, each expressed as a tuple of indices, where a given index corresponds to an action on a particular hidden state factor
        e.g. possiblePolicies[1][2] yields the index of the action under Policy 1 that affects Hidden State Factor 2
    gamma [float]:
        precision over policies, used as the inverse temperature parameter of a softmax transformation of the expected free energies of each policy
    return_numpy [Boolean]:
        True/False flag to determine whether output of function is a numpy array or a Categorical
    
    Returns
    --------
    p_i [1D numpy array or Categorical]:
        posterior beliefs about policies, defined here as a softmax function of the expected free energies of policies
    EFE [1D numpy array or Categorical]:
        the expected free energies of policies
    '''

    # if not isinstance(C,Categorical):
    #     C = Categorical(values = C)
    
    # C = softmax(C.log())

    Np = len(possiblePolicies)

    EFE = np.zeros(Np)
    p_i = np.zeros((Np,1))

    for p_i, policy in enumerate(possiblePolicies):

        Qs_pi = get_expected_states(Qs, B, policy)

        Qo_pi = get_expected_obs(Qs_pi, A)

        utility = calculate_expected_utility(Qo_pi,C)
        EFE[p_i] += utility

        surprise_states = calculate_expected_surprise(A, Qs_pi)
        EFE[p_i] += surprise_states

        infogain_pA = calculate_infogain_pA(pA, Qo_pi, Qs_pi)
        EFE[p_i] += infogain_pA

        infogain_pB = calculate_infogain_pB(pB, Qs_pi, Qs, policy)
        EFE[p_i] += infogain_pB
    
    p_i = softmax(EFE * gamma)
    
    if return_numpy:
        p_i = p_i/p_i.sum(axis=0)
    else:
        p_i = Categorical(values = p_i)
        p_i.normalize()

    return p_i, EFE

def get_expected_states(Qs, B, policy, return_numpy = False):
    '''
    Given a posterior density Qs, a transition likelihood model B, and a policy, 
    get the state distribution expected under that policy's pursuit.

    @TODO: Needs to be amended for use with multi-step policies (where possiblePolicies is a list of np.arrays, not a list of tuples)
    Parameters
    ----------
    Qs [numpy 1D array, array-of-arrays (where each entry is a numpy 1D array), or Categorical (either single-factor or AoA)]:
        Current posterior beliefs about hidden states
    B [numpy nd-array, array-of-arrays (where each entry is a numpy nd-array), or Categorical (either single-factor of AoA)]:
        Transition likelihood mapping from states at t to states at t + 1, with different actions (per factor) stored along the lagging dimension
    policy [tuple of ints]:
        Tuple storing indices of actions along each hidden state factor. E.g. policy[1] gives the index of the action occurring on Hidden State Factor 1
    return_numpy [Boolean]:
        True/False flag to determine whether output of function is a numpy array or a Categorical
    Returns
    -------
    Qs_pi [numpy 1D array, array-of-arrays (where each entry is a numpy 1D array), or Categorical (either single-factor or AoA)]:
        Expected states under the given policy - also known as the 'posterior predictive density'
    '''

    if isinstance(B, Categorical):

        if B.IS_AOA:
            Qs_pi = Categorical( values = np.array([B[f][:,:,a].dot(Qs[f], return_numpy=True)[:,np.newaxis] for f, a in enumerate(policy)], dtype = object) )
        else:
            Qs_pi = B[:,:,policy[0]].dot(Qs)
        
        if return_numpy and Qs_pi.IS_AOA:
            Qs_pi_flattened = np.empty(len(Qs_pi.values), dtype = object)
            for f in range(len(Qs_pi.values)):
                Qs_pi_flattened[f] = Qs_pi[f].values.flatten()
            return Qs_pi_flattened
        elif return_numpy and not Qs_pi.IS_AOA:
            return Qs_pi.values.flatten()
        else:
            return Qs_pi
    
    elif B.dtype == 'object':

        Nf = len(B)

        Qs_pi = np.empty(Nf, dtype = object)

        if isinstance(Qs, Categorical):
            Qs = Qs.values
            for f in range(Nf):
                Qs[f] = Qs[f].flatten()
        for f in range(Nf):
            Qs_pi[f] = spm_dot(B[f][:,:,policy[f]], Qs[f])

    else:

        if isinstance(Qs, Categorical):
            Qs = Qs.values.flatten()

        Qs_pi = spm_dot(B[:,:,policy[0]], Qs)
    
    if not return_numpy:
        Qs_pi = Categorical(values = Qs_pi)
        return Qs_pi
    else:
        return Qs_pi


def get_expected_obs(Qs_pi, A, return_numpy = False):
    '''
    Given a posterior predictive density Qs_pi and an observation likelihood model A,
    get the expected observations given the predictive posterior.

    @TODO: Needs to be amended for use with multi-step policies (where possiblePolicies is a list of np.arrays, not a list of tuples)
    Parameters
    ----------
    Qs_pi [numpy 1D array, array-of-arrays (where each entry is a numpy 1D array), or Categorical (either single-factor or AoA)]:
        Posterior predictive density over hidden states
    A [numpy nd-array, array-of-arrays (where each entry is a numpy nd-array), or Categorical (either single-factor of AoA)]:
        Observation likelihood mapping from hidden states to observations, with different modalities (if there are multiple) stored in different arrays
    return_numpy [Boolean]:
        True/False flag to determine whether output of function is a numpy array or a Categorical
    Returns
    -------
    Qo_pi [numpy 1D array, array-of-arrays (where each entry is a numpy 1D array), or Categorical (either single-factor or AoA)]:
        Expected observations under the given policy 
    '''
    if isinstance(A, Categorical):
        
        if not return_numpy:   
            Qo_pi = A.dot(Qs_pi)
            return Qo_pi
        else:
            Qo_pi = A.dot(Qs_pi,return_numpy=True)
            if Qo_pi.dtype == 'object':
                Qo_pi_flattened = np.empty(len(Qo_pi),dtype=object)
                for g in range(len(Qo_pi)):
                    Qo_pi_flattened[g] = Qo_pi[g].flatten()
                return Qo_pi_flattened
            else:
                return Qo_pi.flatten()
    
    elif A.dtype == 'object':

        Ng = len(A)

        Qo_pi = np.empty(Ng, dtype = object)

        if isinstance(Qs_pi, Categorical):
            Qs_pi = Qs_pi.values
            for f in range(len(Qs_pi)):
                Qs_pi[f] = Qs_pi[f].flatten()
        for g in range(Ng):
            Qo_pi[g] = spm_dot(A[g],Qs_pi)

    else:

        if isinstance(Qs_pi, Categorical):
            Qs_pi = Qs_pi.values

        Qo_pi = spm_dot(A, Qs_pi)
    
    if not return_numpy:
        Qo_pi = Categorical(values = Qo_pi)
        return Qo_pi
    else:
        return Qo_pi

def calculate_expected_utility(Qo_pi,C):
    '''
    Given expected observations under a policy Qo_pi and a prior over observations C
    compute the expected utility of the policy.

    @TODO: Needs to be amended for use with multi-step policies (where possiblePolicies is a list of np.arrays, not a list of tuples)
    Parameters
    ----------
    Qo_pi [numpy 1D array, array-of-arrays (where each entry is a numpy 1D array), or Categorical (either single-factor or AoA)]:
        Posterior predictive density over outcomes
    C [numpy nd-array, array-of-arrays (where each entry is a numpy nd-array):
        Prior beliefs over outcomes, expressed in terms of relative log probabilities
    Returns
    -------
    expected_util [scalar]:
        Utility (reward) expected under the policy in question
    '''

    if isinstance(Qo_pi,Categorical):
        Qo_pi = Qo_pi.values
    
    if Qo_pi.dtype == 'object':
        for g in range(len(Qo_pi)):
            Qo_pi[g] = Qo_pi[g].flatten()

    if C.dtype == 'object':
        
        expected_util = 0
        
        Ng = len(C)
        for g in range(Ng):
            lnC = np.log(softmax(C[g][:,np.newaxis])+1e-16)
            expected_util += Qo_pi[g].flatten().dot(lnC)

    else:

        lnC = np.log(softmax(C[:,np.newaxis]) + 1e-16)
        expected_util = Qo_pi.flatten().dot(lnC)
    
    return expected_util

def calculate_expected_surprise(A, Qs_pi):
    '''
    Given a likelihood mapping A and a posterior predictive density over states Qs_pi,
    compute the Bayesian surprise (about states) expected under that policy

    @TODO: Needs to be amended for use with multi-step policies (where possiblePolicies is a list of np.arrays, not a list of tuples)
    Parameters
    ----------
    A [numpy nd-array, array-of-arrays (where each entry is a numpy nd-array), or Categorical (either single-factor of AoA)]:
        Observation likelihood mapping from hidden states to observations, with different modalities (if there are multiple) stored in different arrays
    Qs_pi [numpy 1D array, array-of-arrays (where each entry is a numpy 1D array), or Categorical (either single-factor or AoA)]:
        Posterior predictive density over hidden states
    Returns
    -------
    states_surprise [scalar]:
        Surprise (about states) expected under the policy in question
    '''

    if isinstance(A, Categorical):
        A = A.values

    if isinstance(Qs_pi, Categorical):
        Qs_pi = Qs_pi.values
    
    if Qs_pi.dtype == 'object':
        for f in range(len(Qs_pi)):
            Qs_pi[f] = Qs_pi[f].flatten()
    else:
        Qs_pi = Qs_pi.flatten()
    
    states_surprise = spm_MDP_G(A,Qs_pi)

    return states_surprise

def calculate_infogain_pA(pA, Qo_pi, Qs_pi):
    '''
    Compute expected Dirichlet information gain about parameters pA under a policy
    Parameters
    ----------
    pA [numpy nd-array, array-of-arrays (where each entry is a numpy nd-array), or Dirichlet (either single-factor of AoA)]:
        Prior dirichlet parameters parameterizing beliefs about the likelihood mapping from hidden states to observations, 
        with different modalities (if there are multiple) stored in different arrays.
    Qo_pi [numpy 1D array, array-of-arrays (where each entry is a numpy 1D array), or Categorical (either single-factor or AoA)]:
        Posterior predictive density over observations, given hidden states expected under a policy
    Qs_pi [numpy 1D array, array-of-arrays (where each entry is a numpy 1D array), or Categorical (either single-factor or AoA)]:
        Posterior predictive density over hidden states
    Returns
    -------
    infogain_pA [scalar]:
        Surprise (about dirichlet parameters) expected under the policy in question
    '''

    if isinstance(pA,Dirichlet):
        if pA.IS_AOA:
            Ng = pA.shape[0]
        else:
            Ng = 1
        wA = pA.expectation_of_log(return_numpy=True)
        pA = pA.values
    elif pA.dtype == 'object':
        Ng = len(pA)
        wA = np.empty(Ng,dtype=object)
        for g in range(Ng):
            wA[g] = spm_wnorm(pA[g])
    else:
        Ng = 1
        wA = spm_wnorm(pA)

    if isinstance(Qo_pi,Categorical):
        Qo_pi = Qo_pi.values
        
    if Qo_pi.dtype == 'object':
        for g in range(len(Qo_pi)):
            Qo_pi[g] = Qo_pi[g].flatten()
    else:
        Qo_pi = Qo_pi.flatten()
    
    if isinstance(Qs_pi,Categorical):
        Qs_pi = Qs_pi.values
    
    if Qs_pi.dtype == 'object':
        for f in range(len(Qs_pi)):
            Qs_pi[f] = Qs_pi[f].flatten()
    else:
        Qs_pi = Qs_pi.flatten()
    
    if Ng > 1:

        infogain_pA = 0

        for g in range(Ng):
            wA_g= wA[g] * (pA[g] > 0).astype('float')
            infogain_pA -= Qo_pi[g].dot(spm_dot(wA_g,Qs_pi)[:,np.newaxis])
    
    else:

        wA = wA * (pA > 0).astype('float')
        infogain_pA = -Qo_pi.dot(spm_dot(wA,Qs_pi)[:,np.newaxis])
    
    return infogain_pA

def calculate_infogain_pB(pB, Qs_next, Qs_previous, policy):
    '''
    Compute expected Dirichlet information gain about parameters pB under a given policy
    Parameters
    ----------
    pB [numpy nd-array, array-of-arrays (where each entry is a numpy nd-array), or Dirichlet (either single-factor of AoA)]:
        Prior dirichlet parameters parameterizing beliefs about the likelihood describing transitions bewteen hidden states,
        with different factors (if there are multiple) stored in different arrays.
    Qs_next [numpy 1D array, array-of-arrays (where each entry is a numpy 1D array), or Categorical (either single-factor or AoA)]:
        Posterior predictive density over hidden states under some policy
    Qs_previous [numpy 1D array, array-of-arrays (where each entry is a numpy 1D array), or Categorical (either single-factor or AoA)]:
        Posterior predictive density over hidden states (prior to observations)
    Returns
    -------
    infogain_pB [scalar]:
        Surprise (about dirichlet parameters) expected under the policy in question
    '''
    if isinstance(pB,Dirichlet):
        if pB.IS_AOA:
            Nf = pB.shape[0]
        else:
            Nf = 1
        wB = pB.expectation_of_log(return_numpy=True)
        pB = pB.values
    elif pB.dtype == 'object':
        Nf = len(pB)
        wB = np.empty(Nf,dtype=object)
        for f in range(Nf):
            wB[f] = spm_wnorm(pB[f])
    else:
        Nf = 1
        wB = spm_wnorm(pB)

    if isinstance(Qs_next,Categorical):
        Qs_next = Qs_next.values
        
    if Qs_next.dtype == 'object':
        for f in range(Nf):
            Qs_next[f] = Qs_next[f].flatten()
    else:
        Qs_next = Qs_next.flatten()
    
    if isinstance(Qs_previous,Categorical):
        Qs_previous = Qs_previous.values
    
    if Qs_previous.dtype == 'object':
        for f in range(Nf):
            Qs_previous[f] = Qs_previous[f].flatten()
    else:
        Qs_previous = Qs_previous.flatten()
    
    if Nf > 1:

        infogain_pB = 0

        for f_i, a_i in enumerate(policy):
            wB_action = wB[f_i][:,:,a_i] * (pB[f_i][:,:,a_i]  > 0).astype('float')
            infogain_pB -= Qs_next[f_i].dot(wB_action.dot(Qs_previous[f_i]))     
    
    else:

        a_i = policy[0]
        
        wB = wB[:,:,a_i] * (pB[:,:,a_i] > 0).astype('float')
        infogain_pB = -Qs_next.dot(wB.dot(Qs_previous))
    
    return infogain_pB

def sample_action(p_i, possiblePolicies, Nu, sampling_type = 'marginal_action'):
    '''
    Samples action from posterior over policies, using one of two methods. 
    @TODO: Needs to be amended for use with multi-step policies (where possiblePolicies is a list of np.arrays (nStep x nFactor), not just a list of tuples as it is now)
    Parameters
    ----------
    p_i [1D numpy.ndarray or Categorical]:
        Variational posterior over policies.
    possiblePolicies [list of tuples]:
        List of tuples that indicate the possible policies under consideration. Each tuple stores the actions taken upon the separate hidden state factors. 
        Same length as p_i.
    Nu [list of integers]:
        List of the dimensionalities of the different (controllable)) hidden states
    sampling_type [string, 'marginal_action' or 'posterior_sample']:
        Indicates whether the sampled action for a given hidden state factor is given by the evidence for that action, marginalized across different policies ('marginal_action')
        or simply the action entailed by the policy sampled from the posterior. 
    Returns
    ----------
    selectedPolicy [tuple]:
        tuple containing the list of actions selected by the agent
    '''
 
    numControls = len(Nu)

    if sampling_type == 'marginal_action':
        action_marginals = np.empty(numControls, dtype=object)
        for nu_i in range(numControls):
            action_marginals[nu_i] = np.zeros(Nu[nu_i])

        # Weight each action according to the posterior probability it gets across policies
        for pol_i, policy in enumerate(possiblePolicies):
                for nu_i, a_i in enumerate(policy):
                    action_marginals[nu_i][a_i] += p_i[pol_i]
        
        action_marginals = Categorical(values = action_marginals)
        action_marginals.normalize()
        selectedPolicy = action_marginals.sample()

    elif sampling_type == 'posterior_sample':
        if isinstance(p_i,Categorical):
            policy_index = p_i.sample()
            selectedPolicy = possiblePolicies[policy_index]
        else:
            sample_onehot = np.random.multinomial(1, p_i.squeeze())
            policy_index = np.where(sample_onehot == 1)[0][0]
            selectedPolicy = possiblePolicies[policy_index]
    
    return selectedPolicy

def update_dirichletA(pA, A, obs, Qs, eta = 1.0, return_numpy = True, which_modalities = 'all'):
    """
    Update Dirichlet parameters that parameterize the observation model of the generative model (describing the probabilistic mapping from hidden states to observations).
    Parameters
    -----------
    pA [numpy nd.array, array-of-arrays (with np.ndarray entries), or Dirichlet (either single-modality or AoA)]:
        The prior Dirichlet parameters of the generative model, parameterizing the agent's beliefs about the observation likelihood. 
    A [numpy nd.array, object-like array of arrays, or Categorical (either single-modality or AoA)]:
        The observation likelihood of the generative model. 
    obs [numpy 1D array, array-of-arrays (with 1D numpy array entries), int or tuple]:
        A discrete observation used in the update equation
    Qs [numpy 1D array, array-of-arrays (where each entry is a numpy 1D array), or Categorical (either single-factor or AoA)]:
        Current marginal posterior beliefs about hidden state factors
    eta [float, optional]:
        Learning rate.
    return_numpy [bool, optional]:
        Logical flag to determine whether output is a numpy array or a Dirichlet
    which_modalities [list, optional]:
        Indices (in terms of range(Ng)) of the observation modalities to include in learning.
        Defaults to 'all, meaning that observation likelihood matrices for all modalities
        are updated as a function of observations in the different modalities.
    """

    if isinstance(pA,Dirichlet):
        if pA.IS_AOA:
            Ng = len(pA)
            No = [pA[g].shape[0] for g in range(Ng)]
        else:
            Ng = 1
            No = [pA.shape[0]]
        if return_numpy:
            pA_new = pA.values.copy()
        else:
            pA_new = Dirichlet(values = pA.values.copy())
    
    else:
        if pA.dtype == object:
            Ng = len(pA)
            No = [pA[g].shape[0] for g in range(Ng)]    
        else:
            Ng = 1
            No = [pA.shape[0]]
        if return_numpy:
            pA_new = pA.copy()
        else:
            pA_new = Dirichlet(values = pA.copy())
    
    if isinstance(A, Categorical):
        A = A.values

    if isinstance(obs, (int, np.integer)):
        obs = np.eye(A.shape[0])[obs]
    
    elif isinstance(obs, tuple):
        obs = np.array( [ np.eye(No[g])[obs[g]] for g in range(Ng) ], dtype = object )
    
    obs = Categorical(values = obs) # convert to Categorical to make the cross product easier

    if which_modalities == 'all':
        if Ng == 1:
            da = obs.cross(Qs, return_numpy = True)
            da = da * (A > 0).astype('float')
            pA_new = pA_new + (eta * da)
        elif Ng > 1:
            for g in range(Ng):
                da = obs[g].cross(Qs,return_numpy=True)
                da = da * (A[g] > 0).astype('float')
                pA_new[g] = pA_new[g] + (eta * da)
    else:
        for g_idx in which_modalities:
            da = obs[g_idx].cross(Qs, return_numpy = True)
            da = da * (A[g_idx] > 0).astype('float')
            pA_new[g_idx] = pA_new[g_idx] + (eta * da)
    
    return pA_new

def update_dirichletB(pB, B, action, Qs_curr, Qs_prev, eta = 1.0, return_numpy = True, which_factors = 'all'):
    """
    Update Dirichlet parameters that parameterize the transition model of the generative model (describing the probabilistic mapping between hidden states over time).
    Parameters
    -----------
    pB [numpy nd.array, array-of-arrays (with np.ndarray entries), or Dirichlet (either single-modality or AoA)]:
        The prior Dirichlet parameters of the generative model, parameterizing the agent's beliefs about the transition likelihood. 
    B [numpy nd.array, object-like array of arrays, or Categorical (either single-modality or AoA)]:
        The transition likelihood of the generative model. 
    action [tuple]:
        A tuple containing the action(s) performed at a given timestep.
    Qs_curr [numpy 1D array, array-of-arrays (where each entry is a numpy 1D array), or Categorical (either single-factor or AoA)]:
        Current marginal posterior beliefs about hidden state factors
    Qs_prev [numpy 1D array, array-of-arrays (where each entry is a numpy 1D array), or Categorical (either single-factor or AoA)]:
        Past marginal posterior beliefs about hidden state factors
    eta [float, optional]:
        Learning rate.
    return_numpy [bool, optional]:
        Logical flag to determine whether output is a numpy array or a Dirichlet
    which_factors [list, optional]:
        Indices (in terms of range(Nf)) of the hidden state factors to include in learning.
        Defaults to 'all', meaning that transition likelihood matrices for all hidden state factors
        are updated as a function of transitions in the different control factors (i.e. actions)
    """

    if isinstance(pB, Dirichlet):
        if pB.IS_AOA:
            Nf = len(pB)
            Ns = [pB[f].shape[0] for f in range(Nf)]
        else:
            Nf = 1
            Ns = [pB.shape[0]]
        if return_numpy:
            pB_new = pB.values.copy()
        else:
            pB_new = Dirichlet(values = pB.values.copy())
    
    else:
        if pB.dtype == object:
            Nf = len(pB)
            Ns = [pB[f].shape[0] for f in range(Nf)]    
        else:
            Nf = 1
            Ns = [pB.shape[0]]
        if return_numpy:
            pB_new = pB.copy()
        else:
            pB_new = Dirichlet(values = pB.copy())
    
    if isinstance(B, Categorical):
        B = B.values
    
    if not isinstance(Qs_curr, Categorical):
        Qs_curr = Categorical(values = Qs_curr)
    
    if which_factors == 'all':
        if Nf == 1:
            db = Qs_curr.cross(Qs_prev, return_numpy = True)
            db = db * (B[:,:,action[0]] > 0).astype('float')
            pB_new = pB_new + (eta * db)
        elif Nf > 1:
            for f in range(Nf):
                db = Qs_curr[f].cross(Qs_prev[f],return_numpy=True)
                db = db * (B[f][:,:,action[f]] > 0).astype('float')
                pB_new[f] = pB_new[f] + (eta * db)
    else:
        for f_idx in which_factors:
            db = Qs_curr[f_idx].cross(Qs_prev[f_idx], return_numpy = True)
            db = db * (B[f_idx][:,:,action[f_idx]] > 0).astype('float')
            pB_new[f_idx] = pB_new[f_idx] + (eta * db)
    
    return pB_new
    
def spm_MDP_G(A, x):
    """
    Calculates the Bayesian surprise in the same way as spm_MDP_G.m does in 
    the original matlab code.
    
    Parameters
    ----------
    A (numpy ndarray or array-object):
        array assigning likelihoods of observations/outcomes under the various hidden state configurations
    
    x (numpy ndarray or array-object):
        Categorical distribution presenting probabilities of hidden states (this can also be interpreted as the 
        predictive density over hidden states/causes if you're calculating the 
        expected Bayesian surprise)
        
    Returns
    -------
    G (float):
        the (expected or not) Bayesian surprise under the density specified by x --
        namely, this scores how much an expected observation would update beliefs about hidden states
        x, were it to be observed. 
    """
    if A.dtype == "object":
        Ng = len(A)
        AOA_flag = True
    else:
        Ng = 1
        AOA_flag = False

    # probability distribution over the hidden causes: i.e., Q(x)
    qx = spm_cross(x)
    G = 0
    qo = 0
    idx = np.array(np.where(qx > np.exp(-16))).T

    if AOA_flag:
        # accumulate expectation of entropy: i.e., E[lnP(o|x)]
        for i in idx:
            # probability over outcomes for this combination of causes
            po = np.ones(1)
            for g in range(Ng):
                index_vector = [slice(0, A[g].shape[0])] + list(i)
                po = spm_cross(po, A[g][tuple(index_vector)])

            po = po.ravel()
            qo += qx[tuple(i)] * po
            G += qx[tuple(i)] * po.dot(np.log(po + np.exp(-16)))
    else:
        for i in idx:

            po = np.ones(1)
            index_vector = [slice(0, A.shape[0])] + list(i)
            po = spm_cross(po, A[tuple(index_vector)])
            po = po.ravel()
            qo += qx[tuple(i)] * po
            G += qx[tuple(i)] * po.dot(np.log(po + np.exp(-16)))

    # subtract negative entropy of expectations: i.e., E[lnQ(o)]
    G = G - qo.dot(np.log(qo + np.exp(-16)))

    return G

def cross_product_beta(dist_a, dist_b):
    """
    @TODO: needs to be replaced by spm_cross
    """
    if not isinstance(type(dist_a), type(Categorical)) or not isinstance(type(dist_b), type(Categorical)):
        raise ValueError(
            '[cross_product] function takes [Categorical] objects')
    values_a = np.copy(dist_a.values)
    values_b = np.copy(dist_b.values)
    a = np.reshape(values_a, (values_a.shape[0], values_a.shape[1], 1, 1))
    b = np.reshape(values_b, (1, 1, values_b.shape[0], values_b.shape[1]))
    values = np.squeeze(a * b)
    return values
