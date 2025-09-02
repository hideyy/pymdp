#!/usr/bin/env python
# -*- coding: utf-8 -*-
# pylint: disable=no-member
# pylint: disable=not-an-iterable

import itertools
import jax.numpy as jnp
import jax.tree_util as jtu
from typing import List, Tuple, Optional
from functools import partial
from jax.scipy.special import xlogy
from jax import lax, jit, vmap, nn
from jax import random as jr
from itertools import chain
from jaxtyping import Array

from pymdp.jax.maths import *
# import pymdp.jax.utils as utils

def get_marginals(q_pi, policies, num_controls):
    """
    Computes the marginal posterior(s) over actions by integrating their posterior probability under the policies that they appear within.

    Parameters
    ----------
    q_pi: 1D ``numpy.ndarray``
        Posterior beliefs over policies, i.e. a vector containing one posterior probability per policy.
    policies: ``list`` of 2D ``numpy.ndarray``
        ``list`` that stores each policy as a 2D array in ``policies[p_idx]``. Shape of ``policies[p_idx]`` 
        is ``(num_timesteps, num_factors)`` where ``num_timesteps`` is the temporal
        depth of the policy and ``num_factors`` is the number of control factors.
    num_controls: ``list`` of ``int``
        ``list`` of the dimensionalities of each control state factor.
    
    Returns
    ----------
    action_marginals: ``list`` of ``jax.numpy.ndarrays``
       List of arrays corresponding to marginal probability of each action possible action
    """
    num_factors = len(num_controls)    

    action_marginals = []
    for factor_i in range(num_factors):
        actions = jnp.arange(num_controls[factor_i])[:, None]
        action_marginals.append(jnp.where(actions==policies[:, 0, factor_i], q_pi, 0).sum(-1))
    
    return action_marginals

def sample_action(policies, num_controls, q_pi, action_selection="deterministic", alpha=16.0, rng_key=None):
    """
    Samples an action from posterior marginals, one action per control factor.

    Parameters
    ----------
    q_pi: 1D ``numpy.ndarray``
        Posterior beliefs over policies, i.e. a vector containing one posterior probability per policy.
    policies: ``list`` of 2D ``numpy.ndarray``
        ``list`` that stores each policy as a 2D array in ``policies[p_idx]``. Shape of ``policies[p_idx]`` 
        is ``(num_timesteps, num_factors)`` where ``num_timesteps`` is the temporal
        depth of the policy and ``num_factors`` is the number of control factors.
    num_controls: ``list`` of ``int``
        ``list`` of the dimensionalities of each control state factor.
    action_selection: string, default "deterministic"
        String indicating whether whether the selected action is chosen as the maximum of the posterior over actions,
        or whether it's sampled from the posterior marginal over actions
    alpha: float, default 16.0
        Action selection precision -- the inverse temperature of the softmax that is used to scale the 
        action marginals before sampling. This is only used if ``action_selection`` argument is "stochastic"

    Returns
    ----------
    selected_policy: 1D ``numpy.ndarray``
        Vector containing the indices of the actions for each control factor
    """

    marginal = get_marginals(q_pi, policies, num_controls)
    
    if action_selection == 'deterministic':
        selected_policy = jtu.tree_map(lambda x: jnp.argmax(x, -1), marginal)
    elif action_selection == 'stochastic':
        logits = lambda x: alpha * log_stable(x)
        selected_policy = jtu.tree_map(lambda x: jr.categorical(rng_key, logits(x)), marginal)
    else:
        raise NotImplementedError

    return jnp.array(selected_policy)

def sample_policy(policies, q_pi, action_selection="deterministic", alpha = 16.0, rng_key=None):

    if action_selection == "deterministic":
        policy_idx = jnp.argmax(q_pi)
    elif action_selection == "stochastic":
        log_p_policies = log_stable(q_pi) * alpha
        policy_idx = jr.categorical(rng_key, log_p_policies)

    ##selected_multiaction = policies[policy_idx, 0]
    selected_multiaction = jnp.take(policies, policy_idx, axis=0)[0]##25/05/26宮口修正
    return selected_multiaction

def construct_policies(num_states, num_controls = None, policy_len=1, control_fac_idx=None):
    """
    Generate a ``list`` of policies. The returned array ``policies`` is a ``list`` that stores one policy per entry.
    A particular policy (``policies[i]``) has shape ``(num_timesteps, num_factors)`` 
    where ``num_timesteps`` is the temporal depth of the policy and ``num_factors`` is the number of control factors.

    Parameters
    ----------
    num_states: ``list`` of ``int``
        ``list`` of the dimensionalities of each hidden state factor
    num_controls: ``list`` of ``int``, default ``None``
        ``list`` of the dimensionalities of each control state factor. If ``None``, then is automatically computed as the dimensionality of each hidden state factor that is controllable
    policy_len: ``int``, default 1
        temporal depth ("planning horizon") of policies
    control_fac_idx: ``list`` of ``int``
        ``list`` of indices of the hidden state factors that are controllable (i.e. those state factors ``i`` where ``num_controls[i] > 1``)

    Returns
    ----------
    policies: ``list`` of 2D ``numpy.ndarray``
        ``list`` that stores each policy as a 2D array in ``policies[p_idx]``. Shape of ``policies[p_idx]`` 
        is ``(num_timesteps, num_factors)`` where ``num_timesteps`` is the temporal
        depth of the policy and ``num_factors`` is the number of control factors.
    """

    num_factors = len(num_states)
    if control_fac_idx is None:
        if num_controls is not None:
            control_fac_idx = [f for f, n_c in enumerate(num_controls) if n_c > 1]
        else:
            control_fac_idx = list(range(num_factors))

    if num_controls is None:
        num_controls = [num_states[c_idx] if c_idx in control_fac_idx else 1 for c_idx in range(num_factors)]
        
    x = num_controls * policy_len
    policies = list(itertools.product(*[list(range(i)) for i in x]))
    
    for pol_i in range(len(policies)):
        policies[pol_i] = jnp.array(policies[pol_i]).reshape(policy_len, num_factors)

    return jnp.stack(policies)


def update_posterior_policies(policy_matrix, qs_init, A, B, C, E, pA, pB, A_dependencies, B_dependencies, gamma=16.0, use_utility=True, use_states_info_gain=True, use_param_info_gain=False):
    # policy --> n_levels_factor_f x 1
    # factor --> n_levels_factor_f x n_policies
    ## vmap across policies
    compute_G_fixed_states = partial(compute_G_policy, qs_init, A, B, C, pA, pB, A_dependencies, B_dependencies,
                                     use_utility=use_utility, use_states_info_gain=use_states_info_gain, use_param_info_gain=use_param_info_gain)

    # only in the case of policy-dependent qs_inits
    # in_axes_list = (1,) * n_factors
    # all_efe_of_policies = vmap(compute_G_policy, in_axes=(in_axes_list, 0))(qs_init_pi, policy_matrix)

    # policies needs to be an NDarray of shape (n_policies, n_timepoints, n_control_factors)
    neg_efe_all_policies = vmap(compute_G_fixed_states)(policy_matrix)

    return nn.softmax(gamma * neg_efe_all_policies + log_stable(E)), neg_efe_all_policies

def compute_expected_state(qs_prior, B, u_t, B_dependencies=None): 
    """
    Compute posterior over next state, given belief about previous state, transition model and action...
    """
    #Note: this algorithm is only correct if each factor depends only on itself. For any interactions, 
    # we will have empirical priors with codependent factors. 
    assert len(u_t) == len(B)  
    qs_next = []
    for B_f, u_f, deps in zip(B, u_t, B_dependencies):
        relevant_factors = [qs_prior[idx] for idx in deps]
        qs_next_f = factor_dot(B_f[...,u_f], relevant_factors, keep_dims=(0,))
        qs_next.append(qs_next_f)
        
    # P(s'|s, u) = \sum_{s, u} P(s'|s) P(s|u) P(u|pi)P(pi) because u </-> pi
    return qs_next

def compute_expected_state_and_Bs(qs_prior, B, u_t): 
    """
    Compute posterior over next state, given belief about previous state, transition model and action...
    """
    assert len(u_t) == len(B)  
    qs_next = []
    Bs = []
    for qs_f, B_f, u_f in zip(qs_prior, B, u_t):
        qs_next.append( B_f[..., u_f].dot(qs_f) )
        Bs.append(B_f[..., u_f])
    
    return qs_next, Bs

def compute_expected_obs(qs, A, A_dependencies):
    """
    New version of expected observation (computation of Q(o|pi)) that takes into account sparse dependencies between observation
    modalities and hidden state factors
    """
        
    def compute_expected_obs_modality(A_m, m):
        deps = A_dependencies[m]
        relevant_factors = [qs[idx] for idx in deps]
        return factor_dot(A_m, relevant_factors, keep_dims=(0,))

    return jtu.tree_map(compute_expected_obs_modality, A, list(range(len(A))))

def compute_info_gain(qs, qo, A, A_dependencies):
    """
    New version of expected information gain that takes into account sparse dependencies between observation modalities and hidden state factors.
    """

    def compute_info_gain_for_modality(qo_m, A_m, m):
        H_qo = stable_entropy(qo_m)#Calculate predictied entropyの計算
        H_A_m = - stable_xlogx(A_m).sum(0)#観測モデル（A,）p(o|s)のエントロピーを計算し，o方向に和を取る．;Calculate the entropy of the observation model (A, p(o|s)) and sum in the direction of o.
        deps = A_dependencies[m]
        relevant_factors = [qs[idx] for idx in deps]
        qs_H_A_m = factor_dot(H_A_m, relevant_factors)#Ambiguityの計算．q(s|π)とH_A_mの内積．;Calculation of ambiguity. Inner product of q(s|π) and H_A_m.
        return H_qo - qs_H_A_m
    
    info_gains_per_modality = jtu.tree_map(compute_info_gain_for_modality, qo, A, list(range(len(A))))
        
    return jtu.tree_reduce(lambda x,y: x+y, info_gains_per_modality)

def compute_expected_utility(t, qo, C):
    
    util = 0.
    for o_m, C_m in zip(qo, C):
        if C_m.ndim > 1:
            util += (o_m * C_m[t]).sum()
        else:
            util += (o_m * C_m).sum()
    
    return util

def calc_pA_info_gain(pA, qo, qs, A_dependencies):
    """
    Compute expected Dirichlet information gain about parameters ``pA`` for a given posterior predictive distribution over observations ``qo`` and states ``qs``.

    Parameters
    ----------
    pA: ``numpy.ndarray`` of dtype object
        Dirichlet parameters over observation model (same shape as ``A``)
    qo: ``list`` of ``numpy.ndarray`` of dtype object
        Predictive posterior beliefs over observations; stores the beliefs about
        observations expected under the policy at some arbitrary time ``t``
    qs: ``list`` of ``numpy.ndarray`` of dtype object
        Predictive posterior beliefs over hidden states, stores the beliefs about
        hidden states expected under the policy at some arbitrary time ``t``

    Returns
    -------
    infogain_pA: float
        Surprise (about Dirichlet parameters) expected for the pair of posterior predictive distributions ``qo`` and ``qs``
    """

    def infogain_per_modality(pa_m, qo_m, m):
        wa_m = spm_wnorm(pa_m) * (pa_m > 0.) #ディリクリパラメータの逆数のようなもの．パラメータが蓄積されているほど小さい．;Something like the reciprocal of the Dirichlet parameter. The more parameters are accumulated, the smaller it becomes.
        fd = factor_dot(wa_m, [s for f, s in enumerate(qs) if f in A_dependencies[m]], keep_dims=(0,))[..., None]#wa_mとq(s|π)の内積;inner product of wa_m and q(s|π)
        return qo_m.dot(fd)#fdとq(o|π)の内積？;The inner product of fd and q(o|π)?

    pA_infogain_per_modality = jtu.tree_map(
        infogain_per_modality, pA, qo, list(range(len(qo)))
    )
    
    infogain_pA = jtu.tree_reduce(lambda x, y: x + y, pA_infogain_per_modality)
    return infogain_pA.squeeze(-1)

def calc_pB_info_gain(pB, qs_t, qs_t_minus_1, B_dependencies, u_t_minus_1):
    """
    Compute expected Dirichlet information gain about parameters ``pB`` under a given policy

    Parameters
    ----------
    pB: ``Array`` of dtype object
        Dirichlet parameters over transition model (same shape as ``B``)
    qs_t: ``list`` of ``Array`` of dtype object
        Predictive posterior beliefs over hidden states expected under the policy at time ``t``
    qs_t_minus_1: ``list`` of ``Array`` of dtype object
        Posterior over hidden states at time ``t-1`` (before receiving observations)
    u_t_minus_1: "Array"
        Actions in time step t-1 for each factor

    Returns
    -------
    infogain_pB: float
        Surprise (about Dirichlet parameters) expected under the policy in question
    """
    
    wB = lambda pb:  spm_wnorm(pb) * (pb > 0.)
    fd = lambda x, i: factor_dot(x, [s for f, s in enumerate(qs_t_minus_1) if f in B_dependencies[i]], keep_dims=(0,))[..., None]
    
    pB_infogain_per_factor = jtu.tree_map(lambda pb, qs, f: qs.dot(fd(wB(pb[..., u_t_minus_1[f]]), f)), pB, qs_t, list(range(len(qs_t))))
    infogain_pB = jtu.tree_reduce(lambda x, y: x + y, pB_infogain_per_factor)[0]
    return infogain_pB

def compute_G_policy(qs_init, A, B, C, pA, pB, A_dependencies, B_dependencies, policy_i, use_utility=True, use_states_info_gain=True, use_param_info_gain=False):
    """ Write a version of compute_G_policy that does the same computations as `compute_G_policy` but using `lax.scan` instead of a for loop. """

    def scan_body(carry, t):

        qs, neg_G = carry

        qs_next = compute_expected_state(qs, B, policy_i[t], B_dependencies)

        qo = compute_expected_obs(qs_next, A, A_dependencies)

        info_gain = compute_info_gain(qs_next, qo, A, A_dependencies) if use_states_info_gain else 0.

        utility = compute_expected_utility(qo, C) if use_utility else 0.

        param_info_gain = calc_pA_info_gain(pA, qo, qs_next) if use_param_info_gain else 0.
        param_info_gain += calc_pB_info_gain(pB, qs_next, qs, policy_i[t]) if use_param_info_gain else 0.

        neg_G += info_gain + utility + param_info_gain

        return (qs_next, neg_G), None

    qs = qs_init
    neg_G = 0.
    final_state, _ = lax.scan(scan_body, (qs, neg_G), jnp.arange(policy_i.shape[0]))
    qs_final, neg_G = final_state
    return neg_G

def compute_G_policy_inductive(qs_init, A, B, C, pA, pB, A_dependencies, B_dependencies, I, policy_i, inductive_epsilon=1e-3, use_utility=True, use_states_info_gain=True, use_param_info_gain=False, use_inductive=False):
    """ 
    Write a version of compute_G_policy that does the same computations as `compute_G_policy` but using `lax.scan` instead of a for loop.
    This one further adds computations used for inductive planning.
    """

    def scan_body(carry, t):

        qs, neg_G = carry

        qs_next = compute_expected_state(qs, B, policy_i[t], B_dependencies)

        qo = compute_expected_obs(qs_next, A, A_dependencies)

        info_gain = compute_info_gain(qs_next, qo, A, A_dependencies) if use_states_info_gain else 0.

        utility = compute_expected_utility(t, qo, C) if use_utility else 0.

        inductive_value = calc_inductive_value_t(qs_init, qs_next, I, epsilon=inductive_epsilon) if use_inductive else 0.

        param_info_gain = 0.
        if pA is not None:
            param_info_gain += calc_pA_info_gain(pA, qo, qs_next, A_dependencies) if use_param_info_gain else 0.
        if pB is not None:
            param_info_gain += calc_pB_info_gain(pB, qs_next, qs, B_dependencies, policy_i[t]) if use_param_info_gain else 0.

        neg_G += info_gain + utility - param_info_gain + inductive_value

        return (qs_next, neg_G), None

    qs = qs_init
    neg_G = 0.
    final_state, _ = lax.scan(scan_body, (qs, neg_G), jnp.arange(policy_i.shape[0]))
    _, neg_G = final_state
    return neg_G

def update_posterior_policies_inductive(policy_matrix, qs_init, A, B, C, E, pA, pB, A_dependencies, B_dependencies, I, gamma=16.0, inductive_epsilon=1e-3, use_utility=True, use_states_info_gain=True, use_param_info_gain=False, use_inductive=True):
    # policy --> n_levels_factor_f x 1
    # factor --> n_levels_factor_f x n_policies
    ## vmap across policies
    compute_G_fixed_states = partial(compute_G_policy_inductive, qs_init, A, B, C, pA, pB, A_dependencies, B_dependencies, I, inductive_epsilon=inductive_epsilon,
                                     use_utility=use_utility,  use_states_info_gain=use_states_info_gain, use_param_info_gain=use_param_info_gain, use_inductive=use_inductive)

    # only in the case of policy-dependent qs_inits
    # in_axes_list = (1,) * n_factors
    # all_efe_of_policies = vmap(compute_G_policy, in_axes=(in_axes_list, 0))(qs_init_pi, policy_matrix)

    # policies needs to be an NDarray of shape (n_policies, n_timepoints, n_control_factors)
    neg_efe_all_policies = vmap(compute_G_fixed_states)(policy_matrix)

    return nn.softmax(gamma * neg_efe_all_policies + log_stable(E)), neg_efe_all_policies

def generate_I_matrix(H: List[Array], B: List[Array], threshold: float, depth: int):
    """ 
    Generates the `I` matrices used in inductive planning. These matrices stores the probability of reaching the goal state backwards from state j (columns) after i (rows) steps.
    Parameters
    ----------    
    H: ``list`` of ``jax.numpy.ndarray``
        Constraints over desired states (1 if you want to reach that state, 0 otherwise)
    B: ``list`` of ``jax.numpy.ndarray``
        Dynamics likelihood mapping or 'transition model', mapping from hidden states at ``t`` to hidden states at ``t+1``, given some control state ``u``.
        Each element ``B[f]`` of this object array stores a 3-D tensor for hidden state factor ``f``, whose entries ``B[f][s, v, u]`` store the probability
        of hidden state level ``s`` at the current time, given hidden state level ``v`` and action ``u`` at the previous time.
    threshold: ``float``
        The threshold for pruning transitions that are below a certain probability
    depth: ``int``
        The temporal depth of the backward induction

    Returns
    ----------
    I: ``numpy.ndarray`` of dtype object
        For each state factor, contains a 2D ``numpy.ndarray`` whose element i,j yields the probability 
        of reaching the goal state backwards from state j after i steps.
    """
    
    num_factors = len(H)
    I = []
    for f in range(num_factors):
        """
        For each factor, we need to compute the probability of reaching the goal state
        """

        # If there exists an action that allows transitioning 
        # from state to next_state, with probability larger than threshold
        # set b_reachable[current_state, previous_state] to 1
        b_reachable = jnp.where(B[f] > threshold, 1.0, 0.0).sum(axis=-1)
        b_reachable = jnp.where(b_reachable > 0., 1.0, 0.0)

        def step_fn(carry, i):
            I_prev = carry
            I_next = jnp.dot(b_reachable, I_prev)
            I_next = jnp.where(I_next > 0.1, 1.0, 0.0) # clamp I_next to 1.0 if it's above 0.1, 0 otherwise
            return I_next, I_next
    
        _, I_f = lax.scan(step_fn, H[f], jnp.arange(depth-1))
        I_f = jnp.concatenate([H[f][None,...], I_f], axis=0)

        I.append(I_f)
    
    return I

def calc_inductive_value_t(qs, qs_next, I, epsilon=1e-3):
    """
    Computes the inductive value of a state at a particular time (translation of @tverbele's `numpy` implementation of inductive planning, formerly
    called `calc_inductive_cost`).

    Parameters
    ----------
    qs: ``list`` of ``jax.numpy.ndarray`` 
        Marginal posterior beliefs over hidden states at a given timepoint.
    qs_next: ```list`` of ``jax.numpy.ndarray`` 
        Predictive posterior beliefs over hidden states expected under the policy.
    I: ``numpy.ndarray`` of dtype object
        For each state factor, contains a 2D ``numpy.ndarray`` whose element i,j yields the probability 
        of reaching the goal state backwards from state j after i steps.
    epsilon: ``float``
        Value that tunes the strength of the inductive value (how much it contributes to the expected free energy of policies)

    Returns
    -------
    inductive_val: float
        Value (negative inductive cost) of visiting this state using backwards induction under the policy in question
    """
    
    # initialise inductive value
    inductive_val = 0.

    log_eps = log_stable(epsilon)
    for f in range(len(qs)):
        # we also assume precise beliefs here?!
        idx = jnp.argmax(qs[f])
        # m = arg max_n p_n < sup p

        # i.e. find first entry at which I_idx equals 1, and then m is the index before that
        m = jnp.maximum(jnp.argmax(I[f][:, idx])-1, 0)
        I_m = (1. - I[f][m, :]) * log_eps
        path_available = jnp.clip(I[f][:, idx].sum(0), min=0, max=1) # if there are any 1's at all in that column of I, then this == 1, otherwise 0
        inductive_val += path_available * I_m.dot(qs_next[f]) # scaling by path_available will nullify the addition of inductive value in the case we find no path to goal (i.e. when no goal specified)

    return inductive_val

# if __name__ == '__main__':

#     from jax import random as jr
#     key = jr.PRNGKey(1)
#     num_obs = [3, 4]

#     A = [jr.uniform(key, shape = (no, 2, 2)) for no in num_obs]
#     B = [jr.uniform(key, shape = (2, 2, 2)), jr.uniform(key, shape = (2, 2, 2))]
#     C = [log_stable(jnp.array([0.8, 0.1, 0.1])), log_stable(jnp.ones(4)/4)]
#     policy_1 = jnp.array([[0, 1],
#                          [1, 1]])
#     policy_2 = jnp.array([[1, 0],
#                          [0, 0]])
#     policy_matrix = jnp.stack([policy_1, policy_2]) # 2 x 2 x 2 tensor
    
#     qs_init = [jnp.ones(2)/2, jnp.ones(2)/2]
#     neg_G_all_policies = jit(update_posterior_policies)(policy_matrix, qs_init, A, B, C)
#     print(neg_G_all_policies)

def update_posterior_policies_inductive_efe(policy_matrix, qs_init, A, B, C, E, pA, pB, A_dependencies, B_dependencies, I, gamma=16.0, inductive_epsilon=1e-3, use_utility=True, use_states_info_gain=True, use_param_info_gain=False, use_inductive=True):
    # policy --> n_levels_factor_f x 1
    # factor --> n_levels_factor_f x n_policies
    ## vmap across policies
    compute_G_fixed_states = partial(compute_G_policy_inductive_efe, qs_init, A, B, C, pA, pB, A_dependencies, B_dependencies, I, inductive_epsilon=inductive_epsilon,
                                     use_utility=use_utility,  use_states_info_gain=use_states_info_gain, use_param_info_gain=use_param_info_gain, use_inductive=use_inductive)

    # policies needs to be an NDarray of shape (n_policies, n_timepoints, n_control_factors)
    results = vmap(compute_G_fixed_states)(policy_matrix)
    
    
    neg_efe_all_policies = results[0]  # 各ポリシーの負の期待自由エネルギー;Negative expected free energy of each policy
    PBS_a_p = results[1]  # 状態情報利得;information gain for states
    PKLD_a_p = results[2]  # 状態情報利得
    PFE_a_p = results[3]  # 
    oRisk_a_p = results[4]  # 
    PBS_pA_a_p = results[5]  # パラメータAに関する情報利得;information gain for pA(parameter of A)
    PBS_pB_a_p = results[6]  # パラメータBに関する情報利得;information gain for pB(parameter of B)
    I_B_o_a_p = results[7]  # パラメータBに関する情報利得;information gain for pB(parameter of B)
    I_B_o_se_a_p = results[8]  # パラメータBに関する情報利得;information gain for pB(parameter of B)
    #print(PBS_a_p)
    # only in the case of policy-dependent qs_inits
    # in_axes_list = (1,) * n_factors
    # all_efe_of_policies = vmap(compute_G_policy, in_axes=(in_axes_list, 0))(qs_init_pi, policy_matrix)

    # policies needs to be an NDarray of shape (n_policies, n_timepoints, n_control_factors)
    #neg_efe_all_policies = vmap(compute_G_fixed_states)(policy_matrix)
    #⇓ポリシーの分布の計算q(π)=softmax(-γG+E)
    return nn.softmax(gamma * neg_efe_all_policies + log_stable(E)), neg_efe_all_policies, PBS_a_p, PKLD_a_p, PFE_a_p, oRisk_a_p, PBS_pA_a_p, PBS_pB_a_p,I_B_o_a_p,I_B_o_se_a_p

def compute_G_policy_inductive_efe(qs_init, A, B, C, pA, pB, A_dependencies, B_dependencies, I, policy_i, inductive_epsilon=1e-3, use_utility=True, use_states_info_gain=True, use_param_info_gain=False, use_inductive=False):
    """ 
    Write a version of compute_G_policy that does the same computations as `compute_G_policy` but using `lax.scan` instead of a for loop.
    This one further adds computations used for inductive planning.
    """

    def scan_body(carry, t):

        #qs, neg_G = carry
        qs, neg_G, info_gain, predicted_KLD, predicted_F, oRisk, param_info_gainA, param_info_gainB,utility, inductive_value, I_B_o,I_B_o_se = carry

        qs_next = compute_expected_state(qs, B, policy_i[t], B_dependencies) #q(s|π)=p(sτ+1|sτ,π)stの計算.stは認識分布;Calculation of q(s|π)=p(sτ+1|sτ,π)st. st is the recognition distribution.

        qo = compute_expected_obs(qs_next, A, A_dependencies) #Calculate q(o|π)=p(o|s)p(sτ+1|sτ,π)st

        info_gain += compute_info_gain(qs_next, qo, A, A_dependencies) if use_states_info_gain else 0.#Calculate pBS(epistemic value) #compute_predicted_KLD(qs_next, qo, A, A_dependencies)

        predicted_KLD += compute_predicted_KLD(qs_next, qo, A, A_dependencies) #Calculate pKLD
        #print("PFE")
        predicted_F += compute_predicted_free_energy(qs_next, qo, A, A_dependencies) #Calculate predicted free energy
        #print("Risk")
        oRisk += compute_oRisk(t, qo, C)#Calculate Risk
        utility += compute_expected_utility(t, qo, C) if use_utility else 0.#Calculate utility(Pragmatic value)

        inductive_value += calc_inductive_value_t(qs_init, qs_next, I, epsilon=inductive_epsilon) if use_inductive else 0.
        val1, val2 = calc_pB_o_mutual_info_gain(pB, qs_next, qs_init, B_dependencies, policy_i[t], qo, A, A_dependencies) if use_param_info_gain else (0., 0.)
        I_B_o += val1
        I_B_o_se += val2
        if pA is not None:
            param_info_gainA -= calc_pA_info_gain(pA, qo, qs_next, A_dependencies) if use_param_info_gain else 0.
        else:
            param_info_gainA = 0.
        if pB is not None:
            param_info_gainB -= calc_pB_info_gain(pB, qs_next, qs, B_dependencies, policy_i[t]) if use_param_info_gain else 0.
        else:
            param_info_gainB = 0.

        #neg_G = info_gain + param_info_gainA + param_info_gainB + inductive_value + utility

        neg_G = info_gain + predicted_KLD - predicted_F - oRisk + param_info_gainA + param_info_gainB + inductive_value
        #neg_G += inductive_value 
        return (qs_next, neg_G, info_gain, predicted_KLD, predicted_F, oRisk, param_info_gainA, param_info_gainB, utility, inductive_value, I_B_o,I_B_o_se), None

    qs = qs_init
    #print(qs)
    #print(policy_i)
    neg_G = 0.
    info_gain = 0.
    predicted_KLD = 0.
    predicted_F = 0.
    oRisk = 0.
    param_info_gainA = 0.
    param_info_gainB = 0.
    utility=0.
    inductive_value=0.
    I_B_o=0.
    I_B_o_se=0.
    #ポリシーの深さ分scan_bodyを反復; Iterate scan_body by policy depth
    final_state, _ = lax.scan(scan_body, (qs, neg_G, info_gain, predicted_KLD, predicted_F, oRisk, param_info_gainA, param_info_gainB, utility, inductive_value, I_B_o,I_B_o_se), jnp.arange(policy_i.shape[0]))
    _, neg_G, info_gain, predicted_KLD, predicted_F, oRisk, param_info_gainA, param_info_gainB, utility, inductive_value, I_B_o,I_B_o_se = final_state
    #print(info_gain)
    #print(predicted_KLD)
    """print(predicted_F)
    print(oRisk)
    print(param_info_gainA)
    print(param_info_gainB)

    print(inductive_value) """
    #print(f"I_B_o:",I_B_o)
    #print(f"I_B_o_se:",I_B_o_se)
    #print(neg_G)
    return neg_G, info_gain, predicted_KLD, predicted_F, oRisk, param_info_gainA, param_info_gainB, I_B_o, I_B_o_se

def compute_predicted_KLD(qs, qo, A, A_dependencies):
    
    #print("qs:", qs)
    #print("qo:", qo)
    #print("A:", A)
    #print("A_dependencies:", A_dependencies)
    def compute_pKLD_for_modality(qo_m, A_m, m):
        H_qo = stable_entropy(qo_m)#Calculate predicted entropy計算
        #print(f"H_qo:",H_qo)
        deps = A_dependencies[m]
        relevant_factors = [qs[idx] for idx in deps]
        #relevant_factors = [jnp.array(qs[idx]) for idx in deps]
        #print("qo_ln_A_m")
        #qs_next_f = factor_dot(B_f[...,u_f], relevant_factors, keep_dims=(0,))
        """ 
        relevant_factors = jnp.array([qs[idx] for idx in deps])
        qs_ln_A_m = - stable_cross_entropy(relevant_factors,A_m)
        #print("qo_qs_ln_A_m")
        qo_qs_ln_A_m = -(qo_m * qs_ln_A_m).sum() """
        #qo_ln_A_m = - stable_cross_entropy(qo_m,A_m)
        #log_A_m = jnp.log(A_m)
        log_A_m = log_stable(A_m)
        #print(log_A_m)
        #print(qo_m)
        #qo_ln_A_m =-(qo_m * log_A_m).sum(0)
        qo_ln_A_m = -(jnp.expand_dims(qo_m, axis=tuple(range(1, log_A_m.ndim))) * log_A_m).sum(0)#Σq(o|π)lnp(o|s)
        #qo_ln_A_m = jnp.einsum('i,ijklm->jklm', qo_m, log_A_m)
        #print(f"qo_ln_A_m:",qo_ln_A_m)
        #qo_ln_A_m = - factor_dot(log_A_m, qo_m)
        #print("qo_qs_ln_A_m")
        #print(qo_ln_A_m)
        #qo_qs_ln_A_m = -(qo_m * qs_ln_A_m).sum()
        #print(relevant_factors)
        qo_qs_ln_A_m = factor_dot(qo_ln_A_m, relevant_factors)#Σq(o|π)lnp(o|s)とq(s|π)の内積を取る;Take the inner product of Σq(o|π)lnp(o|s) and q(s|π)
        #print(f"qo_qs_ln_A_m:",qo_qs_ln_A_m)
        #print(qo_qs_ln_A_m - H_qo)
        #qo_qs_ln_A_m = factor_dot(qs_ln_A_m, qo_m)
        
        
        #qo_qs_ln_A_m = factor_dot(H_A_m, relevant_factors)
        return qo_qs_ln_A_m - H_qo
        """ def compute_info_gain(qs, qo, A, A_dependencies):
            

            def compute_info_gain_for_modality(qo_m, A_m, m):
                H_qo = stable_entropy(qo_m)
                H_A_m = - stable_xlogx(A_m).sum(0)
                deps = A_dependencies[m]
                relevant_factors = [qs[idx] for idx in deps]
                qs_H_A_m = factor_dot(H_A_m, relevant_factors)
                return H_qo - qs_H_A_m
            
            info_gains_per_modality = jtu.tree_map(compute_info_gain_for_modality, qo, A, list(range(len(A))))
                
            return jtu.tree_reduce(lambda x,y: x+y, info_gains_per_modality) """
    
    pKLD_per_modality = jtu.tree_map(compute_pKLD_for_modality, qo, A, list(range(len(A))))
        
    return jtu.tree_reduce(lambda x,y: x+y, pKLD_per_modality)

def compute_predicted_free_energy(qs, qo, A, A_dependencies):
    

    def compute_pF_for_modality(qo_m, A_m, m):
        #H_qo = stable_entropy(qo_m)
        """ deps = A_dependencies[m]
        relevant_factors = jnp.array([qs[idx] for idx in deps])
        
        qs_ln_A_m = - stable_cross_entropy(relevant_factors,A_m)
        
        qo_qs_ln_A_m = -(qo_m * qs_ln_A_m).sum()
        #qo_qs_ln_A_m = factor_dot(H_A_m, relevant_factors) """
        deps = A_dependencies[m]
        relevant_factors = [qs[idx] for idx in deps]
        log_A_m = log_stable(A_m)
        qo_ln_A_m = -(jnp.expand_dims(qo_m, axis=tuple(range(1, log_A_m.ndim))) * log_A_m).sum(0)
        qo_qs_ln_A_m = factor_dot(qo_ln_A_m, relevant_factors)
        #qo_qs_ln_A_m = factor_dot(qs_ln_A_m, qo_m)
        #print(qo_qs_ln_A_m)
        return qo_qs_ln_A_m #- H_qo
    
    pF_per_modality = jtu.tree_map(compute_pF_for_modality, qo, A, list(range(len(A))))
        
    return jtu.tree_reduce(lambda x,y: x+y, pF_per_modality)

def compute_oRisk(t, qo, C):
    def compute_expected_entropy_for_modality(qo_m):
        H_qo = stable_entropy(qo_m)
        return H_qo
    oRisk = 0.
    for o_m, C_m in zip(qo, C):
        if C_m.ndim > 1:
            oRisk -= (o_m * C_m[t]).sum()
        else:
            oRisk -= (o_m * C_m).sum()
    Entropy_per_modality = jtu.tree_map(compute_expected_entropy_for_modality, qo)
    H_qo_all=jtu.tree_reduce(lambda x,y: x+y, Entropy_per_modality)#-Σqolnqo
    oRisk-=H_qo_all#Σqolnqo##-=
    return oRisk

def sample_policy_idx(policies, q_pi, action_selection="deterministic", alpha = 16.0, rng_key=None):

    if action_selection == "deterministic":
        policy_idx = jnp.argmax(q_pi)
    elif action_selection == "stochastic":
        log_p_policies = log_stable(q_pi) * alpha
        policy_idx = jr.categorical(rng_key, log_p_policies)

    selected_multiaction = policies[policy_idx, 0]
    return selected_multiaction, policy_idx

#for infer_policies_detail
def compute_G_policy_inductive_detail(qs_init, A, B, C, pA, pB, A_dependencies, B_dependencies, I, policy_i, alpha_vec=None, inductive_epsilon=1e-3, use_utility=True, use_states_info_gain=True, use_param_info_gain=False, use_inductive=False):
    """ 
    Write a version of compute_G_policy that does the same computations as `compute_G_policy` but using `lax.scan` instead of a for loop.
    This one further adds computations used for inductive planning.
    return detail and calc with alpha if alpha is not None
    """

    def scan_body(carry, t):

        qs, neg_G, G_info = carry

        qs_next = compute_expected_state(qs, B, policy_i[t], B_dependencies)

        qo = compute_expected_obs(qs_next, A, A_dependencies)

        info_gain = (compute_info_gain_alpha(qs_next, qo, A, A_dependencies, alpha_vec) if alpha_vec is not None else compute_info_gain(qs_next, qo, A, A_dependencies)) if use_states_info_gain else 0.

        utility = (compute_expected_utility_alpha(t, qo, C, alpha_vec) if alpha_vec is not None else compute_expected_utility(t, qo, C)) if use_utility else 0.

        inductive_value = calc_inductive_value_t(qs_init, qs_next, I, epsilon=inductive_epsilon) if use_inductive else 0.

        pA_info_gain = 0.
        pB_info_gain = 0.
        if pA is not None:
            pA_info_gain = calc_pA_info_gain(pA, qo, qs_next, A_dependencies) if use_param_info_gain else 0.
        if pB is not None:
            pB_info_gain = calc_pB_info_gain(pB, qs_next, qs, B_dependencies, policy_i[t]) if use_param_info_gain else 0.

        neg_G += info_gain + utility - (pA_info_gain + pB_info_gain) + inductive_value

        G_info = {"state_info_gain":info_gain, "utility":utility, "pA_info_gain":pA_info_gain, "pB_info_gain":pB_info_gain, "inductive_value":inductive_value}

        return (qs_next, neg_G, G_info), None

    qs = qs_init
    neg_G = 0.
    info ={"state_info_gain":0., "utility":0., "pA_info_gain":0., "pB_info_gain":0., "inductive_value":0.}
    final_state, _ = lax.scan(scan_body, (qs, neg_G, info), jnp.arange(policy_i.shape[0]))
    _, neg_G, info = final_state
    return neg_G, info

def update_posterior_policies_inductive_detail(policy_matrix, qs_init, A, B, C, E, pA, pB, A_dependencies, B_dependencies, I, alpha_vec = None, gamma=16.0, inductive_epsilon=1e-3, use_utility=True, use_states_info_gain=True, use_param_info_gain=False, use_inductive=True):
    # policy --> n_levels_factor_f x 1
    # factor --> n_levels_factor_f x n_policies
    ## vmap across policies
    compute_G_fixed_states = partial(compute_G_policy_inductive_detail, qs_init, A, B, C, pA, pB, A_dependencies, B_dependencies, I, alpha_vec=alpha_vec, inductive_epsilon=inductive_epsilon,
                                     use_utility=use_utility,  use_states_info_gain=use_states_info_gain, use_param_info_gain=use_param_info_gain, use_inductive=use_inductive)

    # only in the case of policy-dependent qs_inits
    # in_axes_list = (1,) * n_factors
    # all_efe_of_policies = vmap(compute_G_policy, in_axes=(in_axes_list, 0))(qs_init_pi, policy_matrix)

    # policies needs to be an NDarray of shape (n_policies, n_timepoints, n_control_factors)
    neg_efe_all_policies, info = vmap(compute_G_fixed_states)(policy_matrix)

    return nn.softmax(gamma * neg_efe_all_policies + log_stable(E)), neg_efe_all_policies, info

def update_posterior_policies_inductive_efe_qs_pi(policy_matrix, qs_init_pi, A, B, C, E, pA, pB, A_dependencies, B_dependencies, I, gamma=16.0, inductive_epsilon=1e-3, use_utility=True, use_states_info_gain=True, use_param_info_gain=False, use_inductive=True):
    # policy --> n_levels_factor_f x 1
    # factor --> n_levels_factor_f x n_policies
    """ print(qs_init_pi)
    print(policy_matrix) """
    ## vmap across policies
    compute_G_fixed_states = partial(compute_G_policy_inductive_efe_qs_pi, A, B, C, pA, pB, A_dependencies, B_dependencies, I, inductive_epsilon=inductive_epsilon,
                                     use_utility=use_utility,  use_states_info_gain=use_states_info_gain, use_param_info_gain=use_param_info_gain, use_inductive=use_inductive)

    # policies needs to be an NDarray of shape (n_policies, n_timepoints, n_control_factors)
    
    #vmapmatrix=jtu.tree_map(lambda x,y:(x,list(y)),qs_init_pi, list(policy_matrix))
    #results = vmap(compute_G_fixed_states)(vmapmatrix)
    #print(f"qs_init_pi: {qs_init_pi}")
    results = vmap(compute_G_fixed_states)(jnp.array(qs_init_pi),policy_matrix)
    
    neg_efe_all_policies = results[0]  # 各ポリシーの負の期待自由エネルギー
    PBS_a_p = results[1]  # 状態情報利得
    PKLD_a_p = results[2]  # 状態情報利得
    PFE_a_p = results[3]  # 状態情報利得
    oRisk_a_p = results[4]  # 状態情報利得
    PBS_pA_a_p = results[5]  # パラメータAに関する情報利得
    PBS_pB_a_p = results[6]  # パラメータBに関する情報利得
    # only in the case of policy-dependent qs_inits
    # in_axes_list = (1,) * n_factors
    # all_efe_of_policies = vmap(compute_G_policy, in_axes=(in_axes_list, 0))(qs_init_pi, policy_matrix)

    # policies needs to be an NDarray of shape (n_policies, n_timepoints, n_control_factors)
    #neg_efe_all_policies = vmap(compute_G_fixed_states)(policy_matrix)

    return nn.softmax(gamma * neg_efe_all_policies + log_stable(E)), neg_efe_all_policies, PBS_a_p, PKLD_a_p, PFE_a_p, oRisk_a_p, PBS_pA_a_p, PBS_pB_a_p

def compute_G_policy_inductive_efe_qs_pi( A, B, C, pA, pB, A_dependencies, B_dependencies, I, qs_init, policy_i, inductive_epsilon=1e-3, use_utility=True, use_states_info_gain=True, use_param_info_gain=False, use_inductive=False):
    """ 
    Write a version of compute_G_policy that does the same computations as `compute_G_policy` but using `lax.scan` instead of a for loop.
    This one further adds computations used for inductive planning.
    """
    def scan_body(carry, t):

        #qs, neg_G = carry
        #qsはループごとに更新するのではなくqs_initを参照
        neg_G, info_gain, predicted_KLD, predicted_F, oRisk, param_info_gainA, param_info_gainB,inductive_value = carry#qs, neg_G, info_gain, predicted_KLD, predicted_F, oRisk, param_info_gainA, param_info_gainB,inductive_value = carry
        qs = jtu.tree_map(lambda x: x[:,t,:],qs_init)
        qs =list(qs)
        #qs = [x[:, t] if x.ndim == 2 else x[:, t, :] for x in qs_init]
        qs_next = jtu.tree_map(lambda x: x[:,t+1,:],qs_init)#qs_initはポリシーの長さ分の未来の状態に対する信念，agent.pyで処理する場合と異なりbatchsizeの次元がない？
        qs_next =list(qs_next)
        #print(qs_next)
        #qs_next = compute_expected_state(qs, B, policy_i[t], B_dependencies)

        qo = compute_expected_obs(qs_next, A, A_dependencies)

        info_gain += compute_info_gain(qs_next, qo, A, A_dependencies) if use_states_info_gain else 0.

        predicted_KLD += compute_predicted_KLD(qs_next, qo, A, A_dependencies) 
        #print("PFE")
        predicted_F += compute_predicted_free_energy(qs_next, qo, A, A_dependencies) 
        #print("Risk")
        oRisk += compute_oRisk(t, qo, C)
        #utility = compute_expected_utility(t, qo, C) if use_utility else 0.

        inductive_value += calc_inductive_value_t(qs_init, qs_next, I, epsilon=inductive_epsilon) if use_inductive else 0.

        
        
        if pA is not None:
            param_info_gainA -= calc_pA_info_gain(pA, qo, qs_next, A_dependencies) if use_param_info_gain else 0.
        else:
            param_info_gainA = 0.
        if pB is not None:
            param_info_gainB -= calc_pB_info_gain(pB, qs_next, qs, B_dependencies, policy_i[t]) if use_param_info_gain else 0.
        else:
            param_info_gainB = 0.

        #neg_G = info_gain - param_info_gainA - param_info_gainB + inductive_value

        neg_G = info_gain + predicted_KLD - predicted_F - oRisk + param_info_gainA + param_info_gainB + inductive_value
        #neg_G += inductive_value + utility 
        return (neg_G, info_gain, predicted_KLD, predicted_F, oRisk, param_info_gainA, param_info_gainB,inductive_value), None#(jnp.array(qs_next), neg_G, info_gain, predicted_KLD, predicted_F, oRisk, param_info_gainA, param_info_gainB,inductive_value), None

    #qs = qs_init
    #print(qs)
    #print(policy_i)
    neg_G = 0.
    info_gain = 0.
    predicted_KLD = 0.
    predicted_F = 0.
    oRisk = 0.
    param_info_gainA = 0.
    param_info_gainB = 0.
    inductive_value = 0.
    final_state, _ = lax.scan(scan_body, (neg_G, info_gain, predicted_KLD, predicted_F, oRisk, param_info_gainA, param_info_gainB,inductive_value), jnp.arange(policy_i.shape[0]))
    #final_state, _ = lax.scan(scan_body, (qs, neg_G, info_gain, predicted_KLD, predicted_F, oRisk, param_info_gainA, param_info_gainB,inductive_value), jnp.arange(policy_i.shape[0]))
    neg_G, info_gain, predicted_KLD, predicted_F, oRisk, param_info_gainA, param_info_gainB, inductive_value = final_state
    return neg_G, info_gain, predicted_KLD, predicted_F, oRisk, param_info_gainA, param_info_gainB

def compute_expected_state_obs(qs_prior, A, B, u_t, A_dependencies, B_dependencies=None): 
    """
    Compute posterior over next state, given belief about previous state, transition model and action...
    """
    #Note: this algorithm is only correct if each factor depends only on itself. For any interactions, 
    # we will have empirical priors with codependent factors. 
    assert len(u_t) == len(B)  
    qs_next = []
    for B_f, u_f, deps in zip(B, u_t, B_dependencies):
        relevant_factors = [qs_prior[idx] for idx in deps]
        qs_next_f = factor_dot(B_f[...,u_f], relevant_factors, keep_dims=(0,))
        qs_next.append(qs_next_f)
        
    # P(s'|s, u) = \sum_{s, u} P(s'|s) P(s|u) P(u|pi)P(pi) because u </-> pi
        
    def compute_expected_obs_modality(A_m, m):
        deps = A_dependencies[m]
        relevant_factors = [qs_next[idx] for idx in deps]
        return factor_dot(A_m, relevant_factors, keep_dims=(0,))

    return qs_next,jtu.tree_map(compute_expected_obs_modality, A, list(range(len(A))))

def update_posterior_policies_inductive_efe_qs_pi2(policy_matrix, qs_init_pi, A, B, C, E, pA, pB, A_dependencies, B_dependencies, I, gamma=16.0, inductive_epsilon=1e-3, use_utility=True, use_states_info_gain=True, use_param_info_gain=False, use_inductive=True):
    # policy --> n_levels_factor_f x 1
    # factor --> n_levels_factor_f x n_policies
    """ print(qs_init_pi)
    print(policy_matrix) """
    ## vmap across policies
    compute_G_fixed_states = partial(compute_G_policy_inductive_efe_qs_pi2, A, B, C, pA, pB, A_dependencies, B_dependencies, I, qs_init_pi,inductive_epsilon=inductive_epsilon,
                                     use_utility=use_utility,  use_states_info_gain=use_states_info_gain, use_param_info_gain=use_param_info_gain, use_inductive=use_inductive)

    # policies needs to be an NDarray of shape (n_policies, n_timepoints, n_control_factors)
    #print(f"qs_init_pi: {qs_init_pi}")#ポリシー，因子，状態
    #print(f"policy_matrix: {policy_matrix}")#ポリシー，時点？，因子
    #vmapmatrix=jtu.tree_map(lambda x,y:(x,list(y)),qs_init_pi, list(policy_matrix))
    #results = vmap(compute_G_fixed_states)(vmapmatrix)
    ##results = vmap(compute_G_fixed_states)(jnp.array(qs_init_pi),policy_matrix)
    #results = vmap(compute_G_fixed_states)(qs_init_pi,policy_matrix, in_axes=(0,0))
    policy_indices = jnp.arange(policy_matrix.shape[0])
    results = vmap(lambda p, i: compute_G_fixed_states(p, policy_idx=i))(policy_matrix, policy_indices)
    #results = vmap(compute_G_fixed_states)(policy_matrix)
    neg_efe_all_policies = results[0]  # 各ポリシーの負の期待自由エネルギー
    PBS_a_p = results[1]  # 状態情報利得
    PKLD_a_p = results[2]  # 状態情報利得
    PFE_a_p = results[3]  # 状態情報利得
    oRisk_a_p = results[4]  # 状態情報利得
    PBS_pA_a_p = results[5]  # パラメータAに関する情報利得
    PBS_pB_a_p = results[6]  # パラメータBに関する情報利得
    # only in the case of policy-dependent qs_inits
    # in_axes_list = (1,) * n_factors
    # all_efe_of_policies = vmap(compute_G_policy, in_axes=(in_axes_list, 0))(qs_init_pi, policy_matrix)

    # policies needs to be an NDarray of shape (n_policies, n_timepoints, n_control_factors)
    #neg_efe_all_policies = vmap(compute_G_fixed_states)(policy_matrix)

    return nn.softmax(gamma * neg_efe_all_policies + log_stable(E)), neg_efe_all_policies, PBS_a_p, PKLD_a_p, PFE_a_p, oRisk_a_p, PBS_pA_a_p, PBS_pB_a_p

def compute_G_policy_inductive_efe_qs_pi2( A, B, C, pA, pB, A_dependencies, B_dependencies, I, qs_init, policy_i,policy_idx=None, inductive_epsilon=1e-3, use_utility=True, use_states_info_gain=True, use_param_info_gain=False, use_inductive=False):
    """ 
    Write a version of compute_G_policy that does the same computations as `compute_G_policy` but using `lax.scan` instead of a for loop.
    This one further adds computations used for inductive planning.
    """

    def scan_body(carry, t):

        #qs, neg_G = carry
        qs, neg_G, info_gain, predicted_KLD, predicted_F, oRisk, param_info_gainA, param_info_gainB,inductive_value = carry
        
        
        qs_next = compute_expected_state(qs, B, policy_i[t], B_dependencies)

        qo = compute_expected_obs(qs_next, A, A_dependencies)

        info_gain += compute_info_gain(qs_next, qo, A, A_dependencies) if use_states_info_gain else 0.

        predicted_KLD += compute_predicted_KLD(qs_next, qo, A, A_dependencies) 
        #print("PFE")
        predicted_F += compute_predicted_free_energy(qs_next, qo, A, A_dependencies) 
        #print("Risk")
        oRisk += compute_oRisk(t, qo, C)
        #utility = compute_expected_utility(t, qo, C) if use_utility else 0.

        inductive_value += calc_inductive_value_t(qs_init, qs_next, I, epsilon=inductive_epsilon) if use_inductive else 0.
        #print(qs_next)
        
        
        if pA is not None:
            param_info_gainA -= calc_pA_info_gain(pA, qo, qs_next, A_dependencies) if use_param_info_gain else 0.
        else:
            param_info_gainA = 0.
        if pB is not None:
            param_info_gainB -= calc_pB_info_gain(pB, qs_next, qs, B_dependencies, policy_i[t]) if use_param_info_gain else 0.
        else:
            param_info_gainB = 0.

        #neg_G = info_gain - param_info_gainA - param_info_gainB + inductive_value

        neg_G = info_gain + predicted_KLD - predicted_F - oRisk + param_info_gainA + param_info_gainB + inductive_value
        #neg_G += inductive_value + utility 
        return (qs_next, neg_G, info_gain, predicted_KLD, predicted_F, oRisk, param_info_gainA, param_info_gainB,inductive_value), None
    #print(f"qs:",qs_init)#ポリシー，因子
        # 各因子のbeliefsから現在のポリシーのbeliefsを取得
    def get_qs_pi(i):
        return qs_init[i]

    qs_pi = lax.cond(
        policy_idx == 0,
        lambda: get_qs_pi(0),
        lambda: lax.cond(
            policy_idx == 1,
            lambda: get_qs_pi(1),
            lambda: get_qs_pi(2)
        )
    )
    #qs_pi = lax.switch(policy_idx, qs)
        # qs_pi = lax.index_in_dim(qs, policy_idx, axis=0)
    #qs_pi = jnp.take(qs, policy_idx, axis=0)
    #print(f"qs_pi:",qs_pi)
    qs = qs_pi
    #print(qs)
    #print(policy_i)
    neg_G = 0.
    info_gain = 0.
    predicted_KLD = 0.
    predicted_F = 0.
    oRisk = 0.
    param_info_gainA = 0.
    param_info_gainB = 0.
    inductive_value = 0.
    final_state, _ = lax.scan(scan_body, (qs, neg_G, info_gain, predicted_KLD, predicted_F, oRisk, param_info_gainA, param_info_gainB,inductive_value), jnp.arange(policy_i.shape[0]))
    _, neg_G, info_gain, predicted_KLD, predicted_F, oRisk, param_info_gainA, param_info_gainB, inductive_value = final_state
    return neg_G, info_gain, predicted_KLD, predicted_F, oRisk, param_info_gainA, param_info_gainB

def compute_info_gain_alpha(qs, qo, A, A_dependencies, alpha):
    """
    compute info gain with alpha (as a coefficient of each modalities)
    """
    assert len(qo) == len(alpha), f"alpha(len:{len(alpha)}) should match with qo(len:{len(qo)})"

    def compute_info_gain_for_modality(qo_m, A_m, m):
        H_qo = stable_entropy(qo_m)
        H_A_m = - stable_xlogx(A_m).sum(0)
        deps = A_dependencies[m]
        relevant_factors = [qs[idx] for idx in deps]
        qs_H_A_m = factor_dot(H_A_m, relevant_factors)
        return H_qo - qs_H_A_m
    
    info_gains_per_modality = jtu.tree_map(compute_info_gain_for_modality, qo, A, list(range(len(A))))
    weighted_info_gains_per_modality = jtu.tree_map(lambda x,a: x*a, info_gains_per_modality, list(alpha))

    return jtu.tree_reduce(lambda x,y: x+y, weighted_info_gains_per_modality)

def compute_expected_utility_alpha(t, qo, C, alpha):
    """
    compute utility with alpha (as a coefficient of each modalities)
    """
    assert len(C) == len(alpha), f"alpha(len:{len(alpha)}) should match with qo(len:{len(qo)})"
    
    util = 0.
    for o_m, C_m, alpha_m in zip(qo, C, alpha):
        if C_m.ndim > 1: # when C depend on time
            util += alpha_m * (o_m * C_m[t]).sum()
        else:
            util += alpha_m * (o_m * C_m).sum()
    
    return util

def calc_pB_o_mutual_info_gain(pB, qs_t, qs_t_minus_1, B_dependencies, u_t_minus_1, qo, A, A_dependencies,rng_key=None):
    """
    Compute expected Dirichlet information gain about parameters ``pB`` under a given policy

    Parameters
    ----------
    pB: ``Array`` of dtype object
        Dirichlet parameters over transition model (same shape as ``B``)
    qs_t: ``list`` of ``Array`` of dtype object
        Predictive posterior beliefs over hidden states expected under the policy at time ``t``
    qs_t_minus_1: ``list`` of ``Array`` of dtype object
        Posterior over hidden states at time ``t-1`` (before receiving observations)
    u_t_minus_1: "Array"
        Actions in time step t-1 for each factor

    Returns
    -------
    infogain_pB: float
        Surprise (about Dirichlet parameters) expected under the policy in question
    """
    from typing import List, Sequence, Tuple, Union
    from jax import random, tree_util as jtu
    Array = jnp.ndarray

    def _sample_B_factor_all_actions(
        key,
        alpha_f: Array,
        eps: float = 1e-12
    ) :
        """
        1因子分のディリクレ濃度 alpha_f から、全 (s, a) 列について B をサンプル
        alpha_f: (S_next, S_curr, A)  あるいは (S_next, S_curr) にも対応
        返り値: (B_f, key_out)  B_f は alpha_f と同じ shape
        """
        alpha_f = jnp.asarray(alpha_f, dtype=jnp.float32)

        # 行動非依存 (S_next, S_curr) の場合も扱えるよう分岐
        if alpha_f.ndim == 2:
            S_next, S_curr = alpha_f.shape
            # 列（現状態 s）ごとに dirichlet
            keys = random.split(key, S_curr + 1)
            subkeys, key_out = keys[:-1], keys[-1]  # (S_curr, 2)
            # (S_curr, S_next) を vmapped でサンプル → 転置して (S_next, S_curr)
            cols = jnp.transpose(jnp.clip(alpha_f, eps, jnp.inf), (1, 0))  # (S_curr, S_next)
            samples = vmap(lambda k, a: random.dirichlet(k, a))(subkeys, cols)  # (S_curr, S_next)
            B_f = jnp.transpose(samples, (1, 0))
            return B_f, key_out

        elif alpha_f.ndim == 3:
            S_next, S_curr, A = alpha_f.shape
            alpha = jnp.clip(alpha_f, eps, jnp.inf)  # (S_next, S_curr, A)

            # 全 (s, a) 列の本数 = S_curr * A ぶんの key を用意
            keys = random.split(key, S_curr * A + 1)
            subkeys, key_out = keys[:-1], keys[-1]  # (S_curr*A, 2)

            # 列方向に並べ替え： (S_curr, A, S_next) → (S_curr*A, S_next)
            cols = jnp.transpose(alpha, (1, 2, 0)).reshape(S_curr * A, S_next)

            # 列ごとに Dirichlet サンプル → (S_curr*A, S_next)
            samples = vmap(lambda k, a: random.dirichlet(k, a))(subkeys, cols)

            # 形を戻す： (S_curr*A, S_next) → (S_curr, A, S_next) → 転置で (S_next, S_curr, A)
            samples_Snext_Scurr_A = jnp.transpose(samples.reshape(S_curr, A, S_next), (2, 0, 1))
            return samples_Snext_Scurr_A, key_out

        else:
            raise ValueError("alpha_f の次元は 2 か 3 を想定しています。")

    def sample_B_from_pB_tree(
        key,
        pB: Sequence[Array],
        eps: float = 1e-12
    ) :
        """
        pB: 因子ごとのディリクレ濃度配列のリスト
            各配列の shape は (S_next, S_curr, A)（行動依存）または (S_next, S_curr)（行動非依存）
        返り値: (B_list, key_out)  … B_list は pB と同じ構造（各因子ごとに B を返す）
        """
        # pB の因子数ぶん key を用意
        F = len(pB)
        keys = random.split(key, F + 1)
        key_factors, key_out = keys[:-1], keys[-1]

        # keys を pB と同じ PyTree 構造に（ここでは単純なリスト）
        # 因子ごとにサンプルを実行
        def _per_factor(alpha_f, kf):
            B_f, _ = _sample_B_factor_all_actions(kf, alpha_f, eps=eps)
            return B_f

        B_list = jtu.tree_map(_per_factor, list(pB), list(key_factors))
        # tree_map は同構造を返す（ここでは List[Array]）
        return B_list, key_out

    def compute_entropy_for_modality(qo_m):
        H_qo = stable_entropy(qo_m)#Calculate predictied entropyの計算
        #H_A_m = - stable_xlogx(A_m).sum(0)#観測モデル（A,）p(o|s)のエントロピーを計算し，o方向に和を取る．;Calculate the entropy of the observation model (A, p(o|s)) and sum in the direction of o.
        #deps = A_dependencies[m]
        #relevant_factors = [qs[idx] for idx in deps]
        #qs_H_A_m = factor_dot(H_A_m, relevant_factors)#Ambiguityの計算．q(s|π)とH_A_mの内積．;Calculation of ambiguity. Inner product of q(s|π) and H_A_m.
        return H_qo 

    def compute_expected_state_for_mc(qs_prior, B, u_t, B_dependencies):
        """
        qs_prior : list[Array]                   # 各因子の q_t(s_f)
        B        : list[Array]                   # 各因子の B_f (S_next, S_curr, A_f) or (S_next, S_curr)
        u_t      : Array or list[int]            # 各因子の行動インデックス（JAX配列でも可）
        B_dependencies : list[list[int] or tuple[int]]
            各因子 f について、B_f が依存する hidden state 因子インデックス（静的：Python リスト/タプル）
            例：self-transition だけなら [[f] for f in range(num_factors)]
        """
        assert len(B) == len(u_t) == len(B_dependencies)
        qs_next = []

        for B_f, u_f, deps in zip(B, u_t, B_dependencies):
            # 1) 行動依存なら動的インデクシングで a=u_f の断面を取得
            if B_f.ndim == 3:
                # (S_next, S_curr, A_f) → (S_next, S_curr)
                B_sel = jnp.take(B_f, u_f, axis=-1)
            else:
                # (S_next, S_curr)（行動非依存）
                B_sel = B_f

            # 2) 依存因子の事前を収集（deps は Python の静的なインデックス列であること）
            relevant_factors = [qs_prior[idx] for idx in deps]

            # 3) P(s'_f | B, π) = factor_dot( B_sel, ⨂_{d∈deps} q(s_d) )
            #    factor_dot は (S_next, S_curr_{deps}) × ⨂ q → (S_next,) を返す想定
            qs_next_f = factor_dot(B_sel, relevant_factors, keep_dims=(0,))

            qs_next.append(qs_next_f)

        return qs_next  # list[Array]（各因子の q_{t+1}(s'_f)）
    def compute_expected_obs_for_mc(qs, A, A_dependencies):
        """
        New version of expected observation (computation of Q(o|pi)) that takes into account sparse dependencies between observation
        modalities and hidden state factors
        """
            
        def compute_expected_obs_modality(A_m, m):
            deps = A_dependencies[m]
            relevant_factors = [qs[idx] for idx in deps]
            return factor_dot(A_m, relevant_factors, keep_dims=(0,))

        return jtu.tree_map(compute_expected_obs_modality, A, list(range(len(A))))

    from typing import Tuple

    def _single_H_qo_given_B_pi(
        key,
        pB,
        qs_t_minus_1,
        u_t_minus_1,
        B_dependencies,
        A,
        A_dependencies,
        compute_expected_state_for_mc,
        compute_expected_obs_for_mc,
        compute_entropy_for_modality,
    ):
        B_list, _ = sample_B_from_pB_tree(key, pB)

        # ★ ここで静的インデックス（Python int / tuple）を使う
        qs_next = compute_expected_state_for_mc(qs_t_minus_1, B_list, u_t_minus_1, B_dependencies)
        qo_temp = compute_expected_obs_for_mc(qs_next, A, A_dependencies)

        H_qo_B_per_mod = jtu.tree_map(compute_entropy_for_modality, qo_temp)
        H_qo_B = jtu.tree_reduce(lambda x, y: x + y, H_qo_B_per_mod)
        return H_qo_B

    def mc_E_H_qo_given_B_pi_vmap(
        rng_key,
        pB,
        qs_t_minus_1,
        u_t_minus_1,              
        B_dependencies,
        A,
        A_dependencies,
        compute_expected_state_for_mc,
        compute_expected_obs_for_mc,
        compute_entropy_for_modality,
        nsamples: int = 5000,
    ):
        keys = random.split(rng_key, nsamples + 1)
        key_batch, next_key = keys[:-1], keys[-1]

        vmapped = vmap(
            lambda k: _single_H_qo_given_B_pi(
                k, pB, qs_t_minus_1, u_t_minus_1, B_dependencies,
                A, A_dependencies, compute_expected_state_for_mc, compute_expected_obs_for_mc,
                compute_entropy_for_modality
            ),
            in_axes=0, out_axes=0
        )
        H_batch = vmapped(key_batch)  # (nsamples,)
        # ★ ここを Python 変換しない（float にしない）
        mean_H = jnp.mean(H_batch)
        n = jnp.asarray(nsamples, dtype=H_batch.dtype)
        se_H   = jnp.std(H_batch, ddof=1) / jnp.sqrt(n)
        return mean_H, se_H, next_key

    if rng_key is None:
        rng_key = random.PRNGKey(0)  # ※ 実運用は外から渡すのが推奨
    # ★ 追加：依存を “静的な tuple” に固定（hash 可能にする）
    Bdeps_static = tuple(tuple(int(i) for i in deps) for deps in B_dependencies)
    Adeps_static = tuple(tuple(int(i) for i in deps) for deps in A_dependencies)
    # 1) モンテカルロ推定；MCで E_B[H(o|B,π)] を推定
    # もともと: u_t_minus_1 = [0, 2, ...], B_dependencies = [[0,2], [1], ...] など
    #u_static  = tuple(int(u) for u in u_t_minus_1)
    #Bdeps_static = tuple(tuple(int(i) for i in deps) for deps in B_dependencies)

    mc_fn_jit = jit(
        mc_E_H_qo_given_B_pi_vmap,
        static_argnames=('B_dependencies', 'A_dependencies',
                        'compute_expected_state_for_mc', 'compute_expected_obs_for_mc',
                        'compute_entropy_for_modality', 'nsamples')
    )

    mean_H_v, se_H_v, rng_key = mc_fn_jit(
        rng_key,
        pB,
        qs_t_minus_1,
        u_t_minus_1,        
        Bdeps_static,# ← 静的 tuple
        A,
        Adeps_static,# ← 静的 tuple
        compute_expected_state_for_mc,
        compute_expected_obs_for_mc,
        compute_entropy_for_modality,
        nsamples=8192,
    )
    
    # 2) 解析的に H(o|π) を計算（既存のやり方のまま）
    Hqo_per_modality = jtu.tree_map(compute_entropy_for_modality, qo)    
    Hqo = jtu.tree_reduce(lambda x,y: x+y, Hqo_per_modality)
    #Hqo = jnp.asarray(Hqo)
    # 3) 相互情報量の推定値とSE
    I_pi_est = Hqo - mean_H_v        # JAX scalar
    I_pi_se  = se_H_v                # JAX scalar

    return I_pi_est, I_pi_se
