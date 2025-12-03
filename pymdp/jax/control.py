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

def update_posterior_policies_inductive_efe(policy_matrix, qs_init, A, B, C, E, pA, pB, A_dependencies, B_dependencies, I, gamma=16.0, inductive_epsilon=1e-3, use_utility=True, use_states_info_gain=True, use_param_info_gain=False, use_inductive=True,rng_key=None):
    # policy --> n_levels_factor_f x 1
    # factor --> n_levels_factor_f x n_policies
    ## vmap across policies
    compute_G_fixed_states = partial(compute_G_policy_inductive_efe, qs_init, A, B, C, pA, pB, A_dependencies, B_dependencies, I, inductive_epsilon=inductive_epsilon,
                                     use_utility=use_utility,  use_states_info_gain=use_states_info_gain, use_param_info_gain=use_param_info_gain, use_inductive=use_inductive,rng_key=rng_key)

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

def compute_G_policy_inductive_efe(qs_init, A, B, C, pA, pB, A_dependencies, B_dependencies, I, policy_i, inductive_epsilon=1e-3, use_utility=True, use_states_info_gain=True, use_param_info_gain=False, use_inductive=False,rng_key=None):
    """ 
    Write a version of compute_G_policy that does the same computations as `compute_G_policy` but using `lax.scan` instead of a for loop.
    This one further adds computations used for inductive planning.
    """

    def scan_body(carry, t):

        #qs, neg_G = carry
        qs, neg_G, info_gain, predicted_KLD, predicted_F, oRisk, param_info_gainA, param_info_gainB,utility, inductive_value, I_B_o,I_B_o_se = carry

        qs_next = compute_expected_state(qs, B, policy_i[t], B_dependencies) #q(s|π)=p(sτ+1|sτ,π)stの計算.stは認識分布;Calculation of q(s|π)=p(sτ+1|sτ,π)st. st is the recognition distribution.

        qo = compute_expected_obs(qs_next, A, A_dependencies) #Calculate q(o|π)=p(o|s)p(sτ+1|sτ,π)st

        #info_gain +=compute_info_gain_st(qs_next, qo, A, qs, B, policy_i[t], A_dependencies, B_dependencies)

        info_gain += compute_info_gain(qs_next, qo, A, A_dependencies) if use_states_info_gain else 0.#Calculate pBS(epistemic value) #compute_predicted_KLD(qs_next, qo, A, A_dependencies)

        predicted_KLD += compute_predicted_KLD(qs_next, qo, A, A_dependencies) #Calculate pKLD
        #print("PFE")
        predicted_F += compute_predicted_free_energy(qs_next, qo, A, A_dependencies) #Calculate predicted free energy
        #print("Risk")
        oRisk += compute_oRisk(t, qo, C)#Calculate Risk
        utility += compute_expected_utility(t, qo, C) if use_utility else 0.#Calculate utility(Pragmatic value)

        inductive_value += calc_inductive_value_t(qs_init, qs_next, I, epsilon=inductive_epsilon) if use_inductive else 0.
        
        if pA is not None:
            param_info_gainA -= calc_pA_info_gain(pA, qo, qs_next, A_dependencies) if use_param_info_gain else 0.
        else:
            param_info_gainA = 0.
        if pB is not None:
            param_info_gainB -= calc_pB_info_gain(pB, qs_next, qs, B_dependencies, policy_i[t]) if use_param_info_gain else 0.
            val1, val2 = calc_pB_o_mutual_info_gain(pB, qs_next, qs_init, B_dependencies, policy_i[t], qo, A, A_dependencies,rng_key=rng_key) if use_param_info_gain else (0., 0.)
            I_B_o += val1
            I_B_o_se += val2
        else:
            param_info_gainB = 0.
            I_B_o=0.
            I_B_o_se =0.

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

    """ def _sample_B_factor_all_actions(
        key,
        alpha_f: Array,
        eps: float = 1e-12
    ) :
        
        #1因子分のディリクレ濃度 alpha_f から、全 (s, a) 列について B をサンプル
        #alpha_f: (S_next, S_curr, A)  あるいは (S_next, S_curr) にも対応
        #返り値: (B_f, key_out)  B_f は alpha_f と同じ shape
        
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
            raise ValueError("alpha_f の次元は 2 か 3 を想定しています。") """
    

    def _sample_B_factor_all_actions(
        key,
        alpha_f,
        eps: float = 1e-12,
    ):
        """
        1因子分のディリクレ濃度 alpha_f から、全「列」について B をサンプルする汎用版

        想定:
        - alpha_f の 0 番目の軸が Dirichlet の次元（"次状態" 次元）S_next
        - それ以外の軸は全部「独立なディリクレ分布の集合」とみなす

        例:
        alpha_f.shape = (S_next, S_curr)           → 現在の 2D ケース
        alpha_f.shape = (S_next, S_curr, A)        → 現在の 3D ケース
        alpha_f.shape = (S_next, F1, F2, ..., Fk)  → 一般化されたケース

        戻り値:
        B_f : alpha_f と同じ shape、各 "列" ごとに Dirichlet サンプル済み
        key_out : 消費後の乱数キー
        """
        alpha_f = jnp.asarray(alpha_f, dtype=jnp.float32)

        if alpha_f.ndim == 1:
            # 単一の Dirichlet ベクトル (K,) の場合
            alpha = jnp.clip(alpha_f, eps, jnp.inf)
            B_vec = random.dirichlet(key, alpha)
            return B_vec, key

        if alpha_f.ndim < 2:
            raise ValueError("alpha_f は少なくとも 2 次元（Dirichlet次元 + 何か）が必要です。")

        # 0 番目軸が Dirichlet の次元
        S_next = alpha_f.shape[0]
        rest_shape = alpha_f.shape[1:]           # それ以外の軸
        n_cols = 1
        for d in rest_shape:
            n_cols *= d                          # Python int（shape 情報なので jit 的にOK）

        # (S_next, rest...) → (S_next, n_cols) にまとめる
        alpha = jnp.clip(alpha_f, eps, jnp.inf)
        alpha_2d = alpha.reshape(S_next, n_cols)         # (S_next, n_cols)

        # 列方向に並べ替え → (n_cols, S_next)
        cols = jnp.transpose(alpha_2d, (1, 0))          # (n_cols, S_next)

        # 列ごとに Dirichlet サンプル
        keys = random.split(key, n_cols + 1)
        subkeys, key_out = keys[:-1], keys[-1]          # (n_cols, 2)
        samples = vmap(lambda k, a: random.dirichlet(k, a))(subkeys, cols)  # (n_cols, S_next)

        # 元の shape に戻す:
        # (n_cols, S_next) → (S_next, n_cols) → (S_next, *rest_shape)
        samples_2d = jnp.transpose(samples, (1, 0))     # (S_next, n_cols)
        B_f = samples_2d.reshape((S_next, *rest_shape)) # alpha_f と同じ shape

        return B_f, key_out


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
    
    #高速版？
    """ mc_fn_jit = jit(
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
    ) """
    ##メモリ節約版
    from functools import partial
    from typing import NamedTuple
    from jax import jit, vmap, lax, random
    #import jax.numpy as jnp

    class _AggState(NamedTuple):
        n: jnp.ndarray     # 累積サンプル数
        mean: jnp.ndarray  # 累積平均
        M2: jnp.ndarray    # 分散合算用の M2 (= (n-1)*var の和)

    def mc_E_H_qo_given_B_pi_chunked(
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
        nsamples: int = 8192,
        chunk_size: int = 512,        # ★ ここで並列度を制御（メモリに合わせて調整）
        remat_inner: bool = False,    # ★ True で中間を再計算（さらに省メモリ、やや遅い）
    ):
        """
        - nsamples を chunk_size ずつに割って vmap で並列評価
        - 各チャンクの (mean, M2) を Welford 合算
        - keys はチャンクごとに fold_in → split で少量生成
        """

        # inner: 1サンプルぶんの H を計算
        def one_H_given_B(key):
            B_list, _ = sample_B_from_pB_tree(key, pB)
            qs_next = compute_expected_state_for_mc(qs_t_minus_1, B_list, u_t_minus_1, B_dependencies)
            qo_temp = compute_expected_obs_for_mc(qs_next, A, A_dependencies)
            H_qo_B_per_mod = jtu.tree_map(compute_entropy_for_modality, qo_temp)
            return jtu.tree_reduce(lambda x, y: x + y, H_qo_B_per_mod)  # スカラー

        if remat_inner:
            from jax import checkpoint
            one_H_given_B = checkpoint(one_H_given_B)  # 中間保存を減らし更に省メモリ

        # vmap で chunk_size 個まとめて評価
        vmapped_H = vmap(one_H_given_B, in_axes=0, out_axes=0)  # keys -> (chunk_size,)

        # チャンク数（最後の端数チャンクも固定サイズで回してマスク）
        num_chunks = (nsamples + chunk_size - 1) // chunk_size

        def combine(state: _AggState, batch_mean, batch_M2, batch_n):
            # Welford の2群合算
            n1, m1, M2_1 = state.n, state.mean, state.M2
            n2, m2, M2_2 = batch_n, batch_mean, batch_M2

            n_tot = n1 + n2
            # n2==0（全マスク）でも数値安定に更新
            w = jnp.where(n_tot > 0, n2 / jnp.maximum(n_tot, 1.0), 0.0)
            delta = m2 - m1
            m_tot = m1 + w * delta
            M2_tot = M2_1 + M2_2 + jnp.where(
                (n1 > 0) & (n2 > 0),
                (delta * delta) * (n1 * n2 / jnp.maximum(n_tot, 1.0)),
                0.0
            )
            return _AggState(n_tot, m_tot, M2_tot)

        def body_fun(i, state: _AggState):
            # このチャンクのキーを少量だけ生成
            base = random.fold_in(rng_key, i)
            keys = random.split(base, chunk_size)

            # チャンク内で一括並列評価（B などの中間はチャンク内だけで生存）
            H_batch = vmapped_H(keys)  # (chunk_size,)

            # 末尾チャンク用のマスク（余剰を捨てる）
            start = i * chunk_size
            remain = nsamples - start
            m = jnp.clip(remain, 0, chunk_size)
            mask = (jnp.arange(chunk_size) < m).astype(H_batch.dtype)

            # マスク付きでチャンク統計を計算（2パスでも chunk_size が小さいので軽い）
            batch_n = jnp.sum(mask)
            # すべてマスクならスキップ
            def nonempty():
                s1 = jnp.sum(H_batch * mask)
                mean = s1 / jnp.maximum(batch_n, 1.0)
                M2 = jnp.sum(((H_batch - mean) ** 2) * mask)  # 分散合算用M2
                return mean, M2, batch_n
            def empty():
                z = jnp.array(0.0, dtype=H_batch.dtype)
                return z, z, jnp.array(0.0, dtype=H_batch.dtype)

            batch_mean, batch_M2, batch_n = lax.cond(batch_n > 0, nonempty, empty)
            return combine(state, batch_mean, batch_M2, batch_n)

        init = _AggState(
            n=jnp.array(0.0, dtype=jnp.float32),
            mean=jnp.array(0.0, dtype=jnp.float32),
            M2=jnp.array(0.0, dtype=jnp.float32),
        )

        final = lax.fori_loop(0, num_chunks, body_fun, init)

        mean_H = final.mean
        se_H = jnp.where(final.n > 1.0,
                        jnp.sqrt((final.M2 / (final.n - 1.0)) / final.n),
                        jnp.array(jnp.nan, dtype=final.mean.dtype))
        return mean_H, se_H, rng_key

    mc_fn_jit = jit(
        mc_E_H_qo_given_B_pi_chunked,
        static_argnames=('B_dependencies','A_dependencies',
                        'compute_expected_state_for_mc','compute_expected_obs_for_mc',
                        'compute_entropy_for_modality','nsamples','chunk_size','remat_inner'),
        donate_argnums=(0,)  # rng_key を寄付 → バッファ再利用を促進
    )

    mean_H_v, se_H_v, rng_key = mc_fn_jit(
        rng_key,
        pB,
        qs_t_minus_1,
        u_t_minus_1,
        Bdeps_static,
        A,
        Adeps_static,
        compute_expected_state_for_mc,
        compute_expected_obs_for_mc,
        compute_entropy_for_modality,
        nsamples=8192,       # ← 総サンプル数（従来同等）
        chunk_size=512,      # ← メモリに合わせて 256〜2048 で調整
        remat_inner=False,   # ← まだメモリ厳しければ True（やや遅くなる）
    )
    
    # 2) 解析的に H(o|π) を計算（既存のやり方のまま）
    Hqo_per_modality = jtu.tree_map(compute_entropy_for_modality, qo)    
    Hqo = jtu.tree_reduce(lambda x,y: x+y, Hqo_per_modality)
    #Hqo = jnp.asarray(Hqo)
    # 3) 相互情報量の推定値とSE
    I_pi_est = Hqo - mean_H_v        # JAX scalar
    I_pi_se  = se_H_v                # JAX scalar

    return I_pi_est, I_pi_se

def compute_info_gain_st(qs, qo, A, qs_prior, B, u_t, A_dependencies, B_dependencies=None):

    #assert len(u_t) == len(B)  
    # qs_next = []
    # for B_f, u_f, deps in zip(B, u_t, B_dependencies):
    #     relevant_factors = [qs_prior[idx] for idx in deps]
        
    #     qs_next_f = factor_dot(B_f[...,u_f], relevant_factors, keep_dims=(0,))
    #     qs_next.append(qs_next_f)
    assert len(u_t) == len(B)  
    B_dot_1 = []
    for B_f, u_f, deps_B in zip(B, u_t, B_dependencies):
        #relevant_factors = [qs_prior[idx] for idx in deps]
        ##relevant_factors = [jnp.ones_like(qs_prior[idx]) for idx in deps_B]#B_dependenciesにもとづき，認識分布と同じ形状の全要素1のリストを作成
        B_dot_1_f=B_f[...,u_f]
        ##B_dot_1_f = factor_dot(B_f[...,u_f], relevant_factors, keep_dims=tuple(range(B_f[...,u_f].ndim)))#全要素1のベクトルをかけて，該当する行動utのB行列をそのまま取り出す
        B_dot_1.append(B_dot_1_f)
        
    # P(s'|s, u) = \sum_{s, u} P(s'|s) P(s|u) P(u|pi)P(pi) because u </-> pi
    
    def build_dims_and_keep(A_m, relevant_factors, contract_axes):
        assert len(relevant_factors) == len(contract_axes)
        # 新規ラベルは A_m の既存ラベル(A_m.ndim) 以降を使う
        next_label = A_m.ndim

        dims = []
        keep = [0]  # A_m の軸0（観測軸）を残す

        for ax, x in zip(contract_axes, relevant_factors):
            # x の軸0は A_m の軸 ax と縮約
            labels = [ax]
            # x の軸1以降は出力に残すので新しいラベルを割り当て、keep にも追加
            for _ in range(1, x.ndim):
                labels.append(next_label)
                keep.append(next_label)
                next_label += 1
            dims.append(tuple(labels))
        dims = tuple(dims)
        keep_dims = tuple(keep)
        return dims, keep_dims

    def compute_info_gain_for_modality(qo_m, A_m, m):
        H_qo = stable_entropy(qo_m)#Calculate predictied entropyの計算
        
        deps_A = A_dependencies[m]
        relevant_factors = [B_dot_1[idx] for idx in deps_A]
        #contract_axes = [d + 1 for d in deps_A]  # A は通常 (o, s1, s2, ...) なので状態は +1 シフト
        contract_axes = [1 + i for i in range(len(relevant_factors))]
        dims, keep_dims = build_dims_and_keep(A_m, relevant_factors, contract_axes)
        #relevant_factors = [B[idx] for idx in deps_A]
        #A_B = factor_dot(A_m,relevant_factors)
        for ax, x in zip(contract_axes, relevant_factors):
            ax_size = A_m.shape[ax]
            x0_size = x.shape[0]
            assert ax_size == x0_size, (
                f"Contract size mismatch: A_m axis {ax} has {ax_size}, "
                f"but factor.shape[0] is {x0_size}. "
                f"A_m.shape={A_m.shape}, factor.shape={x.shape}"
            )
        
        A_B = factor_dot_flex(
            A_m,
            xs=relevant_factors,
            dims=dims,
            keep_dims=keep_dims
        )
        deps = A_dependencies[m]
        relevant_factors_st = [qs_prior[idx] for idx in deps_A] #st
        H_AB = - stable_xlogx(A_B).sum(0) #H[A*B]
        st_H_AB = factor_dot(H_AB, relevant_factors_st)
        # H_A_m = - stable_xlogx(A_m).sum(0)#観測モデル（A,）p(o|s)のエントロピーを計算し，o方向に和を取る．;Calculate the entropy of the observation model (A, p(o|s)) and sum in the direction of o.
        # deps = A_dependencies[m]
        # relevant_factors = [qs[idx] for idx in deps]
        # qs_H_A_m = factor_dot(H_A_m, relevant_factors)#Ambiguityの計算．q(s|π)とH_A_mの内積．;Calculation of ambiguity. Inner product of q(s|π) and H_A_m.
        return H_qo - st_H_AB
    
    info_gains_per_modality = jtu.tree_map(compute_info_gain_for_modality, qo, A, list(range(len(A))))#,B_dot_1
        
    return jtu.tree_reduce(lambda x,y: x+y, info_gains_per_modality)

def update_posterior_policies_inductive_efe_curiosity(policy_matrix, qs_init, A, B, C, E, pA, pB, A_dependencies, B_dependencies, I, gamma=16.0, inductive_epsilon=1e-3, use_utility=True, use_states_info_gain=True, use_param_info_gain=False, use_inductive=True,rng_key=None):
    # policy --> n_levels_factor_f x 1
    # factor --> n_levels_factor_f x n_policies
    ## vmap across policies
    compute_G_fixed_states = partial(compute_G_policy_inductive_efe_curiosity, qs_init, A, B, C, pA, pB, A_dependencies, B_dependencies, I, inductive_epsilon=inductive_epsilon,
                                     use_utility=use_utility,  use_states_info_gain=use_states_info_gain, use_param_info_gain=use_param_info_gain, use_inductive=use_inductive,rng_key=rng_key)

    # policies needs to be an NDarray of shape (n_policies, n_timepoints, n_control_factors)
    results = vmap(compute_G_fixed_states)(policy_matrix)
    
    
    neg_efe_all_policies = results[0]  # 各ポリシーの負の期待自由エネルギー;Negative expected free energy of each policy
    PBS_a_p = results[1]  # 状態情報利得;information gain for states
    PBS_st_a_p = results[2]
    PKLD_a_p = results[3]  # 状態情報利得
    PFE_a_p = results[4]  # 
    oRisk_a_p = results[5]  # 
    PBS_pA_a_p = results[6]  # パラメータAに関する情報利得;information gain for pA(parameter of A)
    PBS_pB_a_p = results[7]  # パラメータBに関する情報利得;information gain for pB(parameter of B)
    I_B_o_a_p = results[8]  # パラメータBに関する情報利得;information gain for pB(parameter of B)
    I_B_o_se_a_p = results[9]  # パラメータBに関する情報利得;information gain for pB(parameter of B)
    #print(PBS_a_p)
    # only in the case of policy-dependent qs_inits
    # in_axes_list = (1,) * n_factors
    # all_efe_of_policies = vmap(compute_G_policy, in_axes=(in_axes_list, 0))(qs_init_pi, policy_matrix)

    # policies needs to be an NDarray of shape (n_policies, n_timepoints, n_control_factors)
    #neg_efe_all_policies = vmap(compute_G_fixed_states)(policy_matrix)
    #⇓ポリシーの分布の計算q(π)=softmax(-γG+E)
    return nn.softmax(gamma * neg_efe_all_policies + log_stable(E)), neg_efe_all_policies, PBS_a_p, PBS_st_a_p, PKLD_a_p, PFE_a_p, oRisk_a_p, PBS_pA_a_p, PBS_pB_a_p,I_B_o_a_p,I_B_o_se_a_p

def compute_G_policy_inductive_efe_curiosity(qs_init, A, B, C, pA, pB, A_dependencies, B_dependencies, I, policy_i, inductive_epsilon=1e-3, use_utility=True, use_states_info_gain=True, use_param_info_gain=False, use_inductive=False,rng_key=None):
    """ 
    Write a version of compute_G_policy that does the same computations as `compute_G_policy` but using `lax.scan` instead of a for loop.
    This one further adds computations used for inductive planning.
    """

    def scan_body(carry, t):

        #qs, neg_G = carry
        qs, neg_G, info_gain,info_gain_st, predicted_KLD, predicted_F, oRisk, param_info_gainA, param_info_gainB,utility, inductive_value, I_B_o,I_B_o_se = carry

        qs_next = compute_expected_state(qs, B, policy_i[t], B_dependencies) #q(s|π)=p(sτ+1|sτ,π)stの計算.stは認識分布;Calculation of q(s|π)=p(sτ+1|sτ,π)st. st is the recognition distribution.

        qo = compute_expected_obs(qs_next, A, A_dependencies) #Calculate q(o|π)=p(o|s)p(sτ+1|sτ,π)st

        info_gain_st +=compute_info_gain_st(qs_next, qo, A, qs, B, policy_i[t], A_dependencies, B_dependencies)

        info_gain += compute_info_gain(qs_next, qo, A, A_dependencies) if use_states_info_gain else 0.#Calculate pBS(epistemic value) #compute_predicted_KLD(qs_next, qo, A, A_dependencies)

        predicted_KLD += compute_predicted_KLD(qs_next, qo, A, A_dependencies) #Calculate pKLD
        #print("PFE")
        predicted_F += compute_predicted_free_energy(qs_next, qo, A, A_dependencies) #Calculate predicted free energy
        #print("Risk")
        oRisk += compute_oRisk(t, qo, C)#Calculate Risk
        utility += compute_expected_utility(t, qo, C) if use_utility else 0.#Calculate utility(Pragmatic value)

        inductive_value += calc_inductive_value_t(qs_init, qs_next, I, epsilon=inductive_epsilon) if use_inductive else 0.
        
        if pA is not None:
            param_info_gainA -= calc_pA_info_gain(pA, qo, qs_next, A_dependencies) if use_param_info_gain else 0.
        else:
            param_info_gainA = 0.
        if pB is not None:
            param_info_gainB -= calc_pB_info_gain(pB, qs_next, qs, B_dependencies, policy_i[t]) if use_param_info_gain else 0.
            val1, val2 = calc_pB_o_mutual_info_gain(pB, qs_next, qs_init, B_dependencies, policy_i[t], qo, A, A_dependencies,rng_key=rng_key) if use_param_info_gain else (0., 0.)
            I_B_o += val1
            I_B_o_se += val2
        else:
            param_info_gainB = 0.
            I_B_o=0.
            I_B_o_se =0.

        #neg_G = info_gain + param_info_gainA + param_info_gainB + inductive_value + utility

        neg_G = info_gain + info_gain_st+predicted_KLD - predicted_F - oRisk + param_info_gainA + param_info_gainB + inductive_value
        #neg_G += inductive_value 
        return (qs_next, neg_G, info_gain,info_gain_st, predicted_KLD, predicted_F, oRisk, param_info_gainA, param_info_gainB, utility, inductive_value, I_B_o,I_B_o_se), None

    qs = qs_init
    #print(qs)
    #print(policy_i)
    neg_G = 0.
    info_gain = 0.
    info_gain_st= 0.
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
    final_state, _ = lax.scan(scan_body, (qs, neg_G, info_gain,info_gain_st, predicted_KLD, predicted_F, oRisk, param_info_gainA, param_info_gainB, utility, inductive_value, I_B_o,I_B_o_se), jnp.arange(policy_i.shape[0]))
    _, neg_G, info_gain, info_gain_st,predicted_KLD, predicted_F, oRisk, param_info_gainA, param_info_gainB, utility, inductive_value, I_B_o,I_B_o_se = final_state
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
    return neg_G, info_gain, info_gain_st,predicted_KLD, predicted_F, oRisk, param_info_gainA, param_info_gainB, I_B_o, I_B_o_se