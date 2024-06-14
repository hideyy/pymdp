#!/usr/bin/env python
# -*- coding: utf-8 -*-
# pylint: disable=no-member

import jax.numpy as jnp
from .algos import run_factorized_fpi, run_mmp, run_vmp, run_mmp_vfe, run_mmp_err
from jax import tree_util as jtu
from jax.scipy.special import xlogy

def update_posterior_states(
        A, 
        B, 
        obs, 
        past_actions, 
        prior=None, 
        qs_hist=None, 
        A_dependencies=None, 
        B_dependencies=None, 
        num_iter=16, 
        method='fpi'
    ):

    if method == 'fpi' or method == "ovf":
        # format obs to select only last observation
        curr_obs = jtu.tree_map(lambda x: x[-1], obs)
        qs = run_factorized_fpi(A, curr_obs, prior, A_dependencies, num_iter=num_iter)
    else:
        # format B matrices using action sequences here
        # TODO: past_actions can be None
        if past_actions is not None:
            nf = len(B)
            actions_tree = [past_actions[:, i] for i in range(nf)]
            
            # move time steps to the leading axis (leftmost)
            # this assumes that a policy is always specified as the rightmost axis of Bs
            B = jtu.tree_map(lambda b, a_idx: jnp.moveaxis(b[..., a_idx], -1, 0), B, actions_tree)
        else:
            B = None

        # outputs of both VMP and MMP should be a list of hidden state factors, where each qs[f].shape = (T, batch_dim, num_states_f)
        if method == 'vmp':
            qs = run_vmp(A, B, obs, prior, A_dependencies, B_dependencies, num_iter=num_iter) 
        if method == 'mmp':
            qs = run_mmp(A, B, obs, prior, A_dependencies, B_dependencies, num_iter=num_iter)
    
    if qs_hist is not None:
        if method == 'fpi' or method == "ovf":
            qs_hist = jtu.tree_map(lambda x, y: jnp.concatenate([x, jnp.expand_dims(y, 0)], 0), qs_hist, qs)
        else:
            #TODO: return entire history of beliefs
            qs_hist = qs
    else:
        if method == 'fpi' or method == "ovf":
            qs_hist = jtu.tree_map(lambda x: jnp.expand_dims(x, 0), qs)
        else:
            qs_hist = qs
    
    return qs_hist

def update_posterior_states_err(
        A, 
        B, 
        obs, 
        past_actions, 
        prior=None, 
        qs_hist=None, 
        A_dependencies=None, 
        B_dependencies=None, 
        num_iter=16, 
        method='fpi'
    ):

    if method == 'fpi' or method == "ovf":
        # format obs to select only last observation
        curr_obs = jtu.tree_map(lambda x: x[-1], obs)
        qs = run_factorized_fpi(A, curr_obs, prior, A_dependencies, num_iter=num_iter)
    else:
        # format B matrices using action sequences here
        # TODO: past_actions can be None
        if past_actions is not None:
            nf = len(B)
            actions_tree = [past_actions[:, i] for i in range(nf)]
            
            # move time steps to the leading axis (leftmost)
            # this assumes that a policy is always specified as the rightmost axis of Bs
            B = jtu.tree_map(lambda b, a_idx: jnp.moveaxis(b[..., a_idx], -1, 0), B, actions_tree)
        else:
            B = None

        # outputs of both VMP and MMP should be a list of hidden state factors, where each qs[f].shape = (T, batch_dim, num_states_f)
        if method == 'vmp':
            qs = run_vmp(A, B, obs, prior, A_dependencies, B_dependencies, num_iter=num_iter) 
        if method == 'mmp':
            qs,err = run_mmp_err(A, B, obs, prior, A_dependencies, B_dependencies, num_iter=num_iter)
    
    if qs_hist is not None:
        if method == 'fpi' or method == "ovf":
            qs_hist = jtu.tree_map(lambda x, y: jnp.concatenate([x, jnp.expand_dims(y, 0)], 0), qs_hist, qs)
        else:
            #TODO: return entire history of beliefs
            qs_hist = qs
    else:
        if method == 'fpi' or method == "ovf":
            qs_hist = jtu.tree_map(lambda x: jnp.expand_dims(x, 0), qs)
        else:
            qs_hist = qs
    
    return qs_hist, err

def update_posterior_states_vfe(
        A, 
        B, 
        obs, 
        past_actions, 
        prior=None, 
        qs_hist=None, 
        A_dependencies=None, 
        B_dependencies=None, 
        num_iter=16, 
        method='fpi'
    ):

    if method == 'fpi' or method == "ovf":
        # format obs to select only last observation
        curr_obs = jtu.tree_map(lambda x: x[-1], obs)
        qs = run_factorized_fpi(A, curr_obs, prior, A_dependencies, num_iter=num_iter)
    else:
        # format B matrices using action sequences here
        # TODO: past_actions can be None
        if past_actions is not None:
            nf = len(B)
            actions_tree = [past_actions[:, i] for i in range(nf)]
            
            # move time steps to the leading axis (leftmost)
            # this assumes that a policy is always specified as the rightmost axis of Bs
            B = jtu.tree_map(lambda b, a_idx: jnp.moveaxis(b[..., a_idx], -1, 0), B, actions_tree)
        else:
            B = None

        # outputs of both VMP and MMP should be a list of hidden state factors, where each qs[f].shape = (T, batch_dim, num_states_f)
        if method == 'vmp':
            qs = run_vmp(A, B, obs, prior, A_dependencies, B_dependencies, num_iter=num_iter) 
        if method == 'mmp':
            qs, err, vfe, kld, bs, un = run_mmp_vfe(A, B, obs, prior, A_dependencies, B_dependencies, num_iter=num_iter)
    
    if qs_hist is not None:
        if method == 'fpi' or method == "ovf":
            qs_hist = jtu.tree_map(lambda x, y: jnp.concatenate([x, jnp.expand_dims(y, 0)], 0), qs_hist, qs)
        else:
            #TODO: return entire history of beliefs
            qs_hist = qs
    else:
        if method == 'fpi' or method == "ovf":
            qs_hist = jtu.tree_map(lambda x: jnp.expand_dims(x, 0), qs)
        else:
            qs_hist = qs
    
    return qs_hist, err, vfe, kld, bs, un

def calc_KLD(past_beliefs,current_qs):
    
    def compute_KLD_for_factor(past_beliefs_f, current_qs_f, f):
        H_past_beliefs = xlogy(past_beliefs_f,past_beliefs_f).sum()
        #H_past_beliefs = xlogy(current_qs_f,current_qs_f).sum()
        past_beliefs_lncurrent_qs = xlogy(past_beliefs_f, current_qs_f).sum()
        
        return H_past_beliefs-past_beliefs_lncurrent_qs
    #
    kld_for_factor = jtu.tree_map(compute_KLD_for_factor, past_beliefs, current_qs, list(range(len(past_beliefs)))) #- past_beliefs_lncurrent_qs
    return jtu.tree_reduce(lambda x,y: x+y, kld_for_factor)
