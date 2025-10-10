#!/usr/bin/env python
# -*- coding: utf-8 -*-
# pylint: disable=no-member

import jax.numpy as jnp
from .algos import run_factorized_fpi, run_mmp, run_vmp, run_mmp_vfe, run_mmp_vfe_policies, run_mmp2, run_mmp_vfe_set_prior
from jax import tree_util as jtu, lax ,vmap
from jax.experimental.sparse._base import JAXSparse
from jax.experimental import sparse
from jaxtyping import Array, ArrayLike

from jax.scipy.special import xlogy

from functools import partial

eps = jnp.finfo('float').eps

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

def joint_dist_factor(b: ArrayLike, filtered_qs: list[Array], actions: Array):
    qs_last = filtered_qs[-1]
    qs_filter = filtered_qs[:-1]

    def step_fn(qs_smooth, xs):
        qs_f, action = xs
        time_b = b[..., action]
        qs_j = time_b * qs_f
        norm = qs_j.sum(-1, keepdims=True)
        if isinstance(norm, JAXSparse):
            norm = sparse.todense(norm)
        norm = jnp.where(norm == 0, eps, norm)
        qs_backward_cond = qs_j / norm
        qs_joint = qs_backward_cond * jnp.expand_dims(qs_smooth, -1)
        qs_smooth = qs_joint.sum(-2)
        if isinstance(qs_smooth, JAXSparse):
            qs_smooth = sparse.todense(qs_smooth)
        
        # returns q(s_t), (q(s_t), q(s_t, s_t+1))
        return qs_smooth, (qs_smooth, qs_joint)

    # seq_qs will contain a sequence of smoothed marginals and joints
    _, seq_qs = lax.scan(
        step_fn,
        qs_last,
        (qs_filter, actions),
        reverse=True,
        unroll=2
    )

    # we add the last filtered belief to smoothed beliefs

    qs_smooth_all = jnp.concatenate([seq_qs[0], jnp.expand_dims(qs_last, 0)], 0)
    qs_joint_all = seq_qs[1]
    if isinstance(qs_joint_all, JAXSparse):
        qs_joint_all.shape = (len(actions),) + qs_joint_all.shape
    return qs_smooth_all, qs_joint_all


def smoothing_ovf(filtered_post, B, past_actions):
    assert len(filtered_post) == len(B)
    nf = len(B)  # number of factors

    joint = lambda b, qs, f: joint_dist_factor(b, qs, past_actions[..., f])

    marginals_and_joints = ([], [])
    for b, qs, f in zip(B, filtered_post, list(range(nf))):
        marginals, joints = joint(b, qs, f)
        marginals_and_joints[0].append(marginals)
        marginals_and_joints[1].append(joints)

    return marginals_and_joints

def update_posterior_states_vfe(
        A, 
        B, 
        obs, 
        past_actions, 
        prior=None, 
        qs_hist=None, 
        A_dependencies=None, 
        B_dependencies=None, 
        num_iter=32, #num_iter=16, 
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
            actions_tree = [past_actions[:, i] for i in range(nf)] #過去とった行動のリストを作成;Make a list of actions you have taken in the past.
            
            # move time steps to the leading axis (leftmost)
            # this assumes that a policy is always specified as the rightmost axis of Bs
            B = jtu.tree_map(lambda b, a_idx: jnp.moveaxis(b[..., a_idx], -1, 0), B, actions_tree) #過去とった行動に対応するB行列のリストを作成．;Create a list of B matrices corresponding to past actions.
        else:
            B = None

        # outputs of both VMP and MMP should be a list of hidden state factors, where each qs[f].shape = (T, batch_dim, num_states_f)
        if method == 'vmp':
            qs = run_vmp(A, B, obs, prior, A_dependencies, B_dependencies, num_iter=num_iter) 
        if method == 'mmp':
            #MMPにもとづき認識分布（qs）やvfeの計算;Calculation of recognition distribution (qs) and vfe based on MMP
            qs, err, vfe, kld2, bs, un = run_mmp_vfe(A, B, obs, prior, A_dependencies, B_dependencies, num_iter=num_iter)#qs, err, vfe, kld, bs, un
    
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
    
    return qs_hist, err, vfe, kld2, bs, un#qs_hist, err, vfe, kld, bs, un

def calc_KLD(past_beliefs,current_qs):
    
    def compute_KLD_for_factor(past_beliefs_f, current_qs_f, f):
        #print(past_beliefs_f.shape)
        #print(current_qs_f.shape)
        H_past_beliefs = xlogy(past_beliefs_f,past_beliefs_f).sum(-1)
        #H_past_beliefs = xlogy(current_qs_f,current_qs_f).sum()
        past_beliefs_lncurrent_qs = xlogy(past_beliefs_f, current_qs_f).sum(-1)
        
        return H_past_beliefs-past_beliefs_lncurrent_qs
    #
    kld_for_factor = jtu.tree_map(compute_KLD_for_factor, past_beliefs, current_qs, list(range(len(past_beliefs)))) #- past_beliefs_lncurrent_qs
    return kld_for_factor#jtu.tree_reduce(lambda x,y: x+y, kld_for_factor)

def update_posterior_states_vfe_policies(
        A, 
        B, 
        obs, 
        policies,
        past_actions, 
        prior=None, 
        qs_hist=None, 
        A_dependencies=None, 
        B_dependencies=None, 
        num_iter=16, 
        method='fpi'
    ):

    """     def find_matching_index(realaction_tree, policies_tree):
        realaction_tree = jtu.tree_map(lambda x: jnp.broadcast_to(x, (1,) + x.shape), realaction_tree)
        print(realaction_tree[0][0])
        print(policies_tree[0][0])
        for i in range(len(policies_tree)):
        if jnp.array_equal(realaction_tree[0][0], policies_tree[i][0]):
                return i
            if jnp.all(jnp.equal(realaction_tree[0][0], policies_tree[i][0])):
                return i
        return None  # 一致するインデックスが見つからなかった場合 """

    if method == 'fpi' or method == "ovf":
        # format obs to select only last observation
        curr_obs = jtu.tree_map(lambda x: x[-1], obs)
        qs = run_factorized_fpi(A, curr_obs, prior, A_dependencies, num_iter=num_iter)
    else:
        # format B matrices using action sequences here
        # TODO: past_actions can be None
        """ if past_actions is not None:
            nf = len(B)
            actions_tree = [past_actions[:, i] for i in range(nf)]
            
            # move time steps to the leading axis (leftmost)
            # this assumes that a policy is always specified as the rightmost axis of Bs
            B = jtu.tree_map(lambda b, a_idx: jnp.moveaxis(b[..., a_idx], -1, 0), B, actions_tree)
        else:
            B = None """

        if past_actions is not None and policies is not None:
            K, t, _= policies.shape
            #if past_actions.shape[0]>=t:
                #print(K)
            #print(t)
            nf = len(B)
            #print(nf)#1,factor_number
            #print('actions_tree')
            actions_tree = [past_actions[:, i] for i in range(nf)]
            #print('actions_tree2')
            #actions_tree = jtu.tree_map(lambda x: [x[0:-t, :] for _ in range(K)], actions_tree)
            #print(actions_tree)
            #actions_tree = jtu.tree_map(lambda x: [x[0:-t] for _ in range(K)], actions_tree)
            #realaction_tree=jtu.tree_map(lambda x: x[-t:] , actions_tree)##

            #actions_tree = jtu.tree_map(lambda x: x , actions_tree)##
            #print(actions_tree)
            
            #print(len(actions_tree))
            #print(actions_tree[0].shape)
            #print(policies[0].shape)
            #print('actions_tree3')
            #actions_tree = jtu.tree_map(lambda x: jnp.concatenate([actions_tree, x], axis=0),  policies)
            
            actions_tree = jnp.asarray(actions_tree) # Convert to ndarray##

            #print(actions_tree[0].shape)
            policies_tree=[jnp.asarray([policies[k][:, i] for i in range(nf)]) for k in range(K)]
            #print(policies_tree[0][0].shape)
            actions_tree = [jnp.concatenate([actions_tree, policies_tree[k]], axis=1) for k in range(K)]
            #print(actions_tree)
            #print(realaction_tree[0])
            #print(policies_tree[0][0])
            #print(policies[0])
            #print(policies_tree[2])
            #matching_indices = find_matching_index(realaction_tree, policies_tree)

            
            #actions_tree = [jnp.concatenate([actions_tree, policies[k][:][i]], axis=0) for k in range(K)]##
            
            #actions_tree = jtu.tree_map(lambda x: x[0:-t].reshape(-1), actions_tree)
            #actions_tree = jnp.asarray(actions_tree).flatten()
            #actions_tree = [jnp.concatenate([actions_tree, policies[k]], axis=0) for k in range(K)]
            #print(len(actions_tree))
            #print(actions_tree[0].shape)
            #actions_tree = [jtu.tree_map(lambda x: jnp.concatenate([x[0:-t], policies[k]], axis=0), actions_tree) for k in range(K)]
            #actions_tree = jtu.tree_map(lambda x: [jnp.concatenate([x[i], policies[i]], axis=0) for i in range(K)], actions_tree)
            #print('actions_tree_complete')
            """ t = policies.shape[0]
            actions_tree = jtu.tree_map(lambda x: x[0:-t, :], actions_tree)
            actions_tree = jtu.tree_map(lambda x, y: jnp.concatenate([x, y], axis=0), actions_tree, policies) """
            
            # move time steps to the leading axis (leftmost)
            # this assumes that a policy is always specified as the rightmost axis of Bs
            #B = jtu.tree_map(lambda b, a_idx: jnp.moveaxis(b[..., a_idx], -1, 0), B, actions_tree)
            #B = jtu.tree_map(lambda b: [jnp.moveaxis(b[..., a_idx], -1, 0) for a_idx in actions_tree], B)
            #B = [jtu.tree_map(lambda b, a_idx: jnp.moveaxis(b[..., a_idx], -1, 0), B, actions_tree[k]) for k in range(K)]

            B_list = []
            for k in range(K):
                actions_tree_k = list(actions_tree[k])
                #print(actions_tree_k[0].shape)
                #print(f"actions_tree_k: {actions_tree_k}")
                B_k = jtu.tree_map(lambda b, a_idx: jnp.moveaxis(b[..., a_idx], -1, 0), B, actions_tree_k)
                #print('B_k_complete')
                B_list.append(B_k)
            B = B_list
            B_list = []
            """ for k in range(K):
                # Flatten each sub-array and reshape to match dimensions with policies[k]
                actions_tree = jtu.tree_map(lambda x: x[0:-t].reshape(-1), actions_tree)
                actions_tree = jnp.asarray(actions_tree).reshape(-1, 1, 1)  # Adjusted to match expected dimensions
                
                # Reshape policies[k] to match the dimensions of actions_tree
                policies_k_reshaped = policies[k].reshape(policies[k].shape[0], policies[k].shape[1], 1)
                
                # Concatenate along the first axis
                actions_tree_k = jnp.concatenate([actions_tree, policies_k_reshaped], axis=0)
                print(len(actions_tree_k))
                print(actions_tree_k[0].shape)
                
                print(f"actions_tree_k: {actions_tree_k}")
                B_k = jtu.tree_map(lambda b, a_idx: jnp.moveaxis(b[..., a_idx], -1, 0), B, actions_tree_k)
                print('B_k_complete')
                B_list.append(B_k)
            B = B_list """
                #print('Bdefine_complete')
            """ else:
                nf = len(B)
                actions_tree = [past_actions[:, i] for i in range(nf)]
                #print(len(actions_tree))
                #print(actions_tree[0].shape)
                # move time steps to the leading axis (leftmost)
                # this assumes that a policy is always specified as the rightmost axis of Bs
                #print(f"actions_tree_k: {actions_tree}")
                B = jtu.tree_map(lambda b, a_idx: jnp.moveaxis(b[..., a_idx], -1, 0), B, actions_tree)
                #B = jtu.tree_map(lambda b: [jnp.moveaxis(b[..., a_idx], -1, 0) for a_idx in actions_tree], B) """
        elif past_actions is None and policies is not None:
            K, t, _= policies.shape
            #if past_actions.shape[0]>=t:
                #print(K)
            #print(t)
            nf = len(B)
            policies_tree=[jnp.asarray([policies[k][:, i] for i in range(nf)]) for k in range(K)]
            B_list = []
            for k in range(K):
                actions_tree_k = list(policies_tree[k])
                #print(actions_tree_k[0].shape)
                #print(f"actions_tree_k: {actions_tree_k}")
                B_k = jtu.tree_map(lambda b, a_idx: jnp.moveaxis(b[..., a_idx], -1, 0), B, actions_tree_k)
                #print('B_k_complete')
                B_list.append(B_k)
            B = B_list
        else:
            B = None

        # outputs of both VMP and MMP should be a list of hidden state factors, where each qs[f].shape = (T, batch_dim, num_states_f)
        if method == 'vmp':
            qs = run_vmp(A, B, obs, prior, A_dependencies, B_dependencies, num_iter=num_iter) 
        
        def run_mmp_vfe_single(b):
            
            b=list(b)
            #print(b)
            return run_mmp_vfe_policies(A, b, obs, prior, A_dependencies, B_dependencies, num_iter=num_iter, policy_len=t)
        if method == 'mmp':
            # Use vmap to parallelize the function over the batch dimension
            if B is not None:
            #qs, err, vfe, kld, bs, un = vmap(run_mmp_vfe_single)(jnp.array(B))
                #if past_actions.shape[0]>=t:
                    #print('run_mmp_vfe_single')
                """ run_mmp_policies=partial(
                    run_mmp_vfe_policies,
                    A,
                    obs, 
                    prior, 
                    A_dependencies, 
                    B_dependencies, 
                    num_iter=num_iter, 
                    policy_len=t
                ) """
                """ run_mmp_policies = partial(
                    run_mmp_vfe_policies,
                    A=A,
                    obs=obs,
                    prior=prior,
                    A_dependencies=A_dependencies,
                    B_dependencies=B_dependencies,
                    num_iter=num_iter,
                    policy_len=t
                ) """


                #results=vmap(run_mmp_policies)(B)

                """ run_mmp_vfe_partial = partial(run_mmp_vfe_single)
                B=jnp.array(B)
                # ベクトル化
                results = vmap(run_mmp_vfe_partial)(B) """
                B=jnp.array(B)
                #print(B.shape)
                """ results = vmap(run_mmp_vfe_single,(0,))(B)
                print(results)
                qs, err, vfe, kld, bs, un = zip(*results) """
                qs, err, vfe, kld, bs, un = vmap(run_mmp_vfe_single,(0,))(B)
                #results = [run_mmp_vfe_single(b) for b in B]
                qs=jnp.array(qs)
                #print(qs.shape)
                qs=jnp.moveaxis(qs, 0, 1)
                #print(qs.shape)
                #realqs=qs[selected_policy[0]]
                #print(len(vfe))
                #print(vfe[0])
                #print(vfe[0].shape)
                #print(qs[0][0].shape)
                
                vfe=jtu.tree_map(lambda y: y.sum(2),vfe)
                kld=jtu.tree_map(lambda y: y.sum(2),kld)
                bs=jtu.tree_map(lambda y: y.sum(2),bs)
                un=jtu.tree_map(lambda y: y.sum(2),un)
                

                qs = list(qs)
                err = list(err)
                
                    #realqs=qs
            else:
                
                qs, err, vfe, kld, bs, un = run_mmp_vfe(A, B, obs, prior, A_dependencies, B_dependencies, num_iter=num_iter)
                vfe=jtu.tree_map(lambda y: y.sum(1),vfe)
                kld=jtu.tree_map(lambda y: y.sum(1),kld)
                bs=jtu.tree_map(lambda y: y.sum(1),bs)
                un=jtu.tree_map(lambda y: y.sum(1),un)
                #realqs=qs
    if qs_hist is not None:
        if method == 'fpi' or method == "ovf":
            qs_hist = jtu.tree_map(lambda x, y: jnp.concatenate([x, jnp.expand_dims(y, 0)], 0), qs_hist, qs)
        else:
            #TODO: return entire history of beliefs
            #qs_hist = realqs
            qs_hist=qs
    else:
        if method == 'fpi' or method == "ovf":
            qs_hist = jtu.tree_map(lambda x: jnp.expand_dims(x, 0), qs)
        else:
            #qs_hist = realqs
            qs_hist=qs
    vfe = jnp.array(vfe)
    kld = jnp.array(kld)
    bs = jnp.array(bs)
    un = jnp.array(un) 
    
    return qs_hist, err, vfe, kld, bs, un

def update_posterior_states_vfe_policies2(
        A, 
        B, 
        obs, 
        policies,
        past_actions, 
        prior=None, 
        qs_hist=None, 
        A_dependencies=None, 
        B_dependencies=None, 
        num_iter=16, 
        method='fpi'
    ):

    """     def find_matching_index(realaction_tree, policies_tree):
        realaction_tree = jtu.tree_map(lambda x: jnp.broadcast_to(x, (1,) + x.shape), realaction_tree)
        print(realaction_tree[0][0])
        print(policies_tree[0][0])
        for i in range(len(policies_tree)):
        if jnp.array_equal(realaction_tree[0][0], policies_tree[i][0]):
                return i
            if jnp.all(jnp.equal(realaction_tree[0][0], policies_tree[i][0])):
                return i
        return None  # 一致するインデックスが見つからなかった場合 """

    if method == 'fpi' or method == "ovf":
        # format obs to select only last observation
        curr_obs = jtu.tree_map(lambda x: x[-1], obs)
        qs = run_factorized_fpi(A, curr_obs, prior, A_dependencies, num_iter=num_iter)
    else:
        # format B matrices using action sequences here
        # TODO: past_actions can be None
        """ if past_actions is not None:
            nf = len(B)
            actions_tree = [past_actions[:, i] for i in range(nf)]
            
            # move time steps to the leading axis (leftmost)
            # this assumes that a policy is always specified as the rightmost axis of Bs
            B = jtu.tree_map(lambda b, a_idx: jnp.moveaxis(b[..., a_idx], -1, 0), B, actions_tree)
        else:
            B = None """

        if past_actions is not None and policies is not None:
            K, t, _= policies.shape
            if past_actions.shape[0]>=t:
                #print(K)
                #print(t)
                nf = len(B)
                #print(nf)#1,factor_number
                #print('actions_tree')
                actions_tree = [past_actions[:, i] for i in range(nf)]
                ##print(actions_tree[0].shape)
                
                #realaction_tree=jtu.tree_map(lambda x: x[-t:] , actions_tree)##
                actions_tree = jtu.tree_map(lambda x: x[0:-t] , actions_tree)##
                ##print(actions_tree[0].shape)
                #print(actions_tree)
                #print(len(actions_tree))
                #print(actions_tree[0].shape)
                #print(policies[0].shape)
                #print('actions_tree3')
                
                actions_tree = jnp.asarray(actions_tree) # Convert to ndarray##
                #print(actions_tree[0].shape)
                """ print(f"K",K)
                print(f"policies",policies)
                print(f"policies[0]",policies[0])
                print(f"policies[1]",policies[1])
                print(f"policies[2]",policies[2]) """

                policies_tree=[jnp.asarray([policies[k][:, i] for i in range(nf)]) for k in range(K)]
                #print(policies_tree[0].shape)

                actions_tree = [jnp.concatenate([actions_tree, policies_tree[k]], axis=1) for k in range(K)]
                ##print(actions_tree[0].shape)
                #print(actions_tree)
                #print(policies_tree[0][0])
                #print(policies[0])
                #print(policies_tree[2])
                #matching_indices = find_matching_index(realaction_tree, policies_tree)

                

                B_list = []
                for k in range(K):
                    actions_tree_k = list(actions_tree[k])
                    #print(actions_tree_k[0].shape)
                    #print(f"actions_tree_k: {actions_tree_k}")
                    B_k = jtu.tree_map(lambda b, a_idx: jnp.moveaxis(b[..., a_idx], -1, 0), B, actions_tree_k)
                    #print('B_k_complete')
                    B_list.append(B_k)
                B = B_list

                B_list = []
            
                 
            else:
                nf = len(B)
                actions_tree = [past_actions[:, i] for i in range(nf)]
                #print(len(actions_tree))
                #print(actions_tree[0].shape)
                # move time steps to the leading axis (leftmost)
                # this assumes that a policy is always specified as the rightmost axis of Bs
                #print(f"actions_tree_k: {actions_tree}")
                B = jtu.tree_map(lambda b, a_idx: jnp.moveaxis(b[..., a_idx], -1, 0), B, actions_tree)
                #B = jtu.tree_map(lambda b: [jnp.moveaxis(b[..., a_idx], -1, 0) for a_idx in actions_tree], B)
        else:
            B = None

        # outputs of both VMP and MMP should be a list of hidden state factors, where each qs[f].shape = (T, batch_dim, num_states_f)
        if method == 'vmp':
            qs = run_vmp(A, B, obs, prior, A_dependencies, B_dependencies, num_iter=num_iter) 
        def run_mmp_vfe_single(b):
            b=list(b)
            
            return run_mmp_vfe(A, b, obs, prior, A_dependencies, B_dependencies, num_iter=num_iter)
        
        """ def run_mmp_vfe_single(b):
            # bがリストの場合、各要素を個別に処理
            if isinstance(b, list):
                # 各要素を個別に処理
                results = []
                for b_i in b:
                    b_i_array = jnp.array(b_i)
                    result = run_mmp_vfe(A, [b_i_array], obs, prior, A_dependencies, B_dependencies, num_iter=num_iter)
                    results.append(result)
                # 結果を結合
                qs, err, vfe, kld, bs, un = zip(*results)
                return qs, err, vfe, kld, bs, un
            else:
                return run_mmp_vfe(A, b, obs, prior, A_dependencies, B_dependencies, num_iter=num_iter) """
       
        if method == 'mmp':
            # Use vmap to parallelize the function over the batch dimension
            if B is not None:
            #qs, err, vfe, kld, bs, un = vmap(run_mmp_vfe_single)(jnp.array(B))
                if past_actions.shape[0]>=t:
                    #print('run_mmp_vfe_single')
                    #print(f"B: {B}")
                    # ベクトル化
                    #results = vmap(run_mmp_vfe_partial)(B) 
                    #print(B)
                    results = [run_mmp_vfe_single(b) for b in B]
                    qs, err, vfe, kld, bs, un = zip(*results)
                    #B = [jnp.array(b_i) for b_i in B]
                    #qs, err, vfe, kld, bs, un = vmap(run_mmp_vfe_single)(B_list)
                    # B_list = [[jnp.array(b_ij) 
                    # for b_ij in b_i] 
                    # for b_i in B]
                    #qs, err, vfe, kld, bs, un = vmap(run_mmp_vfe_single)(B)
                    #qs, err, vfe, kld, bs, un = vmap(run_mmp_vfe_single,(0,))(B)
                    """ B=jnp.array(B)##この書き方だと二因子以上かつBの形状が異なる場合にエラーとなる
                    qs, err, vfe, kld, bs, un = vmap(run_mmp_vfe_single,(0,))(B) """
                    #print(B.shape)
                    """ results = []
                    for b_i in B:
                        # 各要素を個別に処理
                        result = run_mmp_vfe_single(b_i)
                        results.append(result)
                    
                    # 結果を結合
                    qs, err, vfe, kld, bs, un = zip(*results) """


                    """ results = vmap(run_mmp_vfe_single,(0,))(B)
                    print(results)
                    qs, err, vfe, kld, bs, un = zip(*results) """
                    
                    #qs, err, vfe, kld, bs, un = vmap(run_mmp_vfe_single)(B)#リストのまま実行
                    
                    #results = [run_mmp_vfe_single(b) for b in B]
                    """ qs=jnp.array(qs)
                    #print(qs.shape)
                    qs=jnp.moveaxis(qs, 0, 1) """
                    #print(f"qs: {qs}")#ポリシー，因子，バッチ？，時刻，状態数
                    # qsの各要素を個別に処理
                    """ qs_arrays = []
                    for q in qs:
                        if isinstance(q, (list, tuple)):
                            # 各サブ配列を個別に処理
                            sub_arrays = [jnp.asarray(sub_q) for sub_q in q]
                            # 必要に応じて形状を調整
                            sub_arrays = [sub_q.reshape(-1, 1) if len(sub_q.shape) == 1 else sub_q for sub_q in sub_arrays]
                            qs_arrays.append(sub_arrays)
                        else:
                            # 単一の配列の場合
                            q_array = jnp.asarray(q)
                            if len(q_array.shape) == 1:
                                q_array = q_array.reshape(-1, 1)
                            qs_arrays.append(q_array)

                    print(f"qs_arrays: {qs_arrays}")

                    # 軸の順序を調整（リストの場合は再帰的に処理）
                    def adjust_axes(q):
                        if isinstance(q, list):
                            return [adjust_axes(sub_q) for sub_q in q]
                        else:
                            return jnp.moveaxis(q, 0, 1) if len(q.shape) > 1 else q

                    qs = [adjust_axes(q) for q in qs_arrays]

                    # 軸の順序を調整
                    qs = [jnp.moveaxis(q, 0, 1) if len(q.shape) > 1 else q for q in qs_arrays] """

                    #print(f"vfe: {vfe}")
                    """ vfe=jtu.tree_map(lambda x:(lambda y: y.sum(2),x),vfe)
                    kld=jtu.tree_map(lambda x:(lambda y: y.sum(2),x),kld)
                    bs=jtu.tree_map(lambda x:(lambda y: y.sum(2),x),bs)
                    un=jtu.tree_map(lambda x:(lambda y: y.sum(2),x),un) """
                    # タプルをリストに変換
                    vfe = list(vfe)
                    kld = list(kld)
                    bs = list(bs)
                    un = list(un)

                    # その後で操作を行う
                    for i in range(len(vfe)):
                        vfe[i] = jtu.tree_map(lambda y: y.sum(1), vfe[i])
                        kld[i] = jtu.tree_map(lambda y: y.sum(1), kld[i])
                        bs[i] = jtu.tree_map(lambda y: y.sum(1), bs[i])
                        un[i] = jtu.tree_map(lambda y: y.sum(1), un[i])

                    #print(f"vfe: {vfe}")
                    vfe = [list(p) for p in zip(*vfe)]#ポリシー，因子，，，から，因子，ポリシー，バッチ？，時点に変更
                    #print(f"vfe: {vfe}")
                    kld = [list(p) for p in zip(*kld)]
                    bs = [list(p) for p in zip(*bs)]
                    un = [list(p) for p in zip(*un)]

                    qs = [list(p) for p in zip(*qs)]
                    err = [list(p) for p in zip(*err)]

                    """ print(f"vfe: {vfe}")
                    for i in range(len(vfe)):
                        print(f"vfe[i]: {vfe[i]}")
                        vfe[i]=jtu.tree_map(lambda y: y.sum(1),vfe[i])
                        kld[i]=jtu.tree_map(lambda y: y.sum(1),kld[i])
                        bs[i]=jtu.tree_map(lambda y: y.sum(1),bs[i])
                        un[i]=jtu.tree_map(lambda y: y.sum(1),un[i]) """
                    
                    """ vfe=jtu.tree_map(lambda y: y.sum(2),vfe)
                    kld=jtu.tree_map(lambda y: y.sum(2),kld)
                    bs=jtu.tree_map(lambda y: y.sum(2),bs)
                    un=jtu.tree_map(lambda y: y.sum(2),un) """
                    qs = list(qs)
                    #print(len(qs))
                    #print(qs[0].shape)
                    err = list(err)
                else:
                    qs, err, vfe, kld, bs, un = run_mmp_vfe(A, B, obs, prior, A_dependencies, B_dependencies, num_iter=num_iter)
                    vfe=jtu.tree_map(lambda y: y.sum(1),vfe)
                    kld=jtu.tree_map(lambda y: y.sum(1),kld)
                    bs=jtu.tree_map(lambda y: y.sum(1),bs)
                    un=jtu.tree_map(lambda y: y.sum(1),un)
                    #realqs=qs
                    #print(qs[0].shape)
            else:
                
                qs, err, vfe, kld, bs, un = run_mmp_vfe(A, B, obs, prior, A_dependencies, B_dependencies, num_iter=num_iter)
                vfe=jtu.tree_map(lambda y: y.sum(1),vfe)
                kld=jtu.tree_map(lambda y: y.sum(1),kld)
                bs=jtu.tree_map(lambda y: y.sum(1),bs)
                un=jtu.tree_map(lambda y: y.sum(1),un)
                #print(qs[0].shape)
                #realqs=qs
    if qs_hist is not None:
        if method == 'fpi' or method == "ovf":
            qs_hist = jtu.tree_map(lambda x, y: jnp.concatenate([x, jnp.expand_dims(y, 0)], 0), qs_hist, qs)
        else:
            #TODO: return entire history of beliefs
            #qs_hist = realqs
            qs_hist=qs
    else:
        if method == 'fpi' or method == "ovf":
            qs_hist = jtu.tree_map(lambda x: jnp.expand_dims(x, 0), qs)
        else:
            #qs_hist = realqs
            qs_hist=qs
    """ vfe = jnp.array(vfe)
    kld = jnp.array(kld)
    bs = jnp.array(bs)
    un = jnp.array(un)  """
    
    return qs_hist, err, vfe, kld, bs, un

def update_posterior_states2(
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
            #qs = run_mmp(A, B, obs, prior, A_dependencies, B_dependencies, num_iter=num_iter)
            qs, err = run_mmp2(A, B, obs, prior, A_dependencies, B_dependencies, num_iter=num_iter)
    
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
    
    #return qs_hist
    return qs_hist, err

def update_posterior_states_vfe_set_prior(
        A, 
        B, 
        obs, 
        past_actions, 
        prior=None, 
        qs_hist=None, 
        expected_states=None,#past_beliefs
        A_dependencies=None, 
        B_dependencies=None, 
        num_iter=16, 
        method='fpi',
        tau=1.
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
            actions_tree = [past_actions[:, i] for i in range(nf)] #過去とった行動のリストを作成;Make a list of actions you have taken in the past.
            
            # move time steps to the leading axis (leftmost)
            # this assumes that a policy is always specified as the rightmost axis of Bs
            B = jtu.tree_map(lambda b, a_idx: jnp.moveaxis(b[..., a_idx], -1, 0), B, actions_tree) #過去とった行動に対応するB行列のリストを作成．;Create a list of B matrices corresponding to past actions.
        else:
            B = None

        # outputs of both VMP and MMP should be a list of hidden state factors, where each qs[f].shape = (T, batch_dim, num_states_f)
        if method == 'vmp':
            qs = run_vmp(A, B, obs, prior, A_dependencies, B_dependencies, num_iter=num_iter) 
        if method == 'mmp':
            #MMPにもとづき認識分布（qs）やvfeの計算;Calculation of recognition distribution (qs) and vfe based on MMP
            if expected_states is not None and qs_hist is not None:
                expected_states = jtu.tree_map(lambda x: x[None, ...], expected_states) # 2次元配列を3次元配列に変換
                #print(expected_states[0].shape)#1,100
                #print(qs_hist[0].shape)#1,100
                    #print("combine")
                past_beliefs = jtu.tree_map(lambda x, y: jnp.concatenate((x, y), axis=0), qs_hist, expected_states)
                #print(past_beliefs[0].shape)#
                #past_beliefs = jtu.tree_map(lambda x: x.squeeze(0), past_beliefs)  # 3次元配列を2次元配列に変換
            else:
                past_beliefs = prior
                past_beliefs = jtu.tree_map(lambda x: jnp.expand_dims(x, 0),past_beliefs)
            qs, err, vfe, kld2, bs, un, qs_1step, err_1step, vfe_1step, kld2_1step, bs_1step, un_1step= run_mmp_vfe_set_prior(A, B, obs, prior, A_dependencies, B_dependencies, num_iter=num_iter,tau=tau, past_beliefs=past_beliefs)#qs, err, vfe, kld, bs, un
    
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
    
    return qs_hist, err, vfe, kld2, bs, un, qs_1step, err_1step, vfe_1step, kld2_1step, bs_1step, un_1step#qs_hist, err, vfe, kld, bs, un
    