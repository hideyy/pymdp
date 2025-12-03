#!/usr/bin/env python
# -*- coding: utf-8 -*-

""" Agent Class implementation in Jax

__author__: Conor Heins, Dimitrije Markovic, Alexander Tschantz, Daphne Demekas, Brennan Klein

"""

import math as pymath
import jax.numpy as jnp
import jax.tree_util as jtu
from jax import nn, vmap, random
from . import inference, control, learning, utils, maths
from equinox import Module, field, tree_at

from typing import List, Optional
from jaxtyping import Array
from functools import partial

from pymdp.jax.maths import *

class Agent(Module):
    """ 
    The Agent class, the highest-level API that wraps together processes for action, perception, and learning under active inference.

    The basic usage is as follows:

    >>> my_agent = Agent(A = A, B = C, <more_params>)
    >>> observation = env.step(initial_action)
    >>> qs = my_agent.infer_states(observation)
    >>> q_pi, G = my_agent.infer_policies()
    >>> next_action = my_agent.sample_action()
    >>> next_observation = env.step(next_action)

    This represents one timestep of an active inference process. Wrapping this step in a loop with an ``Env()`` class that returns
    observations and takes actions as inputs, would entail a dynamic agent-environment interaction.
    """

    A: List[Array]
    B: List[Array]
    C: List[Array] 
    D: List[Array]
    E: Array
    # empirical_prior: List
    gamma: Array
    alpha: Array
    qs: Optional[List[Array]]
    q_pi: Optional[List[Array]]

    # parameters used for inductive inference
    inductive_threshold: Array # threshold for inductive inference (the threshold for pruning transitions that are below a certain probability)
    inductive_epsilon: Array # epsilon for inductive inference (trade-off/weight for how much inductive value contributes to EFE of policies)

    H: List[Array] # H vectors (one per hidden state factor) used for inductive inference -- these encode goal states or constraints
    I: List[Array] # I matrices (one per hidden state factor) used for inductive inference -- these encode the 'reachability' matrices of goal states encoded in `self.H`

    pA: List[Array]
    pB: List[Array]

    policies: Array # matrix of all possible policies (each row is a policy of shape (num_controls[0], num_controls[1], ..., num_controls[num_control_factors-1])
    
    # static parameters not leaves of the PyTree
    A_dependencies: Optional[List[int]] = field(static=True)
    B_dependencies: Optional[List[int]] = field(static=True)
    batch_size: int = field(static=True)
    num_iter: int = field(static=True)
    num_obs: List[int] = field(static=True)
    num_modalities: int = field(static=True)
    num_states: List[int] = field(static=True)
    num_factors: int = field(static=True)
    num_controls: List[int] = field(static=True)
    control_fac_idx: Optional[List[int]] = field(static=True)
    policy_len: int = field(static=True) # depth of planning during roll-outs (i.e. number of timesteps to look ahead when computing expected free energy of policies)
    inductive_depth: int = field(static=True) # depth of inductive inference (i.e. number of future timesteps to use when computing inductive `I` matrix)
    use_utility: bool = field(static=True) # flag for whether to use expected utility ("reward" or "preference satisfaction") when computing expected free energy
    use_states_info_gain: bool = field(static=True) # flag for whether to use state information gain ("salience") when computing expected free energy
    use_param_info_gain: bool = field(static=True)  # flag for whether to use parameter information gain ("novelty") when computing expected free energy
    use_inductive: bool = field(static=True)   # flag for whether to use inductive inference ("intentional inference") when computing expected free energy
    onehot_obs: bool = field(static=True)
    action_selection: str = field(static=True) # determinstic or stochastic action selection 
    sampling_mode : str = field(static=True) # whether to sample from full posterior over policies ("full") or from marginal posterior over actions ("marginal")
    inference_algo: str = field(static=True) # fpi, vmp, mmp, ovf

    learn_A: bool = field(static=True)
    learn_B: bool = field(static=True)
    learn_C: bool = field(static=True)
    learn_D: bool = field(static=True)
    learn_E: bool = field(static=True)

    def __init__(
        self,
        A,
        B,
        C,
        D,
        E,
        pA,
        pB,
        A_dependencies=None,
        B_dependencies=None,
        qs=None,
        q_pi=None,
        H=None,
        I=None,
        policy_len=1,
        control_fac_idx=None,
        policies=None,
        gamma=1.0,
        alpha=1.0,
        inductive_depth=1,
        inductive_threshold=0.1,
        inductive_epsilon=1e-3,
        use_utility=True,
        use_states_info_gain=True,
        use_param_info_gain=False,
        use_inductive=False,
        onehot_obs=False,
        action_selection="deterministic",
        sampling_mode="full",
        inference_algo="fpi",
        num_iter=16,
        learn_A=True,
        learn_B=True,
        learn_C=False,
        learn_D=True,
        learn_E=False
    ):
        ### PyTree leaves
        self.A = A
        self.B = B
        self.C = C
        self.D = D
        # self.empirical_prior = D
        self.H = H
        self.pA = pA
        self.pB = pB
        self.qs = qs
        self.q_pi = q_pi

        self.onehot_obs = onehot_obs

        element_size = lambda x: x.shape[1]
        self.num_factors = len(self.B)
        self.num_states = jtu.tree_map(element_size, self.B) 

        self.num_modalities = len(self.A)
        self.num_obs = jtu.tree_map(element_size, self.A)

        # Ensure consistency of A_dependencies with num_states and num_factors
        if A_dependencies is not None:
            self.A_dependencies = A_dependencies
        else:
            # assume full dependence of A matrices and state factors
            self.A_dependencies = [list(range(self.num_factors)) for _ in range(self.num_modalities)]
        
        for m in range(self.num_modalities):
            factor_dims = tuple([self.num_states[f] for f in self.A_dependencies[m]])
            assert self.A[m].shape[2:] == factor_dims, f"Please input an `A_dependencies` whose {m}-th indices correspond to the hidden state factors that line up with lagging dimensions of A[{m}]..." 
            if self.pA != None:
                assert self.pA[m].shape[2:] == factor_dims if self.pA[m] is not None else True, f"Please input an `A_dependencies` whose {m}-th indices correspond to the hidden state factors that line up with lagging dimensions of pA[{m}]..." 
            assert max(self.A_dependencies[m]) <= (self.num_factors - 1), f"Check modality {m} of `A_dependencies` - must be consistent with `num_states` and `num_factors`..."
           
        # Ensure consistency of B_dependencies with num_states and num_factors
        if B_dependencies is not None:
            self.B_dependencies = B_dependencies
        else:
            self.B_dependencies = [[f] for f in range(self.num_factors)] # defaults to having all factors depend only on themselves

        for f in range(self.num_factors):
            factor_dims = tuple([self.num_states[f] for f in self.B_dependencies[f]])
            assert self.B[f].shape[2:-1] == factor_dims, f"Please input a `B_dependencies` whose {f}-th indices pick out the hidden state factors that line up with the all-but-final lagging dimensions of B[{f}]..." 
            if self.pB != None:
                assert self.pB[f].shape[2:-1] == factor_dims, f"Please input a `B_dependencies` whose {f}-th indices pick out the hidden state factors that line up with the all-but-final lagging dimensions of pB[{f}]..." 
            assert max(self.B_dependencies[f]) <= (self.num_factors - 1), f"Check factor {f} of `B_dependencies` - must be consistent with `num_states` and `num_factors`..."

        self.batch_size = self.A[0].shape[0]

        self.gamma = jnp.broadcast_to(gamma, (self.batch_size,))
        self.alpha = jnp.broadcast_to(alpha, (self.batch_size,))
        self.inductive_threshold = jnp.broadcast_to(inductive_threshold, (self.batch_size,))
        self.inductive_epsilon = jnp.broadcast_to(inductive_epsilon, (self.batch_size,))

        ### Static parameters ###
        self.num_iter = num_iter
        self.inference_algo = inference_algo
        self.inductive_depth = inductive_depth

        # policy parameters
        self.policy_len = policy_len
        self.action_selection = action_selection
        self.sampling_mode = sampling_mode
        self.use_utility = use_utility
        self.use_states_info_gain = use_states_info_gain
        self.use_param_info_gain = use_param_info_gain
        self.use_inductive = use_inductive

        if self.use_inductive and self.H is not None:
            # print("Using inductive inference...")
            self.I = self._construct_I()
        elif self.use_inductive and I is not None:
            self.I = I
        else:
            self.I = jtu.tree_map(lambda x: jnp.expand_dims(jnp.zeros_like(x), 1), self.D)

        # learning parameters
        self.learn_A = learn_A
        self.learn_B = learn_B
        self.learn_C = learn_C
        self.learn_D = learn_D
        self.learn_E = learn_E

        """ Determine number of observation modalities and their respective dimensions """
        self.num_obs = [self.A[m].shape[1] for m in range(len(self.A))]
        self.num_modalities = len(self.num_obs)

        # If no `num_controls` are given, then this is inferred from the shapes of the input B matrices
        self.num_controls = [self.B[f].shape[-1] for f in range(self.num_factors)]

        # Users have the option to make only certain factors controllable.
        # default behaviour is to make all hidden state factors controllable
        # (i.e. self.num_states == self.num_controls)
        # Users have the option to make only certain factors controllable.
        # default behaviour is to make all hidden state factors controllable, i.e. `self.num_factors == len(self.num_controls)`
        if control_fac_idx == None:
            self.control_fac_idx = [f for f in range(self.num_factors) if self.num_controls[f] > 1]
        else:
            assert max(control_fac_idx) <= (self.num_factors - 1), "Check control_fac_idx - must be consistent with `num_states` and `num_factors`..."
            self.control_fac_idx = control_fac_idx

            for factor_idx in self.control_fac_idx:
                assert self.num_controls[factor_idx] > 1, "Control factor (and B matrix) dimensions are not consistent with user-given control_fac_idx"

        if policies is not None:
            self.policies = policies
        else:
            self._construct_policies()
        
        # set E to uniform/uninformative prior over policies if not given
        if E is None:
            self.E = jnp.ones((self.batch_size, len(self.policies)))/ len(self.policies)
        else:
            self.E = E

    def _construct_policies(self):
        
        self.policies =  control.construct_policies(
            self.num_states, self.num_controls, self.policy_len, self.control_fac_idx
        )

    @vmap
    def _construct_I(self):
        return control.generate_I_matrix(self.H, self.B, self.inductive_threshold, self.inductive_depth)

    @property
    def unique_multiactions(self):
        size = pymath.prod(self.num_controls)
        return jnp.unique(self.policies[:, 0], axis=0, size=size, fill_value=-1)

    def infer_parameters(self, beliefs_A, outcomes, actions, beliefs_B=None, lr_pA=1., lr_pB=1., fr_pA=1., fr_pB=1., **kwargs):
        agent = self
        beliefs_B = beliefs_A if beliefs_B is None else beliefs_B
        if self.inference_algo == 'ovf':
            smoothed_marginals_and_joints = vmap(inference.smoothing_ovf)(beliefs_A, self.B, actions)
            marginal_beliefs = smoothed_marginals_and_joints[0]
            joint_beliefs = smoothed_marginals_and_joints[1]
        else:
            marginal_beliefs = beliefs_A
            if self.learn_B:
                nf = len(beliefs_B)
                joint_fn = lambda f: [beliefs_B[f][:, 1:]] + [beliefs_B[f_idx][:, :-1] for f_idx in self.B_dependencies[f]]
                joint_beliefs = jtu.tree_map(joint_fn, list(range(nf)))

        if self.learn_A:
            update_A = partial(
                learning.update_obs_likelihood_dirichlet,
                A_dependencies=self.A_dependencies,
                num_obs=self.num_obs,
                onehot_obs=self.onehot_obs,
            )
            
            lr = jnp.broadcast_to(lr_pA, (self.batch_size,))
            fr = jnp.broadcast_to(fr_pA, (self.batch_size,))
            qA, E_qA = vmap(update_A)(
                self.pA,
                self.A,
                outcomes,
                marginal_beliefs,
                lr=lr,
                fr=fr,
            )
            
            agent = tree_at(lambda x: (x.A, x.pA), agent, (E_qA, qA))
            
        if self.learn_B:
            assert beliefs_B[0].shape[1] == actions.shape[1] + 1
            update_B = partial(
                learning.update_state_transition_dirichlet,
                num_controls=self.num_controls
            )

            lr = jnp.broadcast_to(lr_pB, (self.batch_size,))
            fr = jnp.broadcast_to(fr_pB, (self.batch_size,))
            qB, E_qB = vmap(update_B)(
                self.pB,
                self.B,
                joint_beliefs,
                actions,
                lr=lr,
                fr=fr
            )
            
            # if you have updated your beliefs about transitions, you need to re-compute the I matrix used for inductive inferenece
            if self.use_inductive and self.H is not None:
                I_updated = vmap(control.generate_I_matrix)(self.H, E_qB, self.inductive_threshold, self.inductive_depth)
            else:
                I_updated = self.I

            agent = tree_at(lambda x: (x.B, x.pB, x.I), agent, (E_qB, qB, I_updated))

        return agent
    
    def infer_states(self, observations, empirical_prior, *, past_actions=None, qs_hist=None, mask=None):
        """
        Update approximate posterior over hidden states by solving variational inference problem, given an observation.

        Parameters
        ----------
        observations: ``list`` or ``tuple`` of ints
            The observation input. Each entry ``observation[m]`` stores one-hot vectors representing the observations for modality ``m``.
        past_actions: ``list`` or ``tuple`` of ints
            The action input. Each entry ``past_actions[f]`` stores indices (or one-hots?) representing the actions for control factor ``f``.
        empirical_prior: ``list`` or ``tuple`` of ``jax.numpy.ndarray`` of dtype object
            Empirical prior beliefs over hidden states. Depending on the inference algorithm chosen, the resulting ``empirical_prior`` variable may be a matrix (or list of matrices) 
            of additional dimensions to encode extra conditioning variables like timepoint and policy.
        Returns
        ---------
        qs: ``numpy.ndarray`` of dtype object
            Posterior beliefs over hidden states. Depending on the inference algorithm chosen, the resulting ``qs`` variable will have additional sub-structure to reflect whether
            beliefs are additionally conditioned on timepoint and policy.
            For example, in case the ``self.inference_algo == 'MMP' `` indexing structure is policy->timepoint-->factor, so that 
            ``qs[p_idx][t_idx][f_idx]`` refers to beliefs about marginal factor ``f_idx`` expected under policy ``p_idx`` 
            at timepoint ``t_idx``.
        """
        if not self.onehot_obs:
            o_vec = [nn.one_hot(o, self.num_obs[m]) for m, o in enumerate(observations)]
        else:
            o_vec = observations
        
        A = self.A
        if mask is not None:
            for i, m in enumerate(mask):
                o_vec[i] = m * o_vec[i] + (1 - m) * jnp.ones_like(o_vec[i]) / self.num_obs[i]
                A[i] = m * A[i] + (1 - m) * jnp.ones_like(A[i]) / self.num_obs[i]

        infer_states = partial(
            inference.update_posterior_states,
            A_dependencies=self.A_dependencies,
            B_dependencies=self.B_dependencies,
            num_iter=self.num_iter,
            method=self.inference_algo
        )
        
        output = vmap(infer_states)(
            A,
            self.B,
            o_vec,
            past_actions,
            prior=empirical_prior,
            qs_hist=qs_hist
        )

        return output

    def update_empirical_prior(self, action, qs):
        # return empirical_prior, and the history of posterior beliefs (filtering distributions) held about hidden states at times 1, 2 ... t

        # this computation of the predictive prior is correct only for fully factorised Bs.
        if self.inference_algo in ['mmp', 'vmp']:
            # in the case of the 'mmp' or 'vmp' we have to use D as prior parameter for infer states
            pred = self.D
        else:
            qs_last = jtu.tree_map( lambda x: x[:, -1], qs)
            propagate_beliefs = partial(control.compute_expected_state, B_dependencies=self.B_dependencies)
            pred = vmap(propagate_beliefs)(qs_last, self.B, action)
        
        return (pred, qs)

    def infer_policies(self, qs: List):
        """
        Perform policy inference by optimizing a posterior (categorical) distribution over policies.
        This distribution is computed as the softmax of ``G * gamma + lnE`` where ``G`` is the negative expected
        free energy of policies, ``gamma`` is a policy precision and ``lnE`` is the (log) prior probability of policies.
        This function returns the posterior over policies as well as the negative expected free energy of each policy.

        Returns
        ----------
        q_pi: 1D ``numpy.ndarray``
            Posterior beliefs over policies, i.e. a vector containing one posterior probability per policy.
        G: 1D ``numpy.ndarray``
            Negative expected free energies of each policy, i.e. a vector containing one negative expected free energy per policy.
        """

        latest_belief = jtu.tree_map(lambda x: x[:, -1], qs) # only get the posterior belief held at the current timepoint
        infer_policies = partial(
            control.update_posterior_policies_inductive,
            self.policies,
            A_dependencies=self.A_dependencies,
            B_dependencies=self.B_dependencies,
            use_utility=self.use_utility,
            use_states_info_gain=self.use_states_info_gain,
            use_param_info_gain=self.use_param_info_gain,
            use_inductive=self.use_inductive
        )

        q_pi, G = vmap(infer_policies)(
            latest_belief, 
            self.A,
            self.B,
            self.C,
            self.E,
            self.pA,
            self.pB,
            I = self.I,
            gamma=self.gamma,
            inductive_epsilon=self.inductive_epsilon
        )

        return q_pi, G
    
    def multiaction_probabilities(self, q_pi: Array):
        """
        Compute probabilities of unique multi-actions from the posterior over policies.

        Parameters
        ----------
        q_pi: 1D ``numpy.ndarray``
        Posterior beliefs over policies, i.e. a vector containing one posterior probability per policy.
        
        Returns
        ----------
        multi-action: 1D ``jax.numpy.ndarray``
            Vector containing probabilities of possible multi-actions for different factors
        """

        if self.sampling_mode == "marginal":
            get_marginals = partial(control.get_marginals, policies=self.policies, num_controls=self.num_controls)
            marginals = get_marginals(q_pi)
            outer = lambda a, b: jnp.outer(a, b).reshape(-1)
            marginals = jtu.tree_reduce(outer, marginals)

        elif self.sampling_mode == "full":
            locs = jnp.all(
                self.policies[:, 0] == jnp.expand_dims(self.unique_multiactions, -2),
                  -1
            )
            get_marginals = lambda x: jnp.where(locs, x, 0.).sum(-1)
            marginals = vmap(get_marginals)(q_pi)

        return marginals

    def sample_action(self, q_pi: Array, rng_key=None):
        """
        Sample or select a discrete action from the posterior over control states.
        
        Returns
        ----------
        action: 1D ``jax.numpy.ndarray``
            Vector containing the indices of the actions for each control factor
        action_probs: 2D ``jax.numpy.ndarray``
            Array of action probabilities
        """

        if (rng_key is None) and (self.action_selection == "stochastic"):
            raise ValueError("Please provide a random number generator key to sample actions stochastically")

        if self.sampling_mode == "marginal":
            sample_action = partial(control.sample_action, self.policies, self.num_controls, action_selection=self.action_selection)
            action = vmap(sample_action)(q_pi, alpha=self.alpha, rng_key=rng_key)
        elif self.sampling_mode == "full":
            sample_policy = partial(control.sample_policy, self.policies, action_selection=self.action_selection)
            action = vmap(sample_policy)(q_pi, alpha=self.alpha, rng_key=rng_key)

        return action
    
    def _get_default_params(self):
        method = self.inference_algo
        default_params = None
        if method == "VANILLA":
            default_params = {"num_iter": 8, "dF": 1.0, "dF_tol": 0.001}
        elif method == "MMP":
            raise NotImplementedError("MMP is not implemented")
        elif method == "VMP":
            raise NotImplementedError("VMP is not implemented")
        elif method == "BP":
            raise NotImplementedError("BP is not implemented")
        elif method == "EP":
            raise NotImplementedError("EP is not implemented")
        elif method == "CV":
            raise NotImplementedError("CV is not implemented")

        return default_params
    
    def infer_states_vfe(self, observations, empirical_prior, *, past_actions=None, qs_hist=None, mask=None):
        """
        Update approximate posterior over hidden states by solving variational inference problem, given an observation.

        Parameters
        ----------
        observations: ``list`` or ``tuple`` of ints
            The observation input. Each entry ``observation[m]`` stores one-hot vectors representing the observations for modality ``m``.
        past_actions: ``list`` or ``tuple`` of ints
            The action input. Each entry ``past_actions[f]`` stores indices (or one-hots?) representing the actions for control factor ``f``.
        empirical_prior: ``list`` or ``tuple`` of ``jax.numpy.ndarray`` of dtype object
            Empirical prior beliefs over hidden states. Depending on the inference algorithm chosen, the resulting ``empirical_prior`` variable may be a matrix (or list of matrices) 
            of additional dimensions to encode extra conditioning variables like timepoint and policy.
        Returns
        ---------
        qs: ``numpy.ndarray`` of dtype object
            Posterior beliefs over hidden states. Depending on the inference algorithm chosen, the resulting ``qs`` variable will have additional sub-structure to reflect whether
            beliefs are additionally conditioned on timepoint and policy.
            For example, in case the ``self.inference_algo == 'MMP' `` indexing structure is policy->timepoint-->factor, so that 
            ``qs[p_idx][t_idx][f_idx]`` refers to beliefs about marginal factor ``f_idx`` expected under policy ``p_idx`` 
            at timepoint ``t_idx``.
        """
        if not self.onehot_obs:
            #print("convert to distribution")
            o_vec = [nn.one_hot(o, self.num_obs[m]) for m, o in enumerate(observations)]#観測値のワンホットベクトル化; One-hot vectorization of observed values
            #print(o_vec)
        else:
            o_vec = observations
        
        A = self.A
        if mask is not None:
            for i, m in enumerate(mask):
                o_vec[i] = m * o_vec[i] + (1 - m) * jnp.ones_like(o_vec[i]) / self.num_obs[i]
                A[i] = m * A[i] + (1 - m) * jnp.ones_like(A[i]) / self.num_obs[i]

        infer_states = partial(
            inference.update_posterior_states_vfe,
            A_dependencies=self.A_dependencies,
            B_dependencies=self.B_dependencies,
            num_iter=self.num_iter,
            method=self.inference_algo
        )#並列処理のための関数の宣言;Declaring functions for parallel processing
        
        output, err, vfe, kld2, bs, un = vmap(infer_states)(  #output, err, vfe, kld, bs, un 
            A,
            self.B,
            o_vec,
            past_actions,
            prior=empirical_prior,
            qs_hist=qs_hist
        )#並列計算で認識分布（output）やvfeの計算;Parallel computation of recognition distribution (output) and vfe
        #vfe=vfe[0].sum(2)
        vfe=jtu.tree_map(lambda x: x.sum(2),vfe)#状態量の次元に沿ってVFEを足し上げ．;Add up VFE along the dimensions of the state variables.
        err=jtu.tree_map(lambda x: x.sum(2),err)
        kld2=jtu.tree_map(lambda x: x.sum(2),kld2)
        #kld=jtu.tree_map(lambda x: x.sum(2),kld)
        bs=jtu.tree_map(lambda x: x.sum(2),bs)
        un=jtu.tree_map(lambda x: x.sum(2),un)

        return output, err, vfe, kld2, bs, un#output, err, vfe, kld(S_Hqs)(po), bs, un
    
    #@vmap
    def calc_KLD_past_currentqs(self, empirical_prior, past_qs, current_qs):
        """
        認識によって減少する自由エネルギーを計算．mmpを想定．
        t:現在のタイムステップ
        empirical_prior:t-1におけるtの状態に対する予測,t-1におけるupdate_empirical_priorの出力pred
        past_qs:t-1におけるt-1までの状態に関する信念,t-1におけるupdate_empirical_priorの出力qs
        current_qs:tにおけるtまでの状態に関する信念,tにおけるinfer_statesの出力qs_hist
        D_KL([past_qs, prior]|current_qs)を計算．
        """
        #past_beliefs = jtu.tree_map(lambda x, y: jnp.concatenate([x, y[None, ...]], axis=1), past_qs, empirical_prior)
        #past_beliefs = jtu.tree_map(lambda x, y: jnp.concatenate([x, y], axis=1), past_qs, empirical_prior)
        #past_beliefs = jtu.tree_map(lambda x, y: jnp.concatenate([x, y[:, None, :]], axis=1), past_qs, empirical_prior)
        
        if past_qs is not None:
            #past_beliefs = jtu.tree_map(lambda x, y: jnp.concatenate([x, y], axis=1), past_qs, jnp.expand_dims(empirical_prior, axis=1))

            #past_beliefs = jtu.tree_map(lambda x, y: jnp.concatenate([x, y[None, ...]], axis=1), past_qs, empirical_prior)
            
            #past_beliefs = jnp.concatenate((past_qs, empirical_prior), axis=1)
            #past_beliefs = jtu.tree_map(lambda x, y: jnp.concatenate([x, y], axis=1), past_qs, empirical_prior)
            #empirical_prior = jnp.array(empirical_prior)
            #print("combart")
            empirical_prior = jtu.tree_map(lambda x: x[None, ...], empirical_prior) # 2次元配列を3次元配列に変換
            #print("combine")
            past_beliefs = jtu.tree_map(lambda x, y: jnp.concatenate((x, y), axis=1), past_qs, empirical_prior)
            #past_beliefs = jnp.concatenate((past_qs, empirical_prior), axis=1)
            #print(past_beliefs[0].shape)
            #print(current_qs[0].shape)
        else:
            past_beliefs = empirical_prior
        #past_beliefs = jtu.tree_map(lambda x, y: jnp.concatenate([x, jnp.expand_dims(jnp.array(y), axis=1)], axis=1), past_qs, empirical_prior)
        #past_beliefs = jtu.tree_map(lambda x: x.squeeze(axis=0), past_beliefs)
        #past_beliefs = [jnp.array([x]) for x in past_qs] + [jnp.array([y]) for y in empirical_prior]
        #past_beliefs = past_beliefs = [jnp.array([x]) for x in past_qs] + [jnp.array([y]) for y in empirical_prior]
        #past_beliefs = jtu.tree_map(lambda x, y: jnp.concatenate([x, y[None, ...]], axis=0), past_qs, empirical_prior)
        #H_past_beliefs = xlogy(current_qs,current_qs).sum()
        #H_past_beliefs = xlogy(past_beliefs,past_beliefs).sum()
        #past_beliefs_lncurrent_qs = xlogy(past_beliefs, current_qs).sum()
        #kld = H_past_beliefs #- past_beliefs_lncurrent_qs

        #kld = inference.calc_KLD(past_beliefs,current_qs)
        #print("calculate")
        kld = inference.calc_KLD(past_beliefs,current_qs)
        return kld
    
    def infer_policies_efe(self, qs: List,rng_key=None):
        """
        Perform policy inference by optimizing a posterior (categorical) distribution over policies.
        This distribution is computed as the softmax of ``G * gamma + lnE`` where ``G`` is the negative expected
        free energy of policies, ``gamma`` is a policy precision and ``lnE`` is the (log) prior probability of policies.
        This function returns the posterior over policies as well as the negative expected free energy of each policy.

        Returns
        ----------
        q_pi: 1D ``numpy.ndarray``
            Posterior beliefs over policies, i.e. a vector containing one posterior probability per policy.
        G: 1D ``numpy.ndarray``
            Negative expected free energies of each policy, i.e. a vector containing one negative expected free energy per policy.
        """

        latest_belief = jtu.tree_map(lambda x: x[:, -1], qs) # only get the posterior belief held at the current timepoint
        infer_policies = partial(
            control.update_posterior_policies_inductive_efe,
            self.policies,
            A_dependencies=self.A_dependencies,
            B_dependencies=self.B_dependencies,
            use_utility=self.use_utility,
            use_states_info_gain=self.use_states_info_gain,
            use_param_info_gain=self.use_param_info_gain,
            use_inductive=self.use_inductive,
            rng_key=rng_key
        )

        q_pi, G, PBS, PKLD, PFE, oRisk, PBS_pA, PBS_pB,I_B_o,I_B_o_se = vmap(infer_policies)(
            latest_belief, 
            self.A,
            self.B,
            self.C,
            self.E,
            self.pA,
            self.pB,
            I = self.I,
            gamma=self.gamma,
            inductive_epsilon=self.inductive_epsilon
        )
        #print(PBS)
        return q_pi, G, PBS, PKLD, PFE, oRisk, PBS_pA, PBS_pB,I_B_o,I_B_o_se
    
    def infer_states_vfe_policies(self, observations, empirical_prior, *, past_actions=None, qs_hist=None, mask=None):
        """
        Update approximate posterior over hidden states by solving variational inference problem, given an observation.

        Parameters
        ----------
        observations: ``list`` or ``tuple`` of ints
            The observation input. Each entry ``observation[m]`` stores one-hot vectors representing the observations for modality ``m``.
        past_actions: ``list`` or ``tuple`` of ints
            The action input. Each entry ``past_actions[f]`` stores indices (or one-hots?) representing the actions for control factor ``f``.
        empirical_prior: ``list`` or ``tuple`` of ``jax.numpy.ndarray`` of dtype object
            Empirical prior beliefs over hidden states. Depending on the inference algorithm chosen, the resulting ``empirical_prior`` variable may be a matrix (or list of matrices) 
            of additional dimensions to encode extra conditioning variables like timepoint and policy.
        Returns
        ---------
        qs: ``numpy.ndarray`` of dtype object
            Posterior beliefs over hidden states. Depending on the inference algorithm chosen, the resulting ``qs`` variable will have additional sub-structure to reflect whether
            beliefs are additionally conditioned on timepoint and policy.
            For example, in case the ``self.inference_algo == 'MMP' `` indexing structure is policy->timepoint-->factor, so that 
            ``qs[p_idx][t_idx][f_idx]`` refers to beliefs about marginal factor ``f_idx`` expected under policy ``p_idx`` 
            at timepoint ``t_idx``.
        """
        if not self.onehot_obs:
            o_vec = [nn.one_hot(o, self.num_obs[m]) for m, o in enumerate(observations)]
        else:
            o_vec = observations
        
        A = self.A
        if mask is not None:
            for i, m in enumerate(mask):
                o_vec[i] = m * o_vec[i] + (1 - m) * jnp.ones_like(o_vec[i]) / self.num_obs[i]
                A[i] = m * A[i] + (1 - m) * jnp.ones_like(A[i]) / self.num_obs[i]

        
        policies = self.policies
        batch_size = self.batch_size ##1 # number of agents
        policies = jtu.tree_map(lambda x: jnp.broadcast_to(x, (batch_size,) + x.shape), policies)
        _,K, t,_= policies.shape
        #print('t',t)
        #print('K',K)
        #print(past_actions.shape)
        """ selected_policy=-1
        if past_actions is not None:
            #print(past_actions.shape)
            if past_actions.shape[1]>=t:
                 #for i in range(K):
                    #print(past_actions[:,-t:])
                #print(past_actions[:,-t:][0])
                #print(policies[0][0])
                    #if jnp.array_equal(past_actions[:,-t:][0], policies[0][i]):
                        #selected_policy=i 
                selected_policy = jnp.where(
                    jnp.all(past_actions[:, -t:][0] == policies[0], axis=(1, 2)),
                    jnp.arange(K),
                    -1
                ).max()
                #print(selected_policy)
            else:
                selected_policy=-1
        else:
            selected_policy=-1 """
        """ print('selected_policy')
        print(selected_policy) """
    
        #policies = jtu.tree_map(lambda x: jnp.broadcast_to(x, (batch_size,) + x.shape), policies)
        #print(len(policies))
        #print(policies[0].shape)
        """ if past_actions is not None:
            print(len(past_actions))
            print(past_actions[0].shape) """

        infer_states_policies = partial(
            inference.update_posterior_states_vfe_policies,
            A_dependencies=self.A_dependencies,
            B_dependencies=self.B_dependencies,
            num_iter=self.num_iter,
            method=self.inference_algo
        )
        
        output, err, vfe, kld, bs, un = vmap(infer_states_policies)(#vmap(infer_states)(
            A,
            self.B,
            o_vec,
            policies,
            past_actions,
            prior=empirical_prior,
            qs_hist=qs_hist
        )
        """ if selected_policy !=-1:
            #output=jnp.array(output[0][selected_policyi])
            #output = jtu.tree_map(lambda x: jnp.broadcast_to(x, (batch_size,) + x.shape), output)
            #print(output) 
            output = jtu.tree_map(lambda x: x[0][selected_policy], output)
            output = output[0] """
        #print(output)
        # output = jnp.where(selected_policy != -1, jnp.array(jtu.tree_map(
        #    lambda x: x[0][selected_policy],output)[0]), jnp.array(output))

        #output = jtu.tree_map(lambda x: x[0][:,:-t],output)[0]
        #print(output[0].shape)
        """ if past_actions is not None:
            #print(output[0])
            #print(len(output[0]))
            #output = jtu.tree_map(lambda x: x[:,:-t,:],output[0])#[k][?,F,t]##ポリシー０のqsを取り出す場合
            output = jtu.tree_map(lambda y:jtu.tree_map(lambda x: x[:,:-t,:],y),output) """
            #output = jtu.tree_map(lambda y:jtu.tree_map(lambda x: x[0][:][:-t],y),output)#K,F,t
            #output=output[0]
            #print(output)
        #
        #print(output)
            #output=jtu.tree_map(lambda x: jnp.expand_dims(x, -1).astype(jnp.float32), output) #vfe=vfe[0].sum(2)
        """ vfe=jtu.tree_map(lambda x: x.sum(2),vfe)
        err=jtu.tree_map(lambda x: x.sum(2),err)
        kld=jtu.tree_map(lambda x: x.sum(2),kld)
        bs=jtu.tree_map(lambda x: x.sum(2),bs)
        un=jtu.tree_map(lambda x: x.sum(2),un)
        """
        #vfe=jtu.tree_map(lambda x: jtu.tree_map(lambda y: y.sum(2),x),vfe)
        """ err=jtu.tree_map(lambda x: jtu.tree_map(lambda y: y.sum(2),x),err)
        kld=jtu.tree_map(lambda x: jtu.tree_map(lambda y: y.sum(2),x),kld)
        bs=jtu.tree_map(lambda x: jtu.tree_map(lambda y: y.sum(2),x),bs)
        un=jtu.tree_map(lambda x: jtu.tree_map(lambda y: y.sum(2),x),un) """
       

        return output, err, vfe, kld, bs, un
    
    def infer_policies_posterior(self, neg_efe, vfe_pi, reflect_len=None):
         #print(vfe_pi[0].shape)
         #print(neg_efe[0].shape)
         #vfe_pi=jtu.tree_map(lambda x:jnp.sum(x, axis=-1).flatten(),vfe_pi)
         #vfe_pi=jnp.sum(vfe_pi[0], axis=-1).flatten()
         #print(vfe_pi.shape)
        #agent=self
        if reflect_len is None:
            reflect_len=self.policy_len
        if vfe_pi[0].shape[0]==neg_efe[0].shape[0]:
            print("pi posterior")
            vfe_pi2=vfe_pi[0][:,:,-reflect_len:]
            vfe_pi2=jnp.sum(vfe_pi2, axis=-1).flatten()
            #vfe_pi=jnp.sum(vfe_pi[0], axis=-1).flatten()
            q_pi=nn.softmax(self.gamma * neg_efe + log_stable(self.E) - vfe_pi2)
        else:
            q_pi=nn.softmax(self.gamma * neg_efe + log_stable(self.E))
        return q_pi
    
    def infer_policies_precision(self, neg_efe, vfe_pi, beta=1, policy_len=None):
        agent=self
        if policy_len is None:
            policy_len=self.policy_len
        def scan_fn(carry, iter):
            q_pi, q_pi_0, gamma, Gerror, qb=carry
            #print(vfe_pi[0].shape)
            #print(neg_efe[0])
            if vfe_pi[0].shape[1]==neg_efe[0].shape[0]:
                #print("pi posterior")
                #print(vfe_pi[0][:,:,:])
                #print(reflect_len)
                #print(vfe_pi[0][:,:,-reflect_len-1])
                vfe_pi2=vfe_pi[0][:,:,-policy_len-1]
                vfe_pi2=vfe_pi2.flatten()
                #vfe_pi2=jnp.sum(vfe_pi2, axis=-1).flatten()
                #print(vfe_pi2)
                q_pi=nn.softmax(gamma * neg_efe + log_stable(self.E) - vfe_pi2)
            else:
                q_pi=nn.softmax(gamma * neg_efe + log_stable(self.E))
            q_pi_0=nn.softmax(gamma * neg_efe + log_stable(self.E))
            #print("Gerror@scan")
            #print((q_pi - q_pi_0).flatten())
            #print(neg_efe.flatten())
            #Gerror=jnp.dot((q_pi - q_pi_0), neg_efe)
            Gerror = jnp.broadcast_to(jnp.dot((q_pi - q_pi_0).flatten(), neg_efe.flatten()), (self.batch_size,))
            #print(Gerror)
            dFdg=qb-beta+Gerror
            #print(dFdg)
            qb=qb-dFdg/2
            gamma=1/qb
            return (q_pi, q_pi_0, gamma, Gerror, qb), None
        gamma=self.gamma
        q_pi=nn.softmax(gamma * neg_efe + log_stable(self.E))
        q_pi_0=nn.softmax(gamma * neg_efe + log_stable(self.E))
        #print("initialGerror")
        Gerror=jnp.broadcast_to(jnp.dot((q_pi - q_pi_0).flatten(), neg_efe.flatten()), (self.batch_size,))
        qb=jnp.broadcast_to(beta, (self.batch_size,))
        #print("scan")
        output, _ = lax.scan(scan_fn, (q_pi, q_pi_0, gamma, Gerror, qb), jnp.arange(self.num_iter))
        q_pi, q_pi_0, gamma, Gerror, qb = output
        #self.gamma=gamma
        agent = tree_at(lambda x: x.gamma, agent, gamma)
        return agent, q_pi, q_pi_0, gamma, Gerror
    

    def sample_action_policy_idx(self, q_pi: Array, rng_key=None):
        """
        Sample or select a discrete action from the posterior over control states.
        
        Returns
        ----------
        action: 1D ``jax.numpy.ndarray``
            Vector containing the indices of the actions for each control factor
        action_probs: 2D ``jax.numpy.ndarray``
            Array of action probabilities
        """

        if (rng_key is None) and (self.action_selection == "stochastic"):
            raise ValueError("Please provide a random number generator key to sample actions stochastically")

        if self.sampling_mode == "marginal":
            sample_action = partial(control.sample_action, self.policies, self.num_controls, action_selection=self.action_selection)
            action = vmap(sample_action)(q_pi, alpha=self.alpha, rng_key=rng_key)
        elif self.sampling_mode == "full":
            sample_policy = partial(control.sample_policy_idx, self.policies, action_selection=self.action_selection)
            action, policy_idx = vmap(sample_policy)(q_pi, alpha=self.alpha, rng_key=rng_key)

        return action, policy_idx
        """ def scan_fn(carry, iter):
            qs, err, vfe, kld, bs, un  = carry

            ln_qs = jtu.tree_map(log_stable, qs)
            # messages from future $m_+(s_t)$ and past $m_-(s_t)$ for all time steps and factors. For t = T we have that $m_+(s_T) = 0$
            
            lnB_future, lnB_past, lnB_future_for_kld = get_messages(ln_B, B, qs, ln_prior, B_dependencies)

            #mgds = jtu.Partial(mirror_gradient_descent_step, tau)
            mgds_vfe = jtu.Partial(mirror_gradient_descent_step_vfe_kld, tau)

            ln_As = vmap(all_marginal_log_likelihood, in_axes=(0, 0, None))(qs, log_likelihoods, A_dependencies)

            output = jtu.tree_map(mgds_vfe, ln_As, lnB_past, lnB_future, ln_qs, lnB_future_for_kld)
            qs, err, vfe, kld, bs, un = zip(*output)
            return (list(qs), list(err), list(vfe), list(kld), list(bs), list(un)), None
        err = qs
        vfe = qs
        kld = qs
        bs = qs
        un = qs
        output, _ = lax.scan(scan_fn, (qs, err, vfe, kld, bs, un), jnp.arange(num_iter)) """
    
    def infer_states_vfe_policies2(self, observations, empirical_prior, *, past_actions=None, qs_hist=None, mask=None):
        """
        Update approximate posterior over hidden states by solving variational inference problem, given an observation.

        Parameters
        ----------
        observations: ``list`` or ``tuple`` of ints
            The observation input. Each entry ``observation[m]`` stores one-hot vectors representing the observations for modality ``m``.
        past_actions: ``list`` or ``tuple`` of ints
            The action input. Each entry ``past_actions[f]`` stores indices (or one-hots?) representing the actions for control factor ``f``.
        empirical_prior: ``list`` or ``tuple`` of ``jax.numpy.ndarray`` of dtype object
            Empirical prior beliefs over hidden states. Depending on the inference algorithm chosen, the resulting ``empirical_prior`` variable may be a matrix (or list of matrices) 
            of additional dimensions to encode extra conditioning variables like timepoint and policy.
        Returns
        ---------
        qs: ``numpy.ndarray`` of dtype object
            Posterior beliefs over hidden states. Depending on the inference algorithm chosen, the resulting ``qs`` variable will have additional sub-structure to reflect whether
            beliefs are additionally conditioned on timepoint and policy.
            For example, in case the ``self.inference_algo == 'MMP' `` indexing structure is policy->timepoint-->factor, so that 
            ``qs[p_idx][t_idx][f_idx]`` refers to beliefs about marginal factor ``f_idx`` expected under policy ``p_idx`` 
            at timepoint ``t_idx``.
        """
        if not self.onehot_obs:
            o_vec = [nn.one_hot(o, self.num_obs[m]) for m, o in enumerate(observations)]
        else:
            o_vec = observations
        
        A = self.A
        if mask is not None:
            for i, m in enumerate(mask):
                o_vec[i] = m * o_vec[i] + (1 - m) * jnp.ones_like(o_vec[i]) / self.num_obs[i]
                A[i] = m * A[i] + (1 - m) * jnp.ones_like(A[i]) / self.num_obs[i]

        
        policies = self.policies
        batch_size = 1 # number of agents
        policies = jtu.tree_map(lambda x: jnp.broadcast_to(x, (batch_size,) + x.shape), policies)
        _,K, t,_= policies.shape
        #print(t)
        #print(past_actions.shape)
        """ selected_policy=-1
        if past_actions is not None:
            #print(past_actions.shape)
            if past_actions.shape[1]>=t:
                 #for i in range(K):
                    #print(past_actions[:,-t:])
                #print(past_actions[:,-t:][0])
                #print(policies[0][0])
                    #if jnp.array_equal(past_actions[:,-t:][0], policies[0][i]):
                        #selected_policy=i 
                selected_policy = jnp.where(
                    jnp.all(past_actions[:, -t:][0] == policies[0], axis=(1, 2)),
                    jnp.arange(K),
                    -1
                ).max()
                #print(selected_policy)
            else:
                selected_policy=-1
        else:
            selected_policy=-1 """
        
        infer_states_policies = partial(
            inference.update_posterior_states_vfe_policies2,
            A_dependencies=self.A_dependencies,
            B_dependencies=self.B_dependencies,
            num_iter=self.num_iter,
            method=self.inference_algo
        )
        
        output, err, vfe, kld, bs, un = vmap(infer_states_policies)(#vmap(infer_states)(
            A,
            self.B,
            o_vec,
            policies,
            past_actions,
            prior=empirical_prior,
            qs_hist=qs_hist
        )
        
        #print(len(output))
        #print(output[0].shape)
        #output = jnp.where(selected_policy != -1, jnp.array(jtu.tree_map(
        #    lambda x: x[0][selected_policy],output)[0]), jnp.array(output))
        #print(selected_policy)
        #output = jnp.where(selected_policy != -1, jnp.array(output[selected_policy]), jnp.array(output))#selected_policy
        """ if selected_policy != -1:
            output=output[selected_policy] """
        #print(output)
        ##実際に選択した行動についてのqsを出力する場合⇓
        """ if past_actions is not None:
            if past_actions.shape[1]>=t:
                #output=output[selected_policy]
                #print(jnp.array(output).shape)
                policy_mask = jnp.all(past_actions[:, -t:][0] == policies[0], axis=(1, 2))
                output = jnp.where(policy_mask[:, None, None, None, None], jnp.array(output), jnp.zeros_like(jnp.array(output)))
                output = output.sum(axis=0, keepdims=True)
                output = output.squeeze(axis=0)  # 不要な次元を削除
                #print(output.shape)
                #print(policy_mask)
        #output = jnp.where(selected_policy != -1, jnp.array(jtu.tree_map(lambda x: x[0],output)[selected_policy]), jnp.array(output))
        output=list(output) """
        #print(output)
        #print(len(output))
            #output=jtu.tree_map(lambda x: jnp.expand_dims(x, -1).astype(jnp.float32), output) #vfe=vfe[0].sum(2)
        """ vfe=jtu.tree_map(lambda x: x.sum(2),vfe)
        err=jtu.tree_map(lambda x: x.sum(2),err)
        kld=jtu.tree_map(lambda x: x.sum(2),kld)
        bs=jtu.tree_map(lambda x: x.sum(2),bs)
        un=jtu.tree_map(lambda x: x.sum(2),un)
        """
        #vfe=jtu.tree_map(lambda x: jtu.tree_map(lambda y: y.sum(2),x),vfe)
        """ err=jtu.tree_map(lambda x: jtu.tree_map(lambda y: y.sum(2),x),err)
        kld=jtu.tree_map(lambda x: jtu.tree_map(lambda y: y.sum(2),x),kld)
        bs=jtu.tree_map(lambda x: jtu.tree_map(lambda y: y.sum(2),x),bs)
        un=jtu.tree_map(lambda x: jtu.tree_map(lambda y: y.sum(2),x),un) """
        

        return output, err, vfe, kld, bs, un

    
    def infer_policies_detail(self, qs: List, alpha: List = None):
        """
        more detail breakdown of expected free energy than original infer_policies function
        with vector alpha that allow EFE balancing

        Returns
        ----------
        q_pi: 1D ``numpy.ndarray``
            Posterior beliefs over policies, i.e. a vector containing one posterior probability per policy.
        G: 1D ``numpy.ndarray``
            Negative expected free energies of each policy, i.e. a vector containing one negative expected free energy per policy.
        info: 1D ``dict``
            details of expected free energies, {"state_info_gain":info_gain, "utility":utility, "pA_info_gain":pA_info_gain, "pB_info_gain":pB_info_gain, "inductive_value":inductive_value}
            G(above) = info_gain + utility - (pA_info_gain + pB_info_gain) + inductive_value
        """

        latest_belief = jtu.tree_map(lambda x: x[:, -1], qs) # only get the posterior belief held at the current timepoint
        infer_policies = partial(
            control.update_posterior_policies_inductive_detail,
            self.policies,
            A_dependencies=self.A_dependencies,
            B_dependencies=self.B_dependencies,
            use_utility=self.use_utility,
            use_states_info_gain=self.use_states_info_gain,
            use_param_info_gain=self.use_param_info_gain,
            use_inductive=self.use_inductive
        )

        alpha = jnp.broadcast_to(alpha, (self.batch_size,)+alpha.shape) if alpha is not None else None
        q_pi, G, info = vmap(infer_policies)(
            latest_belief, 
            self.A,
            self.B,
            self.C,
            self.E,
            self.pA,
            self.pB,
            alpha_vec = alpha, # Alpha for EFE balancing
            I = self.I,
            gamma=self.gamma,
            inductive_epsilon=self.inductive_epsilon
        )

        return q_pi, G, info
    
    def infer_policies_efe_qs_pi_sub(self, qs_pi: List):
        """
        Perform policy inference by optimizing a posterior (categorical) distribution over policies.
        This distribution is computed as the softmax of ``G * gamma + lnE`` where ``G`` is the negative expected
        free energy of policies, ``gamma`` is a policy precision and ``lnE`` is the (log) prior probability of policies.
        This function returns the posterior over policies as well as the negative expected free energy of each policy.

        Returns
        ----------
        q_pi: 1D ``numpy.ndarray``
            Posterior beliefs over policies, i.e. a vector containing one posterior probability per policy.
        G: 1D ``numpy.ndarray``
            Negative expected free energies of each policy, i.e. a vector containing one negative expected free energy per policy.
        """
        #latest_belief = jtu.tree_map(lambda x: x[:, -1], qs) 
        policy_len=self.policy_len
        #print(qs_pi[0].shape)
        latest_belief = jtu.tree_map(lambda x: x[:, :,-1-policy_len:,:],qs_pi)#jtu.tree_map(lambda x: x[:, :,-1-policy_len,:],qs_pi)
        #print(latest_belief[0].shape)
        #latest_belief = jtu.tree_map(lambda y:jtu.tree_map(lambda x: x[:, -1-policy_len], y),qs_pi) # only get the posterior belief held at the current timepoint

        #latest_belief = jtu.tree_map(lambda y:jtu.tree_map(lambda x: x[:, -1], y),qs_pi)
        infer_policies = partial(
            control.update_posterior_policies_inductive_efe_qs_pi,
            self.policies,
            A_dependencies=self.A_dependencies,
            B_dependencies=self.B_dependencies,
            use_utility=self.use_utility,
            use_states_info_gain=self.use_states_info_gain,
            use_param_info_gain=self.use_param_info_gain,
            use_inductive=self.use_inductive
        )

        q_pi, G, PBS, PKLD, PFE, oRisk, PBS_pA, PBS_pB = vmap(infer_policies)(
            latest_belief, 
            self.A,
            self.B,
            self.C,
            self.E,
            self.pA,
            self.pB,
            I = self.I,
            gamma=self.gamma,
            inductive_epsilon=self.inductive_epsilon
        )

        return q_pi, G, PBS, PKLD, PFE, oRisk, PBS_pA, PBS_pB

    def infer_policies_efe_qs_pi(self, qs_pi: List,mode=1):
        """
        Perform policy inference by optimizing a posterior (categorical) distribution over policies.
        This distribution is computed as the softmax of ``G * gamma + lnE`` where ``G`` is the negative expected
        free energy of policies, ``gamma`` is a policy pision and ``lnE`` is the (log) prior probability of policies.
        This function returns the posterior over policies as well as the negative expected free energy of each policy.

        Returns
        ----------
        q_pi: 1D ``numpy.ndarray``
            Posterior beliefs over policies, i.e. a vector containing one posterior probability per policy.
        G: 1D ``numpy.ndarray``
            Negative expected free energies of each policy, i.e. a vector containing one negative expected free energy per policy.
        """
        #print(qs_pi[0].shape)
        #q_pi, G, PBS, PKLD, PFE, oRisk, PBS_pA, PBS_pB=jnp.where(len(qs_pi)==1,jnp.array(self.infer_policies_efe(qs_pi)),
                                                                 #jnp.array(self.infer_policies_efe_qs_pi_sub(qs_pi)))

        if len(qs_pi[0])==1:##len(qs_pi)==1だと一因子以外のとき，1時刻目をはじけない，
            q_pi, G, PBS, PKLD, PFE, oRisk, PBS_pA, PBS_pB,_,_=self.infer_policies_efe(qs_pi)
        elif mode==1:
            q_pi, G, PBS, PKLD, PFE, oRisk, PBS_pA, PBS_pB=self.infer_policies_efe_qs_pi_sub(qs_pi)##未来延長の場合？
        elif mode==2:
            q_pi, G, PBS, PKLD, PFE, oRisk, PBS_pA, PBS_pB=self.infer_policies_efe_qs_pi_sub2(qs_pi)##未来延長の場合？
        
        """ q_pi=list(q_pi)
        G=list(G)
        PBS=list(PBS)
        PKLD=list(PKLD)
        PFE=list(PFE)
        oRisk=list(oRisk)
        PBS_pA=list(PBS_pA)
        PBS_pB=list(PBS_pB) """
        return q_pi, G, PBS, PKLD, PFE, oRisk, PBS_pA, PBS_pB
    
    def infer_policies_precision2(self, neg_efe, vfe_pi, beta=1, reflect_len=None, gamma_update=True):
        agent=self
        if reflect_len is None:
            reflect_len=self.policy_len

        vfe_pi2=jnp.array(vfe_pi)
        #print(f"vfe_pi2:",vfe_pi2)
        def scan_fn(carry, iter):
            q_pi, q_pi_0, gamma, Gerror, qb=carry
            #print(vfe_pi[0].shape[0])
            vfe_pi2=jnp.array(vfe_pi)
            #print(f"vfe_pi2:",vfe_pi2.shape)
            #print(f"neg_efe:",neg_efe[0].shape)
            if vfe_pi2.shape[1]==neg_efe[0].shape[0]:
                #print("pi posterior")
                #print(vfe_pi[0][0,:,:])
                #print(reflect_len)
                #print(vfe_pi[0][:,:,-reflect_len:])
                
                ##vfe_pi2=vfe_pi2[:,:,:,-reflect_len:]##

                vfe_pi2 =jnp.sum(vfe_pi2, axis=(0,2,3))
                #vfe_pi2=jnp.sum(vfe_pi2, axis=-1).flatten()
                #print(vfe_pi2)
                q_pi=nn.softmax(gamma * neg_efe + log_stable(self.E) - vfe_pi2)
            else:
                q_pi=nn.softmax(gamma * neg_efe + log_stable(self.E))
            q_pi_0=nn.softmax(gamma * neg_efe + log_stable(self.E))
            #print("Gerror@scan")
            #print((q_pi - q_pi_0).flatten())
            #print(neg_efe.flatten())
            #Gerror=jnp.dot((q_pi - q_pi_0), neg_efe)
            Gerror = jnp.broadcast_to(jnp.dot((q_pi - q_pi_0).flatten(), neg_efe.flatten()), (self.batch_size,))
            #print(Gerror)
            dFdg=qb-beta+Gerror
            #print(dFdg)
            qb=qb-dFdg/2
            if gamma_update:
                gamma=1/qb
            else:
                gamma=self.gamma
            #gamma=1/qb
            return (q_pi, q_pi_0, gamma, Gerror, qb), None
        gamma=self.gamma
        q_pi=nn.softmax(gamma * neg_efe + log_stable(self.E))
        q_pi_0=nn.softmax(gamma * neg_efe + log_stable(self.E))
        #print("initialGerror")
        Gerror=jnp.broadcast_to(jnp.dot((q_pi - q_pi_0).flatten(), neg_efe.flatten()), (self.batch_size,))
        qb=jnp.broadcast_to(beta, (self.batch_size,))
        #print("scan")
        output, _ = lax.scan(scan_fn, (q_pi, q_pi_0, gamma, Gerror, qb), jnp.arange(self.num_iter))
        q_pi, q_pi_0, gamma, Gerror, qb = output
        #self.gamma=gamma
        if gamma_update:
            agent = tree_at(lambda x: x.gamma, agent, gamma)
        return agent, q_pi, q_pi_0, gamma, Gerror
    
    def calc_bayesian_model_averaging(self, qs_pi, q_pi):
        if len(qs_pi)>1:
            beliefs = jnp.array(qs_pi)
            Bayesian_model_avaraging_full_old = list(jnp.mean(beliefs, axis=0))
            #print(beliefs)
            #print(q_pi)
            Bayesian_model_avaraging_full=jnp.tensordot(q_pi, beliefs, axes=(1, 0))
            Bayesian_model_avaraging_full = list(Bayesian_model_avaraging_full[0])
            #print(Bayesian_model_avaraging_full_old)
            #print(Bayesian_model_avaraging_full)
            K, t,_= self.policies.shape
            Bayesian_model_avaraging = jtu.tree_map( lambda x: x[:, :-t], Bayesian_model_avaraging_full)
        else:
            Bayesian_model_avaraging=qs_pi
        return Bayesian_model_avaraging
    
    def calc_bayesian_model_averaging2(self, qs_pi, q_pi):
        #print(f"len(qs_pi): {len(qs_pi[0])}")
        #print(f"self.policies.shape[0]: {self.policies.shape[0]}")
        if len(qs_pi[0])==self.policies.shape[0]:##状態因子数＝ポリシー数だとエラー？
        #if len(qs_pi)>1:##状態因子が複数のとき用にfix必要
            ##beliefs = jnp.array(qs_pi)
            ##Bayesian_model_avaraging_full_old = list(jnp.mean(beliefs, axis=0))
            ##Bayesian_model_avaraging_full=jnp.tensordot(q_pi, beliefs, axes=(1, 0))
            ##Bayesian_model_avaraging_full = list(Bayesian_model_avaraging_full[0])
            #print(f"qs_pi: {qs_pi}")
            #print(f"q_pi: {q_pi}")
            q_pi=jnp.array(q_pi)
            beliefs = qs_pi
            #Bayesian_model_avaraging_full=jnp.zeros(beliefs[0].shape)
            Bayesian_model_avaraging_full=None
            """ for i in range(len(beliefs)):
                if Bayesian_model_avaraging_full is None:
                    Bayesian_model_avaraging_full=q_pi[i]*beliefs[i]
                else:
                    Bayesian_model_avaraging_full+=q_pi[i]*beliefs[i] """
            ##Bayesian_model_avaraging_full=jnp.tensordot(q_pi, beliefs, axes=(0, 0))
            #Bayesian_model_avaraging_full = list(Bayesian_model_avaraging_full[0])
            """ Bayesian_model_avaraging_full = jtu.tree_map(
            lambda beliefs_f: (
                print(f"beliefs_f shape: {beliefs_f.shape}, beliefs_f: {beliefs_f}"),
                jnp.tensordot(q_pi, beliefs_f, axes=(0, 0))
                )[1],  # printの結果を無視して、tensordotの結果のみを返す
                qs_pi
                ) """#1,2,100
            #Bayesian_model_avaraging_full=jnp.tensordot(q_pi, beliefs, axes=(0, 1))
            for i in range(len(beliefs)):
                beliefs[i]=jnp.array(beliefs[i])
            #beliefs=jtu.tree_map(lambda beliefs_f:jnp.array(beliefs_f),qs_pi)
            #print(f"beliefs: {beliefs}")
            Bayesian_model_avaraging_full=jtu.tree_map(lambda beliefs_f:jnp.tensordot(q_pi, beliefs_f, axes=(0, 0)),beliefs)
            #print(f"Bayesian_model_avaraging_full: {Bayesian_model_avaraging_full}")
            #print(Bayesian_model_avaraging_full_old)
            #print(Bayesian_model_avaraging_full)
            ##K, t,_= self.policies.shape
            #Bayesian_model_avaraging = jtu.tree_map( lambda x: x[:, :-t], Bayesian_model_avaraging_full)
            Bayesian_model_avaraging=Bayesian_model_avaraging_full
        else:
            Bayesian_model_avaraging=qs_pi
        return Bayesian_model_avaraging
    
    def calc_KLD_past_currentqs_pi(self, past_qs_pi, current_qs_pi):
        """
        認識によって減少する自由エネルギーを計算．mmpを想定．
        t:現在のタイムステップ
        
        past_beliefs:t-1におけるtまでの状態に関する信念
        current_beliefs:tにおけるtまでの状態に関する信念
        D_KL(past_beliefs|current_beliefs)を計算．
        """
        #past_beliefs = jtu.tree_map(lambda x, y: jnp.concatenate([x, y[None, ...]], axis=1), past_qs, empirical_prior)
        #past_beliefs = jtu.tree_map(lambda x, y: jnp.concatenate([x, y], axis=1), past_qs, empirical_prior)
        #past_beliefs = jtu.tree_map(lambda x, y: jnp.concatenate([x, y[:, None, :]], axis=1), past_qs, empirical_prior)
        K, t,_= self.policies.shape
        
            #print("combart")
        #empirical_prior = jtu.tree_map(lambda x: x[None, ...], empirical_prior) # 2次元配列を3次元配列に変換
            #print("combine")
        #past_beliefs = jtu.tree_map(lambda x, y: jnp.concatenate((x, y), axis=1), past_qs_pi, empirical_prior)
            #past_beliefs = jnp.concatenate((past_qs, empirical_prior), axis=1)
        #print(past_qs_pi[0])   
        #print(current_qs_pi[0].shape)
        if len(past_qs_pi[0].shape) ==2:
            past_beliefs = past_qs_pi
            #current_beliefs = current_qs_pi
        else:
            past_beliefs = past_qs_pi
            past_beliefs = jtu.tree_map( lambda x: x[:,:-t], past_qs_pi)
            #current_beliefs = jtu.tree_map( lambda x: x[0,:,:], current_qs_pi)
        #jtu.tree_map( lambda x: x[:, 1:-t], current_qs_pi)
        current_beliefs = current_qs_pi
        
        kld = inference.calc_KLD(past_beliefs, current_beliefs)
        return kld
    
    def compute_expected_state(self, action, qs):
        # return empirical_prior, and the history of posterior beliefs (filtering distributions) held about hidden states at times 1, 2 ... t

        # this computation of the predictive prior is correct only for fully factorised Bs.
        
        qs_last = jtu.tree_map( lambda x: x[:, -1], qs)
        propagate_beliefs = partial(control.compute_expected_state, B_dependencies=self.B_dependencies)
        pred = vmap(propagate_beliefs)(qs_last, self.B, action)
        
        return pred
    
    def infer_states2(self, observations, empirical_prior, *, past_actions=None, qs_hist=None, mask=None):
        """
        Update approximate posterior over hidden states by solving variational inference problem, given an observation.

        Parameters
        ----------
        observations: ``list`` or ``tuple`` of ints
            The observation input. Each entry ``observation[m]`` stores one-hot vectors representing the observations for modality ``m``.
        past_actions: ``list`` or ``tuple`` of ints
            The action input. Each entry ``past_actions[f]`` stores indices (or one-hots?) representing the actions for control factor ``f``.
        empirical_prior: ``list`` or ``tuple`` of ``jax.numpy.ndarray`` of dtype object
            Empirical prior beliefs over hidden states. Depending on the inference algorithm chosen, the resulting ``empirical_prior`` variable may be a matrix (or list of matrices) 
            of additional dimensions to encode extra conditioning variables like timepoint and policy.
        Returns
        ---------
        qs: ``numpy.ndarray`` of dtype object
            Posterior beliefs over hidden states. Depending on the inference algorithm chosen, the resulting ``qs`` variable will have additional sub-structure to reflect whether
            beliefs are additionally conditioned on timepoint and policy.
            For example, in case the ``self.inference_algo == 'MMP' `` indexing structure is policy->timepoint-->factor, so that 
            ``qs[p_idx][t_idx][f_idx]`` refers to beliefs about marginal factor ``f_idx`` expected under policy ``p_idx`` 
            at timepoint ``t_idx``.
        """
        if not self.onehot_obs:
            o_vec = [nn.one_hot(o, self.num_obs[m]) for m, o in enumerate(observations)]
        else:
            o_vec = observations
        
        A = self.A
        if mask is not None:
            for i, m in enumerate(mask):
                o_vec[i] = m * o_vec[i] + (1 - m) * jnp.ones_like(o_vec[i]) / self.num_obs[i]
                A[i] = m * A[i] + (1 - m) * jnp.ones_like(A[i]) / self.num_obs[i]

        infer_states = partial(
            inference.update_posterior_states2,
            A_dependencies=self.A_dependencies,
            B_dependencies=self.B_dependencies,
            num_iter=self.num_iter,
            method=self.inference_algo
        )
        
        #output = vmap(infer_states)(
        output, err = vmap(infer_states)(
            A,
            self.B,
            o_vec,
            past_actions,
            prior=empirical_prior,
            qs_hist=qs_hist
        )

        #return output

        # print(f'output : {output}')
        # print(f'err : {err}')

        output_last = jtu.tree_map( lambda x: x[:,-1], output)
        errs_last = jtu.tree_map( lambda x: x[:,-1], err)
        # print(f'output_last : {output_last}')
        # print(f'errs_last : {errs_last}')
        # print(f'output_last[0] : {output_last[0]}')
        # print(f'errs_last[0] : {errs_last[0]}')
        # print(f'output_last[0][0] : {output_last[0][0]}')
        # print(f'errs_last[0][0] : {errs_last[0][0]}')

        vfe = [-jnp.dot(output_last[0][0], errs_last[0][0])]

        # print(f'output[0][0].shape[0] : {output[0][0].shape[0]}')
        # vfes = []
        # for i in range(output[0][0].shape[0]):
        #     output_i = jtu.tree_map( lambda x: x[:,i], output)
        #     errs_i = jtu.tree_map( lambda x: x[:,i], err)
        #     vfes.append(-jnp.dot(output_i[0][0], errs_i[0][0]))

        return output, vfe#, vfes
    
    def infer_parameters_epsilon(self, beliefs_A, outcomes, actions, beliefs_B=None, lr_pA=1., lr_pB=1., epsilon=1e-6,**kwargs):
        agent = self
        beliefs_B = beliefs_A if beliefs_B is None else beliefs_B
        if self.inference_algo == 'ovf':
            smoothed_marginals_and_joints = vmap(inference.smoothing_ovf)(beliefs_A, self.B, actions)
            marginal_beliefs = smoothed_marginals_and_joints[0]
            joint_beliefs = smoothed_marginals_and_joints[1]
        else:
            marginal_beliefs = beliefs_A
            if self.learn_B:
                nf = len(beliefs_B)
                joint_fn = lambda f: [beliefs_B[f][:, 1:]] + [beliefs_B[f_idx][:, :-1] for f_idx in self.B_dependencies[f]]
                joint_beliefs = jtu.tree_map(joint_fn, list(range(nf)))

        if self.learn_A:
            update_A = partial(
                learning.update_obs_likelihood_dirichlet_epsilon,
                A_dependencies=self.A_dependencies,
                num_obs=self.num_obs,
                onehot_obs=self.onehot_obs,
            )
            
            lr = jnp.broadcast_to(lr_pA, (self.batch_size,))
            eps = jnp.broadcast_to(epsilon, (self.batch_size,))
            qA, E_qA = vmap(update_A)(
                self.pA,
                self.A,
                outcomes,
                marginal_beliefs,
                lr=lr,
                epsilon=eps
            )
            
            agent = tree_at(lambda x: (x.A, x.pA), agent, (E_qA, qA))
            
        if self.learn_B:
            assert beliefs_B[0].shape[1] == actions.shape[1] + 1
            update_B = partial(
                learning.update_state_transition_dirichlet,
                num_controls=self.num_controls
            )

            lr = jnp.broadcast_to(lr_pB, (self.batch_size,))
            qB, E_qB = vmap(update_B)(
                self.pB,
                joint_beliefs,
                actions,
                lr=lr
            )
            
            # if you have updated your beliefs about transitions, you need to re-compute the I matrix used for inductive inferenece
            if self.use_inductive and self.H is not None:
                I_updated = vmap(control.generate_I_matrix)(self.H, E_qB, self.inductive_threshold, self.inductive_depth)
            else:
                I_updated = self.I

            agent = tree_at(lambda x: (x.B, x.pB, x.I), agent, (E_qB, qB, I_updated))

        return agent
    
    def compute_expected_state_obs(self, action, qs):
        # return empirical_prior, and the history of posterior beliefs (filtering distributions) held about hidden states at times 1, 2 ... t

        # this computation of the predictive prior is correct only for fully factorised Bs.
        qs_last = jtu.tree_map( lambda x: x[:, -1], qs)
        propagate_beliefs = partial(control.compute_expected_state_obs, A_dependencies=self.A_dependencies,B_dependencies=self.B_dependencies)
        qs_pi, qo_pi = vmap(propagate_beliefs)(qs_last, self.A, self.B, action)
        
        return qs_pi, qo_pi
    
    def calc_KLD_prior_posterior(self, past_qs, current_qs):
        
        """ if past_qs is not None:
            
            empirical_prior = jtu.tree_map(lambda x: x[None, ...], empirical_prior) # 2次元配列を3次元配列に変換
            #print("combine")
            past_beliefs = jtu.tree_map(lambda x, y: jnp.concatenate((x, y), axis=1), past_qs, empirical_prior)
            #past_beliefs = jnp.concatenate((past_qs, empirical_prior), axis=1)
            #print(past_beliefs[0].shape)
            #print(current_qs[0].shape)
        else:
            past_beliefs = empirical_prior """
        
        kld = inference.calc_KLD(past_qs,current_qs)
        return kld
    
    def infer_policies_efe_qs_pi_sub2(self, qs_pi: List):
        """
        Perform policy inference by optimizing a posterior (categorical) distribution over policies.
        This distribution is computed as the softmax of ``G * gamma + lnE`` where ``G`` is the negative expected
        free energy of policies, ``gamma`` is a policy precision and ``lnE`` is the (log) prior probability of policies.
        This function returns the posterior over policies as well as the negative expected free energy of each policy.

        Returns
        ----------
        q_pi: 1D ``numpy.ndarray``
            Posterior beliefs over policies, i.e. a vector containing one posterior probability per policy.
        G: 1D ``numpy.ndarray``
            Negative expected free energies of each policy, i.e. a vector containing one negative expected free energy per policy.
        """
        #latest_belief = jtu.tree_map(lambda x: x[:, -1], qs) 
        policy_len=self.policy_len
        #print(qs_pi[0].shape)
        latest_belief = jtu.tree_map(lambda x: x[:, :,-1,:],qs_pi)#jtu.tree_map(lambda x: x[:, :,-1-policy_len,:],qs_pi)
        #print(latest_belief[0].shape)
        #latest_belief = jtu.tree_map(lambda y:jtu.tree_map(lambda x: x[:, -1-policy_len], y),qs_pi) # only get the posterior belief held at the current timepoint
        latest_belief = [list(p) for p in zip(*latest_belief)]
        #print(f"latest_belief: {latest_belief}")

        #latest_belief = jtu.tree_map(lambda y:jtu.tree_map(lambda x: x[:, -1], y),qs_pi)
        infer_policies = partial(
            control.update_posterior_policies_inductive_efe_qs_pi2,
            self.policies,
            A_dependencies=self.A_dependencies,
            B_dependencies=self.B_dependencies,
            use_utility=self.use_utility,
            use_states_info_gain=self.use_states_info_gain,
            use_param_info_gain=self.use_param_info_gain,
            use_inductive=self.use_inductive
        )

        q_pi, G, PBS, PKLD, PFE, oRisk, PBS_pA, PBS_pB = vmap(infer_policies)(
            latest_belief, 
            self.A,
            self.B,
            self.C,
            self.E,
            self.pA,
            self.pB,
            I = self.I,
            gamma=self.gamma,
            inductive_epsilon=self.inductive_epsilon
        )

        return q_pi, G, PBS, PKLD, PFE, oRisk, PBS_pA, PBS_pB
    
    def infer_states_vfe_set_prior(self, observations, empirical_prior, *, past_actions=None, qs_hist=None, mask=None, expected_states=None,tau=1.):#empirical_priorにもqs_histを入れる
        """
        Update approximate posterior over hidden states by solving variational inference problem, given an observation.

        Parameters
        ----------
        observations: ``list`` or ``tuple`` of ints
            The observation input. Each entry ``observation[m]`` stores one-hot vectors representing the observations for modality ``m``.
        past_actions: ``list`` or ``tuple`` of ints
            The action input. Each entry ``past_actions[f]`` stores indices (or one-hots?) representing the actions for control factor ``f``.
        empirical_prior: ``list`` or ``tuple`` of ``jax.numpy.ndarray`` of dtype object
            Empirical prior beliefs over hidden states. Depending on the inference algorithm chosen, the resulting ``empirical_prior`` variable may be a matrix (or list of matrices) 
            of additional dimensions to encode extra conditioning variables like timepoint and policy.
        Returns
        ---------
        qs: ``numpy.ndarray`` of dtype object
            Posterior beliefs over hidden states. Depending on the inference algorithm chosen, the resulting ``qs`` variable will have additional sub-structure to reflect whether
            beliefs are additionally conditioned on timepoint and policy.
            For example, in case the ``self.inference_algo == 'MMP' `` indexing structure is policy->timepoint-->factor, so that 
            ``qs[p_idx][t_idx][f_idx]`` refers to beliefs about marginal factor ``f_idx`` expected under policy ``p_idx`` 
            at timepoint ``t_idx``.
        """

        """ if expected_states is not None and qs_hist is not None:
            expected_states = jtu.tree_map(lambda x: x[None, ...], expected_states) # 2次元配列を3次元配列に変換
                #print("combine")
            past_beliefs = jtu.tree_map(lambda x, y: jnp.concatenate((x, y), axis=1), qs_hist, expected_states)
            #past_beliefs = jtu.tree_map(lambda x: x.squeeze(0), past_beliefs)  # 3次元配列を2次元配列に変換
        else:
            past_beliefs = empirical_prior """
        #print(past_beliefs[0].shape)
        if not self.onehot_obs:
            #print("convert to distribution")
            o_vec = [nn.one_hot(o, self.num_obs[m]) for m, o in enumerate(observations)]#観測値のワンホットベクトル化; One-hot vectorization of observed values
            #print(o_vec)
        else:
            o_vec = observations
        
        A = self.A
        if mask is not None:
            for i, m in enumerate(mask):
                o_vec[i] = m * o_vec[i] + (1 - m) * jnp.ones_like(o_vec[i]) / self.num_obs[i]
                A[i] = m * A[i] + (1 - m) * jnp.ones_like(A[i]) / self.num_obs[i]

        infer_states = partial(
            inference.update_posterior_states_vfe_set_prior,
            A_dependencies=self.A_dependencies,
            B_dependencies=self.B_dependencies,
            num_iter=self.num_iter,
            method=self.inference_algo,
            tau=tau
            
        )#並列処理のための関数の宣言;Declaring functions for parallel processing
        
        output, err, vfe, kld2, bs, un, qs_1step, err_1step, vfe_1step, kld2_1step, bs_1step, un_1step = vmap(infer_states)(  #output, err, vfe, kld, bs, un 
            A,
            self.B,
            o_vec,
            past_actions,
            prior=empirical_prior,
            qs_hist=qs_hist,
            expected_states=expected_states
        )#並列計算で認識分布（output）やvfeの計算;Parallel computation of recognition distribution (output) and vfe
        #vfe=vfe[0].sum(2)
        vfe=jtu.tree_map(lambda x: x.sum(2),vfe)#状態量の次元に沿ってVFEを足し上げ．;Add up VFE along the dimensions of the state variables.
        err=jtu.tree_map(lambda x: x.sum(2),err)
        kld2=jtu.tree_map(lambda x: x.sum(2),kld2)
        #kld=jtu.tree_map(lambda x: x.sum(2),kld)
        bs=jtu.tree_map(lambda x: x.sum(2),bs)
        un=jtu.tree_map(lambda x: x.sum(2),un)
        vfe_1step=jtu.tree_map(lambda x: x.sum(2),vfe_1step)
        err_1step=jtu.tree_map(lambda x: x.sum(2),err_1step)
        kld2_1step=jtu.tree_map(lambda x: x.sum(2),kld2_1step)
        bs_1step=jtu.tree_map(lambda x: x.sum(2),bs_1step)
        un_1step=jtu.tree_map(lambda x: x.sum(2),un_1step)

        return output, err, vfe, kld2, bs, un, qs_1step, err_1step, vfe_1step, kld2_1step, bs_1step, un_1step#output, err, vfe, kld(S_Hqs)(po), bs, un
    
    def calc_bayesian_model_averaging3(self, qs_pi, q_pi):
        #print(f"len(qs_pi): {len(qs_pi[0])}")
        #print(f"self.policies.shape[0]: {self.policies.shape[0]}")
        #if len(qs_pi[0])==q_pi.shape[0]:##状態因子数＝ポリシー数だとエラー？
        #if len(qs_pi)>1:##状態因子が複数のとき用にfix必要
            
        q_pi=jnp.array(q_pi)
        beliefs = qs_pi
        #Bayesian_model_avaraging_full=jnp.zeros(beliefs[0].shape)
        Bayesian_model_avaraging_full=None
        
        #Bayesian_model_avaraging_full=jnp.tensordot(q_pi, beliefs, axes=(0, 1))
        for i in range(len(beliefs)):
            beliefs[i]=jnp.array(beliefs[i])
       
        Bayesian_model_avaraging_full=jtu.tree_map(lambda beliefs_f:jnp.tensordot(q_pi, beliefs_f, axes=(0, 0)),beliefs)
        
        Bayesian_model_avaraging=Bayesian_model_avaraging_full
        #else:
            #Bayesian_model_avaraging=qs_pi
        return Bayesian_model_avaraging
    
    def infer_policies_efe_curiosity(self, qs: List,rng_key=None):
        """
        Perform policy inference by optimizing a posterior (categorical) distribution over policies.
        This distribution is computed as the softmax of ``G * gamma + lnE`` where ``G`` is the negative expected
        free energy of policies, ``gamma`` is a policy precision and ``lnE`` is the (log) prior probability of policies.
        This function returns the posterior over policies as well as the negative expected free energy of each policy.

        Returns
        ----------
        q_pi: 1D ``numpy.ndarray``
            Posterior beliefs over policies, i.e. a vector containing one posterior probability per policy.
        G: 1D ``numpy.ndarray``
            Negative expected free energies of each policy, i.e. a vector containing one negative expected free energy per policy.
        """

        latest_belief = jtu.tree_map(lambda x: x[:, -1], qs) # only get the posterior belief held at the current timepoint
        infer_policies = partial(
            control.update_posterior_policies_inductive_efe_curiosity,
            self.policies,
            A_dependencies=self.A_dependencies,
            B_dependencies=self.B_dependencies,
            use_utility=self.use_utility,
            use_states_info_gain=self.use_states_info_gain,
            use_param_info_gain=self.use_param_info_gain,
            use_inductive=self.use_inductive,
            rng_key=rng_key
        )

        q_pi, G, PBS,PBS_st, PKLD, PFE, oRisk, PBS_pA, PBS_pB,I_B_o,I_B_o_se = vmap(infer_policies)(
            latest_belief, 
            self.A,
            self.B,
            self.C,
            self.E,
            self.pA,
            self.pB,
            I = self.I,
            gamma=self.gamma,
            inductive_epsilon=self.inductive_epsilon
        )
        #print(PBS)
        return q_pi, G, PBS, PBS_st,PKLD, PFE, oRisk, PBS_pA, PBS_pB,I_B_o,I_B_o_se