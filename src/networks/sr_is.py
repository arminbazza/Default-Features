import numpy as np

from utils import create_transition_open, get_complete_DR


class Baseline_SR_IS:
    """
    SR-IS agent that operates solely from a transition matrix T.
    Compatible with the transition-matrix-based environment in test_baseline.py.

    Args:
        T (np.ndarray): Stochastic transition matrix (N x N)
        reward (float): Step reward (affects gamma = exp(reward / lambda))
        alpha (float): TD learning rate
        beta (float): Softmax temperature for policy
        _lambda (float): Reward scaling for LinearRL
        num_steps (int): Number of TD update steps
        policy (str): 'random' or 'softmax'
        imp_samp (bool): Use importance sampling corrections
        shape (tuple): Optional (rows, cols) to initialize DR from open-field T.
                       If None, initializes from identity.
    """
    def __init__(self, T, reward=-0.1, alpha=0.1, beta=1.0, _lambda=1.0,
                 num_steps=25000, policy="random", imp_samp=False, shape=None):
        self.T = T
        self.size = T.shape[0]
        self.shape = shape

        # Terminal states are self-loops (T[s, s] == 1)
        self.terminals = np.diag(self.T) == 1
        # P: transitions from non-terminal to terminal states
        self.P = self.T[~self.terminals][:, self.terminals]

        # Rewards
        self.reward_nt = reward
        self._lambda = _lambda
        self.r = np.full(self.size, reward)
        self.expr_t = np.exp(self.r[self.terminals] / _lambda)
        self.expr_nt = np.exp(reward / _lambda)

        # Parameters
        self.alpha = alpha
        self.beta = beta
        self.gamma = self.expr_nt
        self.num_steps = num_steps
        self.policy = policy
        self.imp_samp = imp_samp

        # State
        self.DR = self._init_DR()
        self.Z = np.full(self.size, 0.01)
        self.V = np.zeros(self.size)
        self.one_hot = np.eye(self.size)

    def _init_DR(self):
        """
        Initialize DR. If shape=(rows, cols) is provided, start from the
        open-field DR (D_os). Otherwise, initialize as near-identity.
        """
        if self.shape is not None:
            T_os = create_transition_open(shape=self.shape)
            return get_complete_DR(T=T_os)

        DR = np.full((self.size, self.size), 0.01)
        np.fill_diagonal(DR, 1.0)
        if self.terminals.any():
            term_idx = np.where(self.terminals)[0]
            DR[term_idx, term_idx] = 1.0 / (1.0 - self.gamma)
        return DR

    def _get_successors(self, state):
        """Return (successor_states, transition_probs) for reachable states."""
        row = self.T[state]
        mask = row > 0
        return np.where(mask)[0], row[mask]

    def update_V(self, term_states=None, term_reward=None):
        """
        Compute Z and V values.

        For environments with fixed terminal states (from T), uses self.terminals.
        Accepts optional term_states/term_reward to set terminal states on the fly
        (useful for planning toward an arbitrary goal).
        """
        if term_states is not None:
            expr_t = np.exp(term_reward / self._lambda)
            is_term = np.zeros(self.size, dtype=bool)
            is_term[np.asarray(term_states)] = True
            P = self.T[~is_term][:, is_term]
            n_term = int(is_term.sum())
            self.Z[~is_term] = self.DR[~is_term][:, ~is_term] @ P @ np.full(n_term, expr_t)
            self.Z[is_term] = expr_t
        elif self.terminals.any():
            self.Z[~self.terminals] = (
                self.DR[~self.terminals][:, ~self.terminals] @ self.P @ self.expr_t
            )
            self.Z[self.terminals] = self.expr_t

        self.V = np.log(np.maximum(self.Z, 1e-20)) * self._lambda

    def importance_sampling(self, state, s_prob):
        """
        Compute IS weight P(x'|x) / u(x'|x).
        P(x'|x) is the uniform random-walk policy (1 / num_successors).
        u(x'|x) = s_prob is the probability under the decision policy.
        """
        successors, _ = self._get_successors(state)
        p_uniform = 1.0 / len(successors)
        return p_uniform / s_prob

    def select_action(self, state):
        """
        Select next state according to policy.
        Returns (next_state_idx, selection_probability).
        """
        successors, probs = self._get_successors(state)

        if len(successors) == 0:
            return state, 1.0

        if self.policy == "random":
            idx = np.random.choice(len(successors), p=probs)
            return int(successors[idx]), float(probs[idx])

        elif self.policy == "softmax":
            log_z = np.log(np.maximum(self.Z[successors], 1e-20)) / self.beta
            log_z -= log_z.max()
            exp_z = np.exp(log_z)
            action_probs = exp_z / exp_z.sum()

            if np.any(np.isnan(action_probs)) or action_probs.sum() == 0:
                action_probs = np.ones(len(successors)) / len(successors)

            idx = np.random.choice(len(successors), p=action_probs)
            return int(successors[idx]), float(action_probs[idx])

        # Fallback
        idx = np.random.choice(len(successors), p=probs)
        return int(successors[idx]), float(probs[idx])

    def take_action(self, state_idx):
        """
        Select and execute an action from state_idx.
        Returns (next_state_idx, IS_weight, done).
        """
        next_state_idx, s_prob = self.select_action(state_idx)

        if self.imp_samp:
            w = self.importance_sampling(state_idx, s_prob)
            w = 1.0 if (np.isnan(w) or w == 0) else w
        else:
            w = 1.0

        done = bool(self.terminals[next_state_idx])
        return next_state_idx, w, done

    def update(self, state_idx, next_state_idx, w, step):
        """TD update for one transition (state_idx -> next_state_idx)."""
        target = self.one_hot[state_idx] + self.gamma * self.DR[next_state_idx]
        self.DR[state_idx] = (1 - self.alpha) * self.DR[state_idx] + self.alpha * target * w

        if self.terminals.any():
            self.Z = self.DR[:, ~self.terminals] @ self.P @ self.expr_t

    def _sample_start(self):
        """Sample a random non-terminal start state."""
        valid = np.where(~self.terminals)[0] if self.terminals.any() else np.arange(self.size)
        return int(np.random.choice(valid))

    def learn(self, seed=None):
        """
        Learn DR via TD updates using transitions sampled from T.
        Resets to a random valid state upon reaching a terminal.
        """
        if seed is not None:
            np.random.seed(seed)

        state_idx = self._sample_start()

        for i in range(self.num_steps):
            next_state_idx, w, done = self.take_action(state_idx)
            self.update(state_idx, next_state_idx, w, i)

            state_idx = self._sample_start() if done else next_state_idx

        self.update_V()
