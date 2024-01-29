import copy
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier as RandomForest
from obp.policy import Random
from obp.dataset import logistic_sparse_reward_function
from obp.simulator.simulator import BanditEnvironmentSimulator, BanditPolicySimulator
from obp.policy.offline import IPWLearner
n_rounds = 1000
n_runs_per_round_size = 20
n_actions = 5
dim_context = 5

policy_class = Random
policy_args = {"n_actions": n_actions, "epsilon": 1.0, "random_state": 12345}

train_rewards = {policy_class.__name__: []}
eval_rewards = {**copy.deepcopy(train_rewards), **{IPWLearner.__name__: []}}
train_rewards["n_rounds"] = []
eval_rewards["n_rounds"] = []

env = BanditEnvironmentSimulator(
    n_actions=10,
    dim_context=5,
    reward_type="binary",  # "binary" or "continuous"
    reward_function=logistic_sparse_reward_function,
    random_state=12345,
)

for experiment in range(n_runs_per_round_size):
    training_bandit_batch = env.next_bandit_round_batch(n_rounds=n_rounds)
    evaluation_bandit_batch = env.next_bandit_round_batch(n_rounds=n_rounds)

    # Train the bandit algorithm (Random policy) and get the rewards for the training and evaluation period
    policy = policy_class(**policy_args)

    training_simulator = BanditPolicySimulator(policy=policy)
    training_simulator.steps(batch_bandit_rounds=training_bandit_batch)
    train_rewards[policy_class.__name__].append(training_simulator.total_reward)

    eval_simulator = BanditPolicySimulator(policy=policy)
    eval_simulator.steps(batch_bandit_rounds=evaluation_bandit_batch)
    eval_rewards[policy_class.__name__].append(eval_simulator.total_reward)

    # Train a propensity model on the actions in the training period to get propensities per round
    propensity_model = LogisticRegression(random_state=12345)
    propensity_model.fit(training_simulator.contexts, training_simulator.selected_actions)
    pscores = propensity_model.predict_proba(training_simulator.contexts)

    # Train an IPW learning from the logged data and learned propensities
    ipw_learner = IPWLearner(n_actions=env.n_actions,
                             base_classifier=RandomForest(n_estimators=30, min_samples_leaf=10, random_state=12345))

    ipw_learner.fit(
        context=training_simulator.contexts,
        action=training_simulator.selected_actions,
        reward=training_simulator.obtained_rewards,
        pscore=np.choose(training_simulator.selected_actions, pscores.T)
    )
    eval_action_dists = ipw_learner.predict(
        context=eval_simulator.contexts
    )

    eval_rewards[ipw_learner.policy_name].append(
        np.sum(eval_action_dists.squeeze(axis=2) * evaluation_bandit_batch.rewards)
    )

    train_rewards["n_rounds"].append(n_rounds)
    eval_rewards["n_rounds"].append(n_rounds)
