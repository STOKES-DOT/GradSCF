"""Resource planning and reward learning on small, deterministic inputs."""
from gradscf.model.autoxc.action_policy import ActionPolicy
from gradscf.model.autoxc.workflows.xc100.schedule import affinity_plan, distribute


def test_v100_cpu_partition_and_disjoint_sample_coverage():
    plan = affinity_plan(range(32))
    assert list(map(len, plan)) == [8, 4, 4, 4, 4, 4, 4]
    assert sorted(cpu for group in plan for cpu in group) == list(range(32))
    costs = {str(i): [i / 5, *([i] * 6)] for i in range(1, 101)}
    groups = distribute(costs, len(plan))
    assert sorted(sample for group in groups for sample in group) == sorted(costs)
    assert all(groups)


def test_policy_credits_improvements_and_counts_unsuccessful_attempts():
    policy = ActionPolicy(seed=7, epsilon=0)
    actions = policy.actions
    rows = [
        {'decision': {'state': 'improving', 'action': a}, 'policy_reward': reward}
        for a, reward in zip(actions, [0.006, 0.002, 0., 0.])
    ]
    rows.insert(1, {'decision': {'state': 'improving', 'action': actions[0]},
                    'policy_reward': 0.})
    decision = policy.choose(6, rows)
    assert decision['action'] == actions[0]
    assert decision['estimates'][actions[0]] == {'count': 2, 'mean': 0.003}
    assert decision == policy.choose(6, rows)


def test_fresh_search_starts_with_a_real_design_instead_of_ablation():
    decision = ActionPolicy().choose(1, [])
    assert decision['action'] == 'physical_exploration'
    assert decision['probability'] == 1
    assert decision['related_rounds'] == []


def test_cpu_rank_environment_is_fixed_before_process_imports(monkeypatch):
    from gradscf.model.autoxc.workflows.xc100.parallel import rank_environment

    monkeypatch.setenv('JAX_PLATFORMS', 'cuda')
    monkeypatch.setenv('CUDA_VISIBLE_DEVICES', '0')
    env = rank_environment((3, 4, 5, 6), gpu=False)
    assert env['JAX_PLATFORMS'] == 'cpu'
    assert env['CUDA_VISIBLE_DEVICES'] == ''
    assert env['OMP_NUM_THREADS'] == env['OPENBLAS_NUM_THREADS'] == '4'
    assert rank_environment((0, 1), gpu=True)['CUDA_VISIBLE_DEVICES'] == '0'
