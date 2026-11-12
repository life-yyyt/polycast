from polycast.simulation import run_simulation


def test_simulation_is_reproducible_for_a_given_seed():
    a = run_simulation(n_agents=25, rounds=40, seed=11)
    b = run_simulation(n_agents=25, rounds=40, seed=11)
    assert a.mean_series == b.mean_series
    assert a.final_spread == b.final_spread


def test_different_seeds_diverge():
    a = run_simulation(n_agents=25, rounds=40, seed=1)
    b = run_simulation(n_agents=25, rounds=40, seed=2)
    assert a.mean_series != b.mean_series


def test_beliefs_converge_toward_the_truth():
    result = run_simulation(n_agents=60, rounds=120, truth=0.7, seed=5)
    assert result.final_spread < result.spread_series[0]
    assert abs(result.final_mean - result.truth) < 0.15


def test_social_pressure_speeds_up_consensus():
    isolated = run_simulation(n_agents=40, rounds=60, social_weight=0.05, seed=9)
    networked = run_simulation(n_agents=40, rounds=60, social_weight=0.9, seed=9)
    assert networked.final_spread < isolated.final_spread


def test_social_pressure_can_lock_in_the_wrong_consensus():
    # Faster agreement is not automatically better: a crowd that leans on its
    # neighbours converges on their answer instead of on the signal.
    isolated = run_simulation(n_agents=40, rounds=60, social_weight=0.05, seed=9)
    networked = run_simulation(n_agents=40, rounds=60, social_weight=0.9, seed=9)
    assert abs(networked.final_mean - networked.truth) > abs(
        isolated.final_mean - isolated.truth
    )


def test_ring_topology_runs():
    result = run_simulation(n_agents=10, rounds=20, topology="ring", seed=3)
    assert len(result.mean_series) == 20
