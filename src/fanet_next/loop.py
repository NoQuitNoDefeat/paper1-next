"""Decision loop wiring: scenario + backend + candidates + constraints + observation + reward.

``SchedulingEnv`` is the only place where the components meet.  Each module
only sees its own contract: the policy gets a :class:`DecisionInput`, the
backend a :class:`Plan`, the reward a :class:`CycleFacts`.
"""

from __future__ import annotations

import time
from dataclasses import dataclass

import numpy as np

from .backend.base import Backend
from .contracts import CycleFacts, EndType, Plan
from .observation.builder import ObservationBuilder
from .policy.base import DecisionInput, Policy
from .reward.metrics import MetricsAccumulator
from .reward.standard import RewardBreakdown
from .scheduling.candidates import make_problem
from .scheduling.constraints import ConstraintSet, full_sinr_violations


@dataclass
class Transition:
    facts: CycleFacts
    reward: RewardBreakdown
    end: EndType
    next_input: DecisionInput  # last valid state even when the episode ended


class SchedulingEnv:
    def __init__(self, *, scenario_source, backend: Backend, candidates, constraints: ConstraintSet,
                 observation: ObservationBuilder, reward, build_graph: bool = True,
                 run_id: str = "run"):
        self.scenario_source = scenario_source
        self.backend = backend
        self.candidates = candidates
        self.constraints = constraints
        self.observation = observation
        self.reward = reward
        self.build_graph = build_graph
        self.run_id = run_id
        self.metrics = MetricsAccumulator()
        self.current: DecisionInput | None = None
        self.seed: int | None = None

    def _input(self, report) -> DecisionInput:
        self.observation.observe(report)
        problem = make_problem(report, self.candidates.select(report))
        graph = self.observation.build(report, problem) if self.build_graph else None
        return DecisionInput(report=report, problem=problem,
                             controller=self.constraints.start(problem), graph=graph)

    def reset(self, seed: int, episode: int = 0) -> DecisionInput:
        self.seed = seed
        self.scenario = self.scenario_source.make(seed)
        self.observation.reset()
        self.metrics = MetricsAccumulator()
        report = self.backend.reset(self.scenario, run_id=self.run_id, episode=episode,
                                    seed=seed + 7_919)
        self.current = self._input(report)
        return self.current

    def step(self, actions: list[int], decision_seconds: float = 0.0) -> Transition:
        inp = self.current
        links = inp.problem.links[np.asarray(actions, dtype=np.int64)] if actions else np.zeros((0, 2))
        plan = Plan(cycle=inp.report.cycle, links=tuple((int(a), int(b)) for a, b in links))
        outcome = self.backend.execute(plan)
        reward = self.reward(outcome.facts)
        p = inp.problem
        viol = full_sinr_violations(np.asarray(plan.links).reshape(-1, 2), p.gain, p.power,
                                    p.noise, p.threshold)
        self.metrics.add(outcome.facts, reward, n_candidates=p.num_candidates,
                         decision_seconds=decision_seconds, planned_sinr_violations=viol)
        self.current = self._input(outcome.next_report)
        return Transition(outcome.facts, reward, outcome.end, self.current)


def run_episodes(policy: Policy, envs: list[SchedulingEnv], seeds: list[int], *,
                 mode: str = "greedy") -> list[dict]:
    """Run one episode per seed, stepping ``len(envs)`` environments in lockstep."""
    summaries: list[dict] = []
    queue = list(enumerate(seeds))
    active: dict[int, int] = {}  # env slot -> seed index
    for slot_id, env in enumerate(envs):
        if queue:
            i, s = queue.pop(0)
            env.reset(s, episode=i)
            active[slot_id] = i
    results: dict[int, dict] = {}
    while active:
        slots = sorted(active)
        inputs = [envs[s].current for s in slots]
        t0 = time.perf_counter()
        outs = policy.act(inputs, mode=mode)
        dt = (time.perf_counter() - t0) / max(len(slots), 1)
        for s, out in zip(slots, outs):
            tr = envs[s].step(out.actions, decision_seconds=dt)
            if tr.end is not EndType.CONTINUE:
                summary = envs[s].metrics.summary()
                summary["seed"] = seeds[active[s]]
                results[active[s]] = summary
                del active[s]
                if queue:
                    i, sd = queue.pop(0)
                    envs[s].reset(sd, episode=i)
                    active[s] = i
    for i in range(len(seeds)):
        summaries.append(results[i])
    return summaries
