"""Python lightweight backend: FIFO next-hop queues, waiting areas, SINR execution.

Per-cycle order (research-method §6):

1. execute the plan over the service window: successful links pop whole FIFO
   head packets within the byte budget; relay arrivals and new births are
   *staged*, not admitted;
2. sample the post-service queue snapshot at the cycle end;
3. terminations due at the cycle end (deadline, retry limit, waiting timeout)
   — expiry is processed before route recovery;
4. routing update (every ``routing.update_every`` cycles); with
   ``stale_queue_policy="rehome"`` queued packets whose next hop changed are
   re-admitted at their node, with ``"keep"`` they stay where they are;
5. admission in (node arrival time, packet id) order: waiting packets that got
   a route, re-homed packets, relay arrivals and births.  Full queues reject
   new packets (``queue_overflow``); packets without a route enter the waiting
   area or are rejected (``waiting_overflow``).  A waiting packet whose target
   queue is full stays in the waiting area.

Enqueue time is the admission time into the current queue; the HOL wait of a
queue is ``sample time - enqueue time`` of its current head.
"""

from __future__ import annotations

from collections import defaultdict, deque

import numpy as np

from ..contracts import (CycleFacts, EndType, ExecutionError, Plan, QueueSnapshot, Report,
                         StepOutcome)
from ..physics import db_to_lin, set_sinr
from ..scenario.base import Scenario
from ..scenario.channel import CHANNEL, ChannelModel
from ..scenario.routing import ROUTING, Routing
from .base import BACKEND, Backend

EPS = 1e-9


class Packet:
    __slots__ = ("pid", "src", "dst", "size", "born", "node", "node_arrived", "enqueued",
                 "waiting_since", "retries", "hops0")

    def __init__(self, pid, src, dst, size, born):
        self.pid, self.src, self.dst, self.size, self.born = pid, src, dst, size, born
        self.node, self.node_arrived = src, born
        self.enqueued = self.waiting_since = None
        self.retries = 0
        self.hops0 = -1  # route length at birth (-1: no route), for per-hop-class metrics


@BACKEND.register("lightweight", role="primary")
class LightweightBackend(Backend):
    """Python packet-level backend with cumulative-SINR execution."""

    supports_state = True
    # class-level defaults: environments pickled by older code (resumed training) keep working
    service_order, room_aware_service, buffer, route_stagger = "fifo", False, "per_queue", False
    node_buffer_packets, rehome_overflow = None, "drop"

    def __init__(self, channel: dict | str = "ideal", routing: dict | str = "min_hop",
                 stale_queue_policy: str = "rehome", deadline_s: float | None = None,
                 retry_limit: int | None = None, terminate_when_drained: bool = False,
                 check_conservation: bool = True, service_order: str = "fifo",
                 room_aware_service: bool = False, buffer: str = "per_queue",
                 node_buffer_packets: int | None = None, route_stagger: bool = False,
                 rehome_overflow: str = "drop"):
        """Stage-0 diagnostic options (defaults = the method's environment):
        ``service_order`` "fifo" | "fewest_hops" (a scheduled link sends its packets with the fewest
        remaining hops from the receiver first, FIFO within a class); ``room_aware_service`` (a
        packet is sent only if the receiver has room for it at service time: destined to the
        receiver, or room in its next queue there, counting slots claimed this cycle; claimed
        packets are admitted first at the cycle end; a route update after service can still
        redirect them); ``buffer`` "per_queue" (each next-hop queue has its own limit) | "shared"
        (a node's queues share ``node_buffer_packets``, fixed for the episode; each queue then
        reports packets + the node's free room as its capacity, so occupancy stays in [0, 1]);
        ``route_stagger`` (route updates staggered by destination: each cycle the routes towards
        one group of destinations are recomputed for all nodes from the same snapshot, so every
        destination tree stays loop-free); ``rehome_overflow`` "drop" | "exceed" (a packet moved
        to its new next-hop queue after a route update enters it even when the queue is full,
        isolating the loss mechanism of re-homing)."""
        if stale_queue_policy not in {"rehome", "keep"}:
            raise ValueError("stale_queue_policy must be 'rehome' or 'keep'")
        if service_order not in {"fifo", "fewest_hops"}:
            raise ValueError("service_order must be 'fifo' or 'fewest_hops'")
        if buffer not in {"per_queue", "shared"}:
            raise ValueError("buffer must be 'per_queue' or 'shared'")
        if buffer == "shared" and not node_buffer_packets:
            raise ValueError("buffer 'shared' needs node_buffer_packets")
        if rehome_overflow not in {"drop", "exceed"}:
            raise ValueError("rehome_overflow must be 'drop' or 'exceed'")
        self.service_order, self.room_aware_service = service_order, bool(room_aware_service)
        self.buffer, self.route_stagger = buffer, bool(route_stagger)
        self.node_buffer_packets = int(node_buffer_packets) if node_buffer_packets else None
        self.rehome_overflow = rehome_overflow
        self.channel: ChannelModel = CHANNEL.build(channel)
        self.routing: Routing = ROUTING.build(routing)
        self.stale_queue_policy = stale_queue_policy
        self.deadline_s = deadline_s
        self.retry_limit = retry_limit
        self.terminate_when_drained = terminate_when_drained
        self.check_conservation = check_conservation

    # ------------------------------------------------------------------ setup
    def reset(self, scenario: Scenario, *, run_id: str = "run", episode: int = 0,
              seed: int = 0) -> Report:
        sc = self.sc = scenario
        radio = sc.radio
        if radio.service_window_s > sc.cycle_length + EPS:
            raise ValueError("service window longer than the cycle")
        self.run_id, self.episode = run_id, episode
        self.rng = np.random.default_rng(seed)
        n = sc.num_nodes
        self.channel.reset(n, self.rng)
        self.power = np.full(n, radio.tx_power_w)
        self.noise = radio.noise_w
        self.threshold = radio.threshold
        # the scheduler plans against the decoding threshold plus the channel's fade margin
        self.plan_threshold = radio.threshold * float(db_to_lin(self.channel.fade_margin_db))
        self.qlinks = np.asarray(sc.queue_links, dtype=np.int64)
        self.qindex = {(int(u), int(v)): q for q, (u, v) in enumerate(self.qlinks)}
        self.qcap = np.asarray(sc.queue_capacity, dtype=np.int64)
        self.queues: list[deque[Packet]] = [deque() for _ in range(len(self.qlinks))]
        self.node_queues = [np.nonzero(self.qlinks[:, 0] == u)[0] for u in range(n)]
        self.waiting: list[list[Packet]] = [[] for _ in range(n)]
        self.wcap = np.asarray(sc.waiting_capacity, dtype=np.int64)
        self.cycle = 0
        self.next_pid = 0
        self.totals = defaultdict(int)
        self.next_hop = self._compute_routes(0)
        self.report = self._make_report()
        return self.report

    def _compute_routes(self, cycle: int) -> np.ndarray:
        gain = self.channel.true_gain(self.sc.positions(cycle), self.sc.radio)
        ratio = self.power[:, None] * gain / (self.threshold * self.noise)
        # routes only use links the scheduler can use: SNR >= planning threshold (decoding
        # threshold x fade margin); the routing margin stays relative to decoding
        adjacency = ratio >= self.plan_threshold / self.threshold * (1 - EPS)
        np.fill_diagonal(adjacency, False)
        return self.routing.compute(adjacency, ratio)

    # --------------------------------------------------- routes and buffer room
    def _route_len(self, a: int, d: int) -> int:
        """Hops from node ``a`` to ``d`` along the current next-hop table (0 at d, -1: no route)."""
        x, h, n = a, 0, self.sc.num_nodes
        while x != d and h < n:
            nh = int(self.next_hop[x, d])
            if nh < 0:
                return -1
            x, h = nh, h + 1
        return h if x == d else -1

    def _node_cap(self, u: int) -> int:
        return int(self.node_buffer_packets)

    def _node_total(self, u: int) -> int:
        if not hasattr(self, "node_queues"):  # state from older code
            self.node_queues = [np.nonzero(self.qlinks[:, 0] == v)[0] for v in range(self.sc.num_nodes)]
        return sum(len(self.queues[q]) for q in self.node_queues[u])

    def _room(self, node: int, q: int) -> tuple[tuple, int]:
        """(claim key, free slots) for admitting a packet into queue ``q`` at ``node``."""
        if self.buffer == "shared":
            return ("n", node), self._node_cap(node) - self._node_total(node)
        return ("q", q), int(self.qcap[q]) - len(self.queues[q])

    def _admissible_now(self, p: Packet, rx: int, claims: dict) -> bool:
        if p.dst == rx:
            return True
        nh = int(self.next_hop[rx, p.dst])
        if nh < 0:
            key, free = ("w", rx), int(self.wcap[rx]) - len(self.waiting[rx])
        else:
            key, free = self._room(rx, self.qindex[(rx, nh)])
        if free - claims.get(key, 0) > 0:
            claims[key] = claims.get(key, 0) + 1
            return True
        return False

    def _select_service(self, queue: deque, rx: int, budget: int, claims: dict) -> list[Packet]:
        """Packets a successful link sends under the diagnostic service options; the rest keep
        their order (so the head is still the oldest packet)."""
        cands = list(queue)
        order = range(len(cands))
        if self.service_order == "fewest_hops":
            rem = {}
            def key(i):
                d = cands[i].dst
                if d not in rem:
                    r = 0 if d == rx else self._route_len(rx, d)
                    rem[d] = r if r >= 0 else self.sc.num_nodes + 1
                return (rem[d], i)
            order = sorted(order, key=key)
        picked = []
        for i in order:
            p = cands[i]
            if p.size > budget:
                break
            if self.room_aware_service:
                if not self._admissible_now(p, rx, claims):
                    continue
                claims.setdefault("pids", set()).add(p.pid)
            picked.append(i)
            budget -= p.size
        if picked:
            gone = set(picked)
            queue.clear()
            queue.extend(cands[i] for i in range(len(cands)) if i not in gone)
        return [cands[i] for i in picked]

    # ---------------------------------------------------------------- reports
    def _snapshot(self, t: float) -> QueueSnapshot:
        q = len(self.queues)
        packets = np.fromiter((len(d) for d in self.queues), dtype=np.int64, count=q)
        nbytes = np.fromiter((sum(p.size for p in d) for d in self.queues), dtype=np.int64, count=q)
        hol = np.fromiter((t - d[0].enqueued if d else 0.0 for d in self.queues), dtype=float, count=q)
        cap = self.qcap.copy()
        if self.buffer == "shared":  # each queue may still grow by the node's free room
            free = {u: max(self._node_cap(u) - self._node_total(u), 0) for u in range(self.sc.num_nodes)}
            cap = packets + np.array([free[int(u)] for u in self.qlinks[:, 0]], dtype=np.int64)
        return QueueSnapshot(packets, nbytes, cap, np.maximum(hol, 0.0))

    def _make_report(self) -> Report:
        sc, k = self.sc, self.cycle
        t = k * sc.cycle_length
        pos = sc.positions(k)
        u, v = self.qlinks[:, 0], self.qlinks[:, 1]
        route_next = (self.next_hop[u] == v[:, None]).any(axis=1)
        wait_n = np.array([len(w) for w in self.waiting], dtype=np.int64)
        wait_old = np.array([t - min(p.waiting_since for p in w) if w else 0.0 for w in self.waiting])
        queue_dst = np.zeros((len(self.queues), sc.num_nodes), dtype=np.int64)
        for i, d in enumerate(self.queues):
            for p in d:
                queue_dst[i, p.dst] += 1
        return Report(
            run_id=self.run_id, episode=self.episode, cycle=k, time=t,
            cycle_length=sc.cycle_length, num_nodes=sc.num_nodes, positions=pos,
            velocities=sc.velocities(k), gain=self.channel.estimate(pos, sc.radio, self.rng),
            tx_power=self.power.copy(), noise=self.noise, threshold=self.plan_threshold,
            service_bytes=sc.radio.service_bytes, packet_size=sc.packet_size,
            queue_links=self.qlinks.copy(), queues=self._snapshot(t), route_next=route_next,
            next_hop=self.next_hop.copy(), waiting_packets=wait_n,
            waiting_capacity=self.wcap.copy(), waiting_oldest=np.maximum(wait_old, 0.0),
            waiting_max_wait=sc.waiting_max_wait, queue_dst=queue_dst)

    def in_system(self) -> tuple[int, int]:
        return sum(len(d) for d in self.queues), sum(len(w) for w in self.waiting)

    # -------------------------------------------------------------- execution
    def _validate(self, plan: Plan) -> list[tuple[int, int]]:
        if plan.cycle != self.cycle:
            raise ExecutionError(f"plan for cycle {plan.cycle} but backend is at {self.cycle}")
        links = [(int(a), int(b)) for a, b in plan.links]
        used: set[int] = set()
        for link in links:
            if link not in self.qindex:
                raise ExecutionError(f"plan link {link} is not a registered queue")
            if link[0] in used or link[1] in used:
                raise ExecutionError(f"plan uses node of {link} twice (single radio, half duplex)")
            used.update(link)
        return links

    def execute(self, plan: Plan) -> StepOutcome:
        links = self._validate(plan)
        sc, radio, k = self.sc, self.sc.radio, self.cycle
        t0, t1 = k * sc.cycle_length, (k + 1) * sc.cycle_length
        queued0, waiting0 = self.in_system()
        nq = len(self.queues)
        served_p = np.zeros(nq, dtype=np.int64)
        served_b = np.zeros(nq, dtype=np.int64)
        facts_delivered, facts_delays, delivered_hops0 = [], [], []
        delivered_bytes = 0
        staged: list[Packet] = []

        # 1. service over the window, actual channel
        exec_gain = self.channel.execution(sc.positions(k), sc.velocities(k), radio, self.rng)
        sinr = set_sinr(np.array(links, dtype=np.int64).reshape(-1, 2), exec_gain, self.power,
                        self.noise)
        ok = sinr >= self.threshold * (1 - EPS)
        succeeded, failed = [], []
        claims: dict = {}  # receiver slots claimed this cycle (room_aware_service)
        byte_time = 8.0 / radio.rate_bps
        for (tx, rx), good in zip(links, ok):
            q = self.qindex[(tx, rx)]
            queue = self.queues[q]
            if not good:
                failed.append((tx, rx))
                if queue:
                    queue[0].retries += 1
                continue
            succeeded.append((tx, rx))
            budget, t_done = radio.service_bytes, t0
            if self.service_order != "fifo" or self.room_aware_service:
                sent = iter(self._select_service(queue, rx, budget, claims))
            else:
                sent = None
            while True:
                if sent is None:
                    if not (queue and queue[0].size <= budget):
                        break
                    p = queue.popleft()
                else:
                    p = next(sent, None)
                    if p is None:
                        break
                budget -= p.size
                t_done += p.size * byte_time
                served_p[q] += 1
                served_b[q] += p.size
                if p.dst == rx:
                    facts_delivered.append(p.pid)
                    facts_delays.append(t_done - p.born)
                    delivered_hops0.append(getattr(p, "hops0", -1))  # -1 for packets of older state
                    delivered_bytes += p.size
                else:
                    p.node, p.node_arrived, p.retries = rx, t_done, 0
                    staged.append(p)
        births = sc.births(k)
        born_hops0 = []
        for b in births:
            p = Packet(self.next_pid, b.src, b.dst, b.size, b.time)
            p.hops0 = self._route_len(int(b.src), int(b.dst))
            born_hops0.append(p.hops0)
            self.next_pid += 1
            staged.append(p)

        # 2. post-service snapshot (before terminations and admission)
        post = self._snapshot(t1)
        waiting_post = sum(len(w) for w in self.waiting)

        # 3. terminations due at t1
        term: dict[str, list[int]] = defaultdict(list)
        term_nodes: dict[int, int] = defaultdict(int)
        relay_terminated = 0

        def terminate(reason: str, pkt: Packet) -> None:
            nonlocal relay_terminated
            term[reason].append(pkt.pid)
            term_nodes[int(pkt.node)] += 1
            relay_terminated += int(pkt.node != pkt.src)

        if self.retry_limit is not None:
            for queue in self.queues:
                while queue and queue[0].retries > self.retry_limit:
                    terminate("retry_limit", queue.popleft())
        if self.deadline_s is not None:
            late = lambda p: p.born + self.deadline_s <= t1 + EPS  # noqa: E731
            for i, queue in enumerate(self.queues):
                if any(late(p) for p in queue):
                    for p in queue:
                        if late(p):
                            terminate("deadline", p)
                    self.queues[i] = deque(p for p in queue if not late(p))
            for w in self.waiting:
                for p in w:
                    if late(p):
                        terminate("deadline", p)
                w[:] = [p for p in w if not late(p)]
            for p in staged:
                if late(p):
                    terminate("deadline", p)
            staged = [p for p in staged if not late(p)]
        max_wait = sc.waiting_max_wait
        for w in self.waiting:
            expired = [p for p in w if p.waiting_since + max_wait <= t1 + EPS]
            if expired:
                for p in expired:
                    terminate("waiting_timeout", p)
                w[:] = [p for p in w if p.waiting_since + max_wait > t1 + EPS]

        # 4. routing update, stale queues
        rehomed = 0
        rehomed_ids: set[int] = set()
        pending: list[tuple[Packet, bool]] = [(p, False) for p in staged]
        updated = None  # nodes whose routes changed this cycle (None: none)
        if self.route_stagger:
            per, n = self.routing.update_every, self.sc.num_nodes
            dests = [d for d in range(n) if (k + 1 + (d * per) // n) % per == 0]
            if dests:
                new = self._compute_routes(k + 1)
                self.next_hop[:, dests] = new[:, dests]
                updated = set(dests)
        elif (k + 1) % self.routing.update_every == 0:
            self.next_hop = self._compute_routes(k + 1)
            updated = range(self.sc.num_nodes)
        if updated is not None:
            if self.stale_queue_policy == "rehome":
                for i, queue in enumerate(self.queues):
                    u, v = self.qlinks[i]
                    stale = [p for p in queue if self.next_hop[u, p.dst] != v]
                    if stale:
                        self.queues[i] = deque(p for p in queue if self.next_hop[u, p.dst] == v)
                        pending += [(p, False) for p in stale]
                        rehomed += len(stale)
                        rehomed_ids.update(p.pid for p in stale)
        for w in self.waiting:
            pending += [(p, True) for p in w if self.next_hop[p.node, p.dst] >= 0]

        # 5. admission in (node arrival, id) order
        if self.room_aware_service and claims.get("pids"):  # packets sent into claimed room first
            first = claims["pids"]
            pending.sort(key=lambda e: (e[0].pid not in first, e[0].node_arrived, e[0].pid))
        else:
            pending.sort(key=lambda e: (e[0].node_arrived, e[0].pid))
        moved_to_waiting = restored = 0
        for p, from_waiting in pending:
            nh = int(self.next_hop[p.node, p.dst])
            if nh >= 0:
                q = self.qindex.get((p.node, nh))
                if q is None:
                    raise ExecutionError(f"route {p.node}->{nh} points to an unregistered queue")
                if ((len(self.queues[q]) < self.qcap[q] if self.buffer == "per_queue"
                        else self._node_total(p.node) < self._node_cap(p.node))
                        or (self.rehome_overflow == "exceed" and p.pid in rehomed_ids)):
                    if from_waiting:
                        self.waiting[p.node].remove(p)
                        restored += 1
                    p.enqueued, p.retries = t1, 0
                    self.queues[q].append(p)
                elif not from_waiting:  # a waiting packet stays in its area
                    terminate("queue_overflow", p)
            elif len(self.waiting[p.node]) < self.wcap[p.node]:
                p.waiting_since = t1
                self.waiting[p.node].append(p)
                moved_to_waiting += 1
            else:
                terminate("waiting_overflow", p)

        # bookkeeping and conservation
        n_term = sum(len(v) for v in term.values())
        self.totals["born"] += len(births)
        self.totals["delivered"] += len(facts_delivered)
        self.totals["terminated"] += n_term
        queued1, waiting1 = self.in_system()
        if self.check_conservation:
            lhs = self.totals["born"]
            rhs = self.totals["delivered"] + self.totals["terminated"] + queued1 + waiting1
            if lhs != rhs:
                raise ExecutionError(f"packet conservation broken at cycle {k}: {lhs} != {rhs}")

        facts = CycleFacts(
            cycle=k, t_start=t0, t_end=t1, planned=tuple(links), succeeded=tuple(succeeded),
            failed=tuple(failed), exec_sinr=sinr, served_packets=served_p, served_bytes=served_b,
            service_ref_bytes=radio.service_bytes, post_service=post, births=len(births),
            risk_packets=queued0 + waiting0 + len(births), delivered_ids=facts_delivered,
            delivered_delays=facts_delays, delivered_bytes=delivered_bytes,
            terminations=dict(term), queued_end=queued1, waiting_end=waiting1,
            waiting_post_service=waiting_post, moved_to_waiting=moved_to_waiting,
            restored_from_waiting=restored, rehomed=rehomed, relay_terminated=relay_terminated,
            terminated_nodes=dict(term_nodes), born_hops0=born_hops0, delivered_hops0=delivered_hops0)

        self.cycle = k + 1
        self.report = self._make_report()
        if (self.terminate_when_drained and sc.traffic_done(self.cycle)
                and queued1 + waiting1 == 0):
            end = EndType.TERMINATED
        elif self.cycle >= sc.horizon:
            end = EndType.TRUNCATED
        else:
            end = EndType.CONTINUE
        return StepOutcome(facts=facts, next_report=self.report, end=end)

    # ------------------------------------------------------------ state I/O
    def state_dict(self) -> dict:
        import copy

        return copy.deepcopy({k: v for k, v in self.__dict__.items()})

    def load_state_dict(self, state: dict) -> None:
        import copy

        self.__dict__.update(copy.deepcopy(state))
