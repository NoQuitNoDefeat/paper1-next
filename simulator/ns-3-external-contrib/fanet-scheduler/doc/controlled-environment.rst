Controlled FANET research environment
=====================================

Scope and ownership
-------------------

``ns3::fanet::ControlledEnvironment`` independently implements the deterministic packet rules
of the project's research version 19. Python remains the main training environment. This component
owns packet records, next-hop FIFOs, routing waits, pending arrivals and cycle execution. It uses
ns-3 Nodes and Simulator events, without installing WifiMac, NetDevice or an over-the-air protocol.
Packet identities are research value records; they are not ns-3 Packet UIDs or transmitted frames.

Each caller supplies a complete private trajectory of source births, boundary routes and service
conditions. Complete plans contain queue keys only. Expected served packet IDs, next states and
rewards are never input. Graph construction, features, decoding and reward remain shared Python
algorithms. Protocol behavior, ns3-ai transport and production process supervision are future work.

API and lifecycle
-----------------

Create the controller with ``CreateObject<ns3::fanet::ControlledEnvironment>()``.
``Reset`` validates all initial records and optional trajectory before replacing an episode.
``Observe`` returns
an independent value snapshot. Empty registered containers are retained in configuration order;
preloaded FIFO entries and packet/route input order are preserved. Research node IDs map
explicitly to ns-3 Nodes, independently of NodeList numbering. Reset does not run the simulator.

``ValidateAction`` checks the current reference, registered unique queues and all route queries
required by the prospective admission. A missing route is different from explicit no-route.
``StartCycle`` schedules the next cycle; the caller then calls ``Simulator::Run`` and ``Result``.
Each action carries a run/epoch/time reference. Repeated or stale actions fail before execution.
This component validates queue actions, not physical SINR feasibility or the shared graph decoder.

The controller is confined to the simulation thread. It rejects observations, resets and further
actions during a cycle. It owns and removes its pending events on disposal or execution failure;
it does not destroy the global simulator or cancel another component's events. Unexpected callback
failures poison the episode and prevent successful observations/results until a fresh valid reset.
Disposal permanently closes that controller. A finite run always retains its final observation.

The development caller owns an isolated Simulator lifecycle. Logical timestamps are absolute ns-3
nanoseconds: an initial boundary may be later than Simulator::Now(), but a cycle cannot start after
its declared start has passed. A successful cycle calls Stop at its exact ending boundary. Run can
then resume for the next cycle. The caller must Dispose and Destroy when the experiment ends.
Repeating an earlier absolute time range requires a fresh Simulator/controller lifecycle; Reset
on a live controller does not rewind the global simulation clock or another owner's events.

Events and settlement
---------------------

Packets remain in the sending FIFO until a registered whole-packet completion. A completion ticket
binds the queue, packet and exact time; duplicate, non-head or unregistered completions fail before
another packet can be removed. Success moves the packet to relay pending or final delivery. Only
the initial FIFO prefix is eligible; newly admitted packets cannot take another hop that cycle.

Owned input events hold outstanding-work tickets. Each root registers a same-time completion child
before releasing its own ticket. Closing the cycle merely records a request. Settlement requires
both that request and zero outstanding tickets, including boundary-input children. There is no
polling loop or 1 ns time shift. This mechanism covers controlled registered work only; it is not
a claim that unknown wireless callbacks or ACK timers are automatically accounted for.

Settlement freezes the post-service snapshot first, expires old waiting packets before restoration,
then merges pending/restored arrivals by (node arrival time, numeric packet ID). Old FIFO entries
keep their order. Capacity refusal and waiting expiry retain their research terminal categories.
Risk IDs are initial active IDs union new source IDs. Activity, delivery, termination and per-queue
service bytes are checked for conservation before publishing the next state.

Semantic PacketEvent order follows the Python reference's presentation order at equal timestamps.
The separate callback records preserve actual controller execution order and work-ticket counts.
Changing closing-event registration order must not change logical packet facts or snapshots.

Numeric limits
--------------

Research node/packet IDs and byte quantities are uint64; zero IDs are valid. Times use nonnegative
int64 nanoseconds and require ns-3 Time resolution NS. Addition and byte products are checked.
The finite horizon is a positive integer number of periods, and waiting limits are positive period
multiples. The maximum admitted waiting expiry (end + maxWait) must also fit int64. A cycle's risk
population times packet size must fit the byte range. These are explicit C++ representability
limits, not changes to Python's arbitrary-precision integers or a measured drone capacity.

Run references are fresh printable ASCII tokens of 1-128 bytes. They cannot be reused on one
controller. These local constraints do not finalize the future ns3-ai binary encoding.
Constant-rate service requires exact integer packet durations and the exact floor byte budget;
checked quotient/remainder arithmetic avoids overflowing a naive rate-times-duration product.
Packet deadlines and wireless retransmission mechanisms are explicitly unsupported in this slice.

Development fixture executable
------------------------------

``fanet-controlled`` reads a bounded whitespace-delimited fixture on stdin and prints complete
JSON only after every requested plan succeeds and the final observation is available. The optional
``--boundary-last`` changes event registration for diagnostics. Invalid input or incomplete
execution returns nonzero with an error on stderr and no success JSON. The process creates no
children or shared memory. Input is capped at 8 MiB and record counts at 100000 solely to bound this
development reader; these are not shared-memory or network-capacity measurements.

The input starts with FANET_CONTROLLED_V1, RUN and CONFIG, followed by NODES, QUEUES, WAITS,
PACKETS, QUEUE_STATE, WAIT_STATE and ROUTES. TRAJECTORY carries a presence flag and, if present,
the frame count; each FRAME contains SERVICES, BIRTHS and UPDATES. PLANS carries referenced
queue actions. Integers are decimal, booleans are 0/1 and missing optional numbers are ``none``.
The exact writer is ``tests/integration/test_ns3_controlled.py::_fixture``. No expected answer is
serialized. This development format is deliberately separate from the pending S6-P04 protocol.

Tests and provenance
--------------------

``fanet-controlled-environment`` is a native ns-3 TestSuite, also available via the bounded
``fanet-controlled-tests`` executable. Its assertions cover initialization, FIFO service,
same-time child accounting, numeric bounds, duplicate facts and lifecycle behavior.
The Python integration suite reuses existing hand-answer tests and fixtures, adds the B-T cases,
and compares complete observations/events/results against an independent C++ process. Longer
prefixes are replayed in fresh processes; each process independently retains state across all
requested cycles. Generated inputs are drawn once in Python and transmitted explicitly, never
recreated in C++ by assuming identical seeds. The shared reward receives reconstructed C++ facts.

The executable reports the module C++/header/CMake source digest, ns-3 commit and compiler. Tests
reject stale binaries. Actual environment, commands, results and limits are recorded in
``docs/ns3-controlled-validation.md`` in the project root, separately from wireless probe evidence.
