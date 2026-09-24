Controlled ns3-ai bridge
=======================

Scope
-----

``fanet-bridge`` exposes ControlledEnvironment through the locked ns3-ai Message
``uint8_t`` vectors. Explicit profiles select controlled or actual-radio execution,
and optional complete-report nonfrozen control. The transport itself is not a radio model.
The Python graph, history, neural model, complete decoder and reward remain shared.
C++ owns the packet state, FIFO, waiting, private inputs and ns-3 events.

The current normative schema is ``docs/architecture/ns3-ai-wire-v2.md`` in the project.
The historical v1 document is retained; current peers explicitly reject its version byte.
The wire version is independent of feature/model/checkpoint versions.

Records
-------

Each message has a 96-byte explicit little-endian header and one canonical typed
value. IDs use uint64, nonnegative nanosecond times use int64 and physical reals
use IEEE754 binary64. Booleans/null have independent tags. Objects contain sorted,
unique ASCII names; arrays retain order. Unknown versions/fields, malformed lengths,
nonfinite reals and out-of-range values fail in release builds too.

The config digest is generated from canonical INIT bytes by Python and bound to
this session by C++; it is a correlation identity, not authentication. C++ does
not recompute SHA-256. Native source and dependency identities are also checked.

INIT carries explicit configuration, initial state, complete private event and
physical trajectories, and algorithm configuration identities. STATE returns only
the current boundary. PLAN contains the complete ordered stable links and separate
synchronous or explicitly tagged nonfrozen timestamps. RESULT contains actual packet facts,
the next state and profile-specific control/PHY/grant-channel evidence.
No expected service packet list, Python next state or precomputed reward is sent.

Lifecycle
---------

The strict exchange sequence is HELLO/READY, INIT/STATE, zero or more PLAN/RESULT,
STOP/FINAL and ACK/CLOSED. Sequence numbers are echoed; successful cycles advance
the epoch exactly once. External trace completion is truncation, not a natural
terminal event. Missing final messages and process failures are never zero rewards.

The Python parent is an independent supervisor. Its direct children are the native
binding worker and the ns-3 executable, each in an owned process group. Blocking
ns3-ai Begin/End calls cannot block this parent. On error it kills and waits both
children before removing only its token-matching shared segment.

Upstream creator construction removes an existing name and uses static mappings.
The project instead reserves the segment with create_only and an owner token, then
constructs the vectors/synchronizers in the worker. Both peers use noncreator mode
and disable automatic finish. A fresh worker process is required for each run.
No upstream source or semaphore implementation is changed.

Both vectors preallocate their declared capacities. Logical sizes change inside
the handshake without reallocating storage. Receivers validate actual size and
header size and copy values before releasing the lock. Short messages cannot
consume the previous message's stale tail. Actual allocation and both directional
capacities are checked; configured overhead is not a universal allocator bound.

Use and validation
------------------

Build the ``fanet-bridge`` and ``fanet_bridge_native`` targets in the configured
ns-3 tree. The extension stays under build/bindings/fanet-scheduler; the project
Python package imports it only in an isolated worker. No install into ns3-ai is needed.

Use the Python ``fanet_scheduler.ns3_adapter.run`` module with an explicit scenario,
ns3-root and output file, or pass Ns3Backend to the common DeterministicRunner.
An incomplete run never replaces a previous valid output file. Explicit close or
a context manager is required; automatic recovery after killing the supervisor
itself remains outside the guarantee. Recorded computation replay now has an explicit
evaluation profile; it is not crash recovery or a new independent experiment. Formal
radio outcome validation and GPU support still require their own registered evidence.

Native tests use the ``fanet-bridge`` TestSuite and dedicated test executable. Python
unit tests check literal bytes and schema rejection; integration tests exercise
real shared-memory exchanges, existing independent answers, cumulative interference,
fixed neural weights, final observations, process failures and ownership isolation.
