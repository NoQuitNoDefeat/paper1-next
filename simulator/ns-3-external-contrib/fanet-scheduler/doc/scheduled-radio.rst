Scheduled radio and explicit execution profiles
==============================================

``ScheduledRadio`` is a plan-driven, single-radio half-duplex development executor. It uses
``SingleModelSpectrumChannel``, ``HalfDuplexIdealPhy`` and the pinned Shannon error model. It is
not an IEEE 802.11 MAC. Receivers synchronize ideally to their scheduled peer; all other signals
remain interference. Two profiles are explicit; neither claims a deployed wireless standard.

* ``ideal-spectrum-v1`` confirms every correct DATA reception to the centralized ledger out of
  band, without an over-the-air ACK. This is paper1-next's execution profile: it matches the
  planning model's centralized confirmation. It may be combined with continuous motion.
* ``transaction-ack-v1`` (inherited, not used by paper1-next) sends an actual reverse ACK after a configured turnaround interval.
  DATA reception creates a non-authoritative copy. Only actual ACK reception atomically commits
  service, relay or delivery in the centralized research ledger. Failed copies are discarded;
  the original packet ID, FIFO position and HOL time survive for a later scheduled cycle.
  There is no automatic same-cycle retry or contention/backoff.

The existing ``ControlledEnvironment`` owns every packet, queue and waiting state. Its external
execution ticket disables synthetic service callbacks. Actual confirmed receptions invoke that
same ledger's FIFO service method. Failed heads stay in place and block followers for the rest
of the period. Successful heads permit another complete transaction only when it fits.

Configuration supplies a complete directed gain matrix, noise, powers and nominal rates at every
boundary, including the final boundary. A fixed propagation delay is explicit. Without a motion
profile the mobility objects are association placeholders and gains stay boundary-frozen.
Public position/velocity reports are separate graph inputs, not private execution facts.

``continuous-motion-frame-quasistatic-v1`` adds ``MotionTrace`` and ``TraceMobility``. Node positions
follow independent native specular reflection between causal boundary samples. Each DATA/ACK
reception start samples current positions and rescales the registered directed gain by its
distance ratio. Registered shadow/block factors retain their explicit boundary update times.
The signal's received power is then held within that frame; signal arrivals/departures still
change aggregate interference. This is not continuously sampled within-frame fast fading.
No pinned upstream PHY source is modified. Position queries are supported; ``CourseChange`` is
not emitted at every reflection, so users must not infer a complete event log from that trace.

Frames include a 24-byte stable packet/source/destination identity header; additional declared
overhead is padding. All bytes consume transmit time. Whole-frame duration must be an exact
integer number of nanoseconds. The caller owns ``Simulator::Run`` and ``Simulator::Destroy``.
On failure, dispose both radio and ledger, then destroy the simulator before any further run.
All successful transactions and every interfering signal callback must drain before settlement.
The next cycle cannot inherit old interference events or duplicate confirmations.

The native ``fanet-scheduled-radio`` suite checks standalone and concurrent DATA, accumulated
interference, failed FIFO heads, no-plan/window gating, exact boundary ordering and the Shannon
capacity boundary. The pinned Shannon model requires strictly more decodable bytes than the
packet size; this differs from the analytic planning predicate at exact equality. The project
keeps that discrepancy visible rather than loosening either comparison.

``ExecutionSnapshot`` separately records current noise, powers and every selected-transmitter to
selected-receiver gain at the actual grant event. Offline set-level SINR auditing uses these
facts, not packet outcomes. ``ReceptionSamples`` records intended moving DATA/ACK receive starts.
Neither record is fed back into the preceding policy observation or used as a Python service gate.

``ControlChannel`` is a separate reliable ordered finite-rate pipe in each direction per node,
with real serialized ns-3 Packets and an independent ground node. ``NonfrozenController`` collects
actual node reports and schedules endpoint commands with one common execution window. The
``fresh-complete-control-v1`` envelope requires every report fresh and complete, checks actual
encoded report bytes, and fails explicitly outside that envelope. Full audit packet records
consume control bandwidth. Per-node capacity scales with node count; this is an abstraction,
not a contention-based control radio or a calibrated minimal hardware message format.

Actual collection, measured or explicitly registered computation, and command distribution
consume simulation time; ns3-ai shared-memory waiting does not. Only timely authorized original
links may execute. Global miss, command miss, physical invalidity and retention are exclusive.
Missing/expired reports followed by continued partial-observation scheduling are not implemented.

The bridge returns actual control/PHY events and the grant-time channel audit, with source
identity and owned-resource supervision. New tests include exact transaction windows, actual ACK
failure/retry, moving reverse paths, report causality, common execution, exclusive removal reasons,
and early close. Historical evidence is in ``docs/ns3-radio-validation.md``; current development
evidence and uncompleted formal work are in ``docs/architecture/python-ns3-stage68-plan.md``.
