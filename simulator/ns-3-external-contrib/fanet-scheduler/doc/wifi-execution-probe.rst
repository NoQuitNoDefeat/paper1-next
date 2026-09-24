Wi-Fi execution component probe
==============================

Scope
-----

This external contrib module implements the step 6.2 development probe. It does not implement the
FANET packet/next-hop/waiting ledger, routing, ns3-ai messages, GNN, rewards, or training. Its local
MAC buffer is diagnostic state, not the research queue. A grant authorizes one MPDU attempt; there
is no automatic service loop that fills an entire scheduling period.

``ProbeTxop`` extends the locked non-QoS ``Txop``. Enqueue retains an MPDU without requesting
channel access. Explicit grants invoke the existing ``FrameExchangeManager::StartTransmission``.
Release does not request another opportunity. DATA and ACK still traverse ``SpectrumWifiPhy``.
No third-party source is modified. This path deliberately bypasses DCF admission, including NAV
and carrier-sense arbitration, but refuses grants while its PHY is transmitting/receiving or its
previous exchange remains active. Simultaneous planned starts are tested in both registration
orders. This is an extension experiment, not a general replacement Wi-Fi MAC.

Fixed development profile
-------------------------

``MakeWifiProbeConfig`` defines all CLI cases before execution. The JSON report includes these
inputs, the fixed radio profile, RNG stream allocation and the complete event sequence.

* Two, four or six static nodes at ``(30*i, 0, 0)`` metres; stable local IDs start at one.
* One radio/channel, non-QoS 802.11a, channel 36, 5180 MHz, 20 MHz.
* Constant ``OfdmRate6Mbps`` DATA/control; 16 dBm transmission, zero antenna gains,
  7 dB noise figure.
* Nist error model; threshold preamble detection at 4 dB and -82 dBm; no frame capture or fading.
* Receiver sensitivity -101 dBm, CCA energy threshold -62 dBm, CCA sensitivity -82 dBm.
* Full directed matrix loss: 60 dB within fixed pairs and 84 dB otherwise. Interference cases use
  80 dB within pairs and retain 84 dB cross loss. Propagation speed is 300000000 m/s.
* 1000-byte payload, no Internet stack or routing protocol, no aggregation or fragmentation.
  RTS threshold is 65535 bytes and fragmentation threshold 65534 bytes.
* ``FrameRetryLimit=1`` means one total attempt, not one extra retry. Normal ACK remains enabled.
* Normal internal queue capacity is 16 packets and lifetime 10 seconds, longer than the run.
  Expiry and overflow cases deliberately change these internal settings.
* Seed 41, run 1, explicit Wi-Fi stream allocation starting at zero.

These values are component fixtures, not accepted formal experiment parameters. The C++ API allows
bounded variations checked by ``ValidateWifiProbeConfig``. Complete plans are validated before node
creation: ordered nonoverlapping windows, distinct half-duplex endpoints and supported fixed peers.
Only odd-ID senders to their paired even-ID receivers are supported. Sleep, switching, QoS,
arbitrary routing and multi-radio operation require separate implementation and acceptance.

Boundary and error diagnostics
------------------------------

With normal reservation enabled, a grant must fit the predicted DATA/SIFS/ACK duration plus both
propagation legs. This is a component guard for the fixed profile, not a research service rule.
It neither terminates an in-flight frame nor guarantees that every failure timer has expired.
The ``window-overrun`` case explicitly bypasses reservation to expose a late ACK. The ``boundary``
case places window close at the independently calculated ACK receive time. Its later same-time
snapshot is diagnostic: a single ``ScheduleNow`` is not a general phase barrier.

The post-reception error hook receives the MPDU payload without its MAC header. The ACK-loss
selector corrupts zero-length payloads at senders, which are ACKs in this DATA/ACK-only profile.
It must not be reused for a profile with other empty control frames. Receiver upper delivery,
sender confirmation, timeout and internal retry-limit drop remain distinct events.

The locked MAC queue may return false before its generic ``DropBeforeEnqueue`` trace. The custom
``EnqueueRejected`` trace captures this return value. The expiry trace can occur when a later
operation wipes expired items; its timestamp is not automatically the nominal expiry deadline.

No research service count, terminal classification or reward is inferred from these diagnostics.
ACK loss, duplicates, in-flight ownership and equal-time settlement must be defined before results
enter the research ledger.

Build and test
--------------

From the project root, with the already configured locked ns-3 environment::

    cmake -S simulator/ns-3-dev -B simulator/ns-3-dev/cmake-cache
    cmake --build simulator/ns-3-dev/cmake-cache \
      --target fanet-wifi-probe fanet-wifi-probe-tests -j 4

Examples and tests must be enabled in that environment. The bounded native test executable uses
the same ``TestSuite`` source registered through ``build_lib(TEST_SOURCES ...)``. It avoids linking
all ns-3 module tests; the global test runner is not required for this component acceptance.

For the tested default profile::

    simulator/ns-3-dev/build/contrib/fanet-scheduler/test/ns3.48-fanet-wifi-probe-tests-default \
      --suite=fanet-wifi-execution-probe --verbose
    simulator/ns-3-dev/build/contrib/fanet-scheduler/examples/ns3.48-fanet-wifi-probe-default \
      --case=single
    .venv/bin/python -m pytest tests/integration/test_ns3_wifi_probe_cli.py --require-ns3

Cases are ``single``, ``empty``, ``gated``, ``dcf``, ``parallel``, ``interference-0``,
``interference-1``, ``interference-2``, ``ack-loss``, ``window-short``, ``window-overrun``,
``boundary``, ``expiry`` and ``overflow``. CLI options also accept explicit ``--seed``
and ``--run``.
Unknown or invalid inputs exit nonzero without a success report. Build artifacts and paths must be
regenerated on another system. The copied ``.clang-format`` matches the locked upstream file.

Report contract
---------------

JSON schema 1 is a completed diagnostic report, not the future ns3-ai wire protocol. It embeds a
SHA-256 digest of sorted component C++/header/CMake inputs, ns-3 commit and compiler identity.
Pytest rejects a stale binary by comparing the compiled digest with current project inputs.

Event ``sequence`` records callback order; ``time_ns`` is integer simulation time. Packet identity
comes from a DATA tag, not ``Packet::GetUid``. Zero means no applicable identity, size or duration;
absent signal/noise values are JSON null. The tag adds no on-air bytes. ACKs have no invented DATA
identity. Frame events use MAC Addr1 as ``peer``; upper delivery uses the sender, and grants use
the planned receiver. ``bytes`` is layer-specific: injected/upper payload, or MPDU including
MAC/FCS for frame traces. ``queue_packets`` is meaningful only in queue snapshots.

``rx_ok`` is a monitor receive event; signal/noise is that monitor sample, not the minimum SINR
over the execution window. ``acked_mpdu`` precedes the upstream dequeue within the same callback.
``window_close`` and ``close_followup`` expose ordering; neither is the research post-service
snapshot. Raw unsuccessful receive reasons are retained.

``RunWifiExecutionProbe`` owns process-global Simulator/RNG state exclusively, destroys its
simulation on normal or C++ exception exit, and creates no subprocess or shared memory.
Repeated runs and invalid-input recovery are covered. The CLI has one report on stdout; the
caller should only publish that report after verifying the exit status and complete JSON.
