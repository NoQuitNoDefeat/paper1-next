build_lib_example(
  NAME fanet-controlled
  SOURCE_FILES fanet-controlled.cc
  LIBRARIES_TO_LINK ${libfanet-scheduler}
)

build_lib_example(
  NAME fanet-wifi-probe
  SOURCE_FILES fanet-wifi-probe.cc
  LIBRARIES_TO_LINK ${libfanet-scheduler}
)

build_lib_example(
  NAME fanet-bridge
  SOURCE_FILES fanet-bridge.cc
  LIBRARIES_TO_LINK ${libfanet-scheduler} ${libai}
)
