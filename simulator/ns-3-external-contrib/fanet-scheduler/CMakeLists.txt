execute_process(
  COMMAND git -C "${PROJECT_SOURCE_DIR}/contrib/ai" rev-parse HEAD
  OUTPUT_VARIABLE bridge_ai_commit OUTPUT_STRIP_TRAILING_WHITESPACE COMMAND_ERROR_IS_FATAL ANY
)
add_compile_definitions(FANET_BRIDGE_AI_COMMIT="${bridge_ai_commit}")
build_lib(
  LIBNAME fanet-scheduler
  SOURCE_FILES model/probe-txop.cc model/wifi-execution-probe.cc
               model/controlled-state.cc model/controlled-environment.cc
               model/bridge-wire.cc model/bridge-records.cc
               model/scheduled-radio.cc model/control-channel.cc model/moving-channel.cc
               model/nonfrozen-controller.cc
  HEADER_FILES model/probe-txop.h model/wifi-execution-probe.h
               model/controlled-state.h model/controlled-environment.h
               model/bridge-wire.h model/bridge-records.h model/bridge-transport.h
               model/scheduled-radio.h model/control-channel.h model/moving-channel.h
               model/nonfrozen-controller.h
  LIBRARIES_TO_LINK
    ${libcore} ${libnetwork} ${libwifi} ${libspectrum} ${libmobility} ${libpropagation}
  TEST_SOURCES test/wifi-execution-probe-test-suite.cc test/controlled-environment-test-suite.cc
               test/bridge-test-suite.cc
               test/scheduled-radio-test-suite.cc
               test/control-channel-test-suite.cc
               test/moving-channel-test-suite.cc
)

# Embed the exact component source identity; changing a source regenerates the manifest.
file(GLOB_RECURSE probe_inputs CONFIGURE_DEPENDS
     RELATIVE "${CMAKE_CURRENT_SOURCE_DIR}" "*.cc" "*.h" "CMakeLists.txt")
list(SORT probe_inputs)
set(probe_manifest "")
foreach(probe_input IN LISTS probe_inputs)
  set(probe_path "${CMAKE_CURRENT_SOURCE_DIR}/${probe_input}")
  set_property(DIRECTORY APPEND PROPERTY CMAKE_CONFIGURE_DEPENDS "${probe_path}")
  file(SHA256 "${probe_path}" probe_hash)
  string(APPEND probe_manifest "${probe_input}:${probe_hash}\n")
endforeach()
string(SHA256 probe_source_sha256 "${probe_manifest}")
execute_process(
  COMMAND git -C "${PROJECT_SOURCE_DIR}" rev-parse HEAD
  OUTPUT_VARIABLE probe_ns3_commit OUTPUT_STRIP_TRAILING_WHITESPACE COMMAND_ERROR_IS_FATAL ANY
)
target_compile_definitions(fanet-scheduler PRIVATE
  FANET_PROBE_SOURCE_SHA256="${probe_source_sha256}"
  FANET_PROBE_NS3_COMMIT="${probe_ns3_commit}"
  FANET_PROBE_COMPILER="${CMAKE_CXX_COMPILER_ID} ${CMAKE_CXX_COMPILER_VERSION}"
)

if(ENABLE_TESTS)
  build_exec(
    EXECNAME fanet-moving-channel-tests
    SOURCE_FILES test/probe-test-runner.cc test/moving-channel-test-suite.cc
    LIBRARIES_TO_LINK ${libfanet-scheduler}
    EXECUTABLE_DIRECTORY_PATH "${CMAKE_CURRENT_SOURCE_DIR}/test"
  )
  build_exec(
    EXECNAME fanet-control-channel-tests
    SOURCE_FILES test/probe-test-runner.cc test/control-channel-test-suite.cc
    LIBRARIES_TO_LINK ${libfanet-scheduler}
    EXECUTABLE_DIRECTORY_PATH "${CMAKE_CURRENT_SOURCE_DIR}/test"
  )
  build_exec(
    EXECNAME fanet-scheduled-radio-tests
    SOURCE_FILES test/probe-test-runner.cc test/scheduled-radio-test-suite.cc
    LIBRARIES_TO_LINK ${libfanet-scheduler}
    EXECUTABLE_DIRECTORY_PATH "${CMAKE_CURRENT_SOURCE_DIR}/test"
  )
  build_exec(
    EXECNAME fanet-controlled-tests
    SOURCE_FILES test/probe-test-runner.cc test/controlled-environment-test-suite.cc
    LIBRARIES_TO_LINK ${libfanet-scheduler}
    EXECUTABLE_DIRECTORY_PATH "${CMAKE_CURRENT_SOURCE_DIR}/test"
  )
  # Same native TestSuite source, with a bounded entry point that does not link all ns-3 tests.
  build_exec(
    EXECNAME fanet-wifi-probe-tests
    SOURCE_FILES test/probe-test-runner.cc test/wifi-execution-probe-test-suite.cc
    LIBRARIES_TO_LINK ${libfanet-scheduler}
    EXECUTABLE_DIRECTORY_PATH "${CMAKE_CURRENT_SOURCE_DIR}/test"
  )
endif()

# Project bindings remain in the build tree; never install into the upstream ai module.
find_package(Python REQUIRED COMPONENTS Interpreter Development.Module)
set(PYBIND11_FINDPYTHON ON)
find_package(pybind11 CONFIG REQUIRED)
pybind11_add_module(fanet_bridge_native bindings/bridge-binding.cc)
target_link_libraries(fanet_bridge_native PRIVATE ${libfanet-scheduler} ${libai})
set_target_properties(fanet_bridge_native PROPERTIES
  LIBRARY_OUTPUT_DIRECTORY "${CMAKE_OUTPUT_DIRECTORY}/bindings/fanet-scheduler")

if(ENABLE_TESTS)
  build_exec(
    EXECNAME fanet-bridge-tests
    SOURCE_FILES test/probe-test-runner.cc test/bridge-test-suite.cc
    LIBRARIES_TO_LINK ${libfanet-scheduler} ${libai}
    EXECUTABLE_DIRECTORY_PATH "${CMAKE_CURRENT_SOURCE_DIR}/test"
  )
endif()
