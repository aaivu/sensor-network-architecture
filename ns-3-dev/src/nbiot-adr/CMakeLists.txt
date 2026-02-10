set(source_files
    model/nbiot-common.cc
    model/nbiot-channel.cc
    model/nbiot-net-device.cc
    model/nbiot-energy-model.cc
    helper/nbiot-helper.cc
)

set(header_files
    model/nbiot-common.h
    model/nbiot-channel.h
    model/nbiot-net-device.h
    model/nbiot-energy-model.h
    helper/nbiot-helper.h
)

build_lib(
  LIBNAME nbiot-adr
  SOURCE_FILES ${source_files}
  HEADER_FILES ${header_files}
  LIBRARIES_TO_LINK
    ${libcore}
    ${libnetwork}
    ${libmobility}
    ${libpropagation}
    ${libenergy}
    ${libpoint-to-point}
    ${libbuildings}
    ${libspectrum}
  TEST_SOURCES
)
