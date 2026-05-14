idf_component_register(
    SRCS
        "dsp/dsp.cpp"
        "main.cpp"
        "pq_adc.c"
        "pq_frame_protocol.c"
        "pq_serial.c"
    INCLUDE_DIRS
        "."
        "dsp"
    REQUIRES esp_adc driver esp_timer esp_driver_usb_serial_jtag
)

# Accept -DPQ_RAW_MODE=1 on the cmake/idf.py command line (persists in CMake cache)
# or fall back to the environment variable, then default to 0.
if(NOT DEFINED PQ_RAW_MODE)
    if(DEFINED ENV{PQ_RAW_MODE})
        set(PQ_RAW_MODE "$ENV{PQ_RAW_MODE}" CACHE STRING "Raw ADC frame output (0=inference,1=raw)")
    else()
        set(PQ_RAW_MODE "0" CACHE STRING "Raw ADC frame output (0=inference,1=raw)")
    endif()
endif()
target_compile_definitions(${COMPONENT_LIB} PRIVATE PQ_RAW_MODE=${PQ_RAW_MODE})
