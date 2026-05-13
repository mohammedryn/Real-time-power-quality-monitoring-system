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

if(DEFINED ENV{PQ_RAW_MODE})
    target_compile_definitions(${COMPONENT_LIB} PRIVATE PQ_RAW_MODE=$ENV{PQ_RAW_MODE})
else()
    target_compile_definitions(${COMPONENT_LIB} PRIVATE PQ_RAW_MODE=0)
endif()
