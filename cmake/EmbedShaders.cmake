#! OpenDlssRr::Shaders: the shaders compiled with slangc and embedded, for a host that does not compile Slang itself.
find_program(OPENDLSS_RR_SLANGC NAMES slangc HINTS "$ENV{SLANGC}" "$ENV{VULKAN_SDK}/bin" REQUIRED)

set(shader_dir "${PROJECT_SOURCE_DIR}/src/OpenDlssRr/Shaders")
set(output_dir "${CMAKE_CURRENT_BINARY_DIR}/EmbeddedShaders")
file(GLOB shader_includes CONFIGURE_DEPENDS "${shader_dir}/*.slang")
set(embedded_sources "")
foreach(shader Block Mlp PooledHistory HistogramBasic Downsample Post)
    set(spirv "${output_dir}/${shader}.spv")
    set(source "${output_dir}/${shader}.cpp")
    add_custom_command(OUTPUT "${source}"
        COMMAND "${OPENDLSS_RR_SLANGC}" "${shader_dir}/${shader}.slang" -I "${shader_dir}" -target spirv -profile spirv_1_5
                -fvk-use-entrypoint-name -warnings-disable 41012 -o "${spirv}"
        COMMAND "${CMAKE_COMMAND}" -DSPIRV=${spirv} -DSOURCE=${source} -DNAME=${shader} -P "${PROJECT_SOURCE_DIR}/cmake/EmbedFile.cmake"
        DEPENDS "${shader_dir}/${shader}.slang" ${shader_includes} "${PROJECT_SOURCE_DIR}/cmake/EmbedFile.cmake"
        COMMENT "Compiling ${shader}.slang"
        VERBATIM)
    list(APPEND embedded_sources "${source}")
endforeach()

add_library(OpenDlssRrShaders STATIC "${PROJECT_SOURCE_DIR}/src/OpenDlssRr/EmbeddedShaders.cpp" ${embedded_sources})
add_library(OpenDlssRr::Shaders ALIAS OpenDlssRrShaders)
target_link_libraries(OpenDlssRrShaders PUBLIC OpenDlssRr::OpenDlssRr)
