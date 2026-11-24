cmake_minimum_required(VERSION 3.29)
project(OpenDlssRr LANGUAGES CXX)

include(cmake/CompilerOptions.cmake)

#! A host that compiles the Slang shaders itself (as NoorRay does) turns this off and passes its SPIR-V in Settings::shaders.
option(OPENDLSS_RR_EMBED_SHADERS "Compile the shaders with slangc and embed them in OpenDlssRr::Shaders" ON)

add_subdirectory(src)
