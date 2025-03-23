
###################################################
# LibTorch (required by FlashAttention)
###################################################

# If that fails, try using "python -m pip show torch"
if(NOT pip_show_result EQUAL 0)
  execute_process(
    COMMAND python -m pip show torch
    RESULT_VARIABLE pip_show_result
    OUTPUT_VARIABLE pip_output
    OUTPUT_STRIP_TRAILING_WHITESPACE
  )
endif()

if(NOT pip_show_result EQUAL 0)
  message(FATAL_ERROR "Could not detect torch installation via pip. Please ensure pip is in your PATH and torch is installed.")
endif()

# Extract the installation location.
# The pip output should contain a line like:
#   Location: /some/path/to/site-packages
string(REGEX MATCH "Location: ([^\n]+)" _match "${pip_output}")
if(NOT _match)
  message(FATAL_ERROR "Failed to parse torch location from pip output.")
endif()
set(Torch_INSTALL_PATH "${CMAKE_MATCH_1}")
string(STRIP "${Torch_INSTALL_PATH}" Torch_INSTALL_PATH)
message(STATUS "Detected torch installation path: ${Torch_INSTALL_PATH}")

# # Assume that the Torch CMake files are under: <Torch_INSTALL_PATH>/torch/share/cmake/Torch
set(Torch_DIR "${Torch_INSTALL_PATH}/torch/share/cmake/Torch")
message(STATUS "Using Torch_DIR: ${Torch_DIR}")
find_package(Torch REQUIRED)
message(STATUS "LIBTORCH_PATH: ${LIBTORCH_PATH}")
message(STATUS "TORCH_LIBRARIES: ${TORCH_LIBRARIES}")
find_package(Python3 COMPONENTS Interpreter Development)
list(APPEND FLEXFLOW_INCLUDE_DIRS ${Python3_INCLUDE_DIRS})
list(APPEND FLEXFLOW_EXT_LIBRARIES ${Python3_LIBRARIES})
list(APPEND FLEXFLOW_INCLUDE_DIRS ${TORCH_INCLUDE_DIRS})
list(APPEND FLEXFLOW_EXT_LIBRARIES "${TORCH_LIBRARIES}")


