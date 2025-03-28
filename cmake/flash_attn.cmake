
###################################################
# LibTorch (required by FlashAttention)
###################################################
if(FF_GPU_BACKEND STREQUAL "cuda")
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
  set(LIBTORCH_PYTHON_DIR "${Torch_INSTALL_PATH}/torch/lib")
  message(STATUS "Using LIBTORCH_PYTHON_DIR: ${LIBTORCH_PYTHON_DIR}")
  find_package(Torch REQUIRED)
  message(STATUS "LIBTORCH_PATH: ${LIBTORCH_PATH}")
  message(STATUS "TORCH_LIBRARIES: ${TORCH_LIBRARIES}")
  find_package(Python3 COMPONENTS Interpreter Development)
  list(APPEND FLEXFLOW_INCLUDE_DIRS ${Python3_INCLUDE_DIRS})
  list(APPEND FLEXFLOW_EXT_LIBRARIES ${Python3_LIBRARIES})
  list(APPEND FLEXFLOW_INCLUDE_DIRS ${TORCH_INCLUDE_DIRS})
  list(APPEND FLEXFLOW_EXT_LIBRARIES "${TORCH_LIBRARIES}")


  ###################################################
  # FlashAttention (installed with pip)
  ###################################################


  if(NOT pip_show_result EQUAL 0)
    execute_process(
      COMMAND python -m pip show flash-attn
      RESULT_VARIABLE pip_show_result
      OUTPUT_VARIABLE pip_output
      OUTPUT_STRIP_TRAILING_WHITESPACE
    )
  endif()

  if(NOT pip_show_result EQUAL 0)
    message(FATAL_ERROR "Could not detect flash-attn installation via pip. Please ensure pip is in your PATH and flash-attn is installed.")
  endif()

  # Extract the installation location.
  # The pip output should contain a line like:
  #   Location: /some/path/to/site-packages
  string(REGEX MATCH "Location: ([^\n]+)" _match "${pip_output}")
  if(NOT _match)
    message(FATAL_ERROR "Failed to parse flash-attn location from pip output.")
  endif()
  set(FlashAttn_INSTALL_PATH "${CMAKE_MATCH_1}")
  string(STRIP "${FlashAttn_INSTALL_PATH}" FlashAttn_INSTALL_PATH)
  message(STATUS "Detected FlashAttn installation path: ${FlashAttn_INSTALL_PATH}")

  file(GLOB FLASH_ATTN_LIB "${FlashAttn_INSTALL_PATH}/flash_attn*.so")
  if(NOT FLASH_ATTN_LIB)
    message(FATAL_ERROR "Could not find FlashAttention shared object file in ${FlashAttn_INSTALL_PATH}. Please ensure FlashAttention is installed.")
  endif()
  get_filename_component(FLASH_ATTN_SO ${FLASH_ATTN_LIB} NAME)
  message(STATUS "FLASH_ATTN_SO: ${FLASH_ATTN_SO}")
  set(FLASH_ATTN_PATH "${FlashAttn_INSTALL_PATH}/${FLASH_ATTN_SO}")
  message(STATUS "FLASH_ATTN_PATH: ${FLASH_ATTN_PATH}")

  add_library(flash_attn_cuda SHARED IMPORTED)
  set_target_properties(flash_attn_cuda PROPERTIES IMPORTED_LOCATION ${FLASH_ATTN_PATH})

  list(APPEND FLEXFLOW_EXT_LIBRARIES flash_attn_cuda)
  message(STATUS "FLEXFLOW_EXT_LIBRARIES: ${FLEXFLOW_EXT_LIBRARIES}")
endif()
