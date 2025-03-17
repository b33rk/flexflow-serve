cmake_minimum_required(VERSION 3.10)

project(FlexFlow_Peft)

# Normal PEFT
set(project_target1 peft)
set(CPU_SRC1
  ${FLEXFLOW_CPP_DRV_SRC}
  peft.cc
  ../models/llama.cc
  ../models/opt.cc
  ../models/falcon.cc
  ../models/starcoder.cc
  ../models/mpt.cc)

if (FF_GPU_BACKEND STREQUAL "cuda" OR FF_GPU_BACKEND STREQUAL "hip_cuda")
  cuda_add_executable(${project_target1} ${CPU_SRC1})
  if (FF_GPU_BACKEND STREQUAL "hip_cuda")
    target_compile_definitions(${project_target1} PRIVATE __HIP_PLATFORM_NVIDIA__)
  endif()
elseif(FF_GPU_BACKEND STREQUAL "hip_rocm")
  set_source_files_properties(${CPU_SRC1} PROPERTIES LANGUAGE HIP)
  hip_add_executable(${project_target1} ${CPU_SRC1})
  if (FF_HIP_ARCH STREQUAL "")
    message(FATAL_ERROR "FF_HIP_ARCH is empty!")
  endif()
  set_property(TARGET ${project_target1} PROPERTY HIP_ARCHITECTURES "${FF_HIP_ARCH}")
  target_compile_definitions(${project_target1} PRIVATE __HIP_PLATFORM_AMD__)
else()
  message(FATAL_ERROR "Compilation of ${project_target1} for ${FF_GPU_BACKEND} backend not yet supported")
endif()

target_include_directories(${project_target1} PRIVATE ${FLEXFLOW_INCLUDE_DIRS} ${CMAKE_INSTALL_INCLUDEDIR})
target_include_directories(${project_target1} PRIVATE ${CMAKE_SOURCE_DIR}/inference)
target_link_libraries(${project_target1} -Wl,--whole-archive flexflow -Wl,--no-whole-archive ${FLEXFLOW_EXT_LIBRARIES})
set(BIN_DEST "bin")
install(TARGETS ${project_target1} DESTINATION ${BIN_DEST})
