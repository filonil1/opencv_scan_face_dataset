cmake_minimum_required(VERSION 3.20)
project(FaceDetector LANGUAGES CXX)
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

# ── OpenCV (standard brew install — no contrib needed) ────────────────────────
list(APPEND CMAKE_PREFIX_PATH
    "/opt/homebrew/opt/opencv"
    "/usr/local/opt/opencv")
find_package(OpenCV 4 REQUIRED
    COMPONENTS core imgproc highgui dnn videoio)
message(STATUS "OpenCV ${OpenCV_VERSION}")

# ── Sources ───────────────────────────────────────────────────────────────────
set(SRC ${CMAKE_SOURCE_DIR}/src)
set(SOURCES
    ${SRC}/face_detector.cpp
    ${SRC}/dataset_builder.cpp
    ${SRC}/person_registry.cpp
    ${SRC}/dataset_manager.cpp
    ${SRC}/face_recognizer.cpp
    ${SRC}/live_window.cpp
    ${SRC}/main.cpp
)
foreach(f ${SOURCES})
    if(NOT EXISTS "${f}")
        message(FATAL_ERROR "Missing: ${f}")
    endif()
endforeach()

add_executable(face_detector ${SOURCES})
target_include_directories(face_detector PRIVATE ${SRC} ${OpenCV_INCLUDE_DIRS})
target_link_libraries(face_detector PRIVATE ${OpenCV_LIBS})

option(USE_MEDIAPIPE_NATIVE "" OFF)
if(USE_MEDIAPIPE_NATIVE)
    set(MP_ROOT "/usr/local/mediapipe" CACHE PATH "")
    find_path(MP_INC mediapipe/tasks/cc/vision/face_detector/face_detector.h
              HINTS ${MP_ROOT}/include REQUIRED)
    find_library(MP_LIB mediapipe_tasks_vision HINTS ${MP_ROOT}/lib REQUIRED)
    target_compile_definitions(face_detector PRIVATE USE_MEDIAPIPE_NATIVE)
    target_include_directories(face_detector PRIVATE ${MP_INC})
    target_link_libraries(face_detector PRIVATE ${MP_LIB})
endif()

if(APPLE)
    set_target_properties(face_detector PROPERTIES
        BUILD_WITH_INSTALL_RPATH TRUE
        INSTALL_RPATH "@executable_path/../lib")
endif()