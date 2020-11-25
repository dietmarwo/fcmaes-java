#
# This is a CMake makefile.  You can find the cmake utility and
# information about it at http://www.cmake.org
#
cmake_minimum_required(VERSION 3.10)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=gnu++11")
set(CMAKE_CXX_FLAGS_DEBUG          "-g")
set(CMAKE_CXX_FLAGS_MINSIZEREL     "-Os -DNDEBUG")
set(CMAKE_CXX_FLAGS_RELEASE        "-O3 -DNDEBUG -march=native")
set(CMAKE_CXX_FLAGS_RELWITHDEBINFO "-O2 -g")

INCLUDE_DIRECTORIES(/home/xxx/fcmaes-java/cppsrc/include)
INCLUDE_DIRECTORIES(/home/xxx/.sdkman/candidates/java/current/include)
INCLUDE_DIRECTORIES(/home/xxx/.sdkman/candidates/java/current/include/linux)

#set default build type to Release
if(NOT CMAKE_BUILD_TYPE)
   set(CMAKE_BUILD_TYPE "Release" CACHE STRING
      "Choose the type of build, options are: Debug Release
      RelWithDebInfo MinSizeRel." FORCE)
endif()

PROJECT(fcmaeslib)

add_library(fcmaeslib SHARED acmaesoptimizer.cpp hawksoptimizer.cpp deoptimizer.cpp de2optimizer.cpp daoptimizer.cpp gcldeoptimizer.cpp cldeoptimizer.cpp lcldeoptimizer.cpp ldeoptimizer.cpp ascent.cpp  biteoptimizer.cpp csmaoptimizer.cpp)

set(CMAKE_INSTALL_LIBDIR /home/xxx/fcmaes-java/src/main/resources/natives)

install(TARGETS fcmaeslib LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR})
 
