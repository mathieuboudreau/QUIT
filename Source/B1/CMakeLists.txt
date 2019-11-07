option( BUILD_B1 "Build the B1+/- programs" ON )
if( ${BUILD_B1} )
    file(GLOB_RECURSE B1_SOURCES 
        RELATIVE ${CMAKE_CURRENT_SOURCE_DIR}/../ *.cpp
        )
        target_sources(qi PRIVATE ${B1_SOURCES})
        target_compile_definitions(qi PRIVATE "-DBUILD_B1")
endif()