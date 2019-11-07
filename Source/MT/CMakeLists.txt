option( BUILD_MT "Build the MT Programs" ON )
if( ${BUILD_MT} )
    file(GLOB_RECURSE MT_SOURCES 
        RELATIVE ${CMAKE_CURRENT_SOURCE_DIR}/../ *.cpp
        )
    target_sources(qi PRIVATE ${MT_SOURCES})
    target_compile_definitions(qi PRIVATE "-DBUILD_MT")
endif()