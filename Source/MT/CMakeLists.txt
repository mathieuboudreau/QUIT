option( BUILD_MT "Build the MT Programs" ON )
if( ${BUILD_MT} )
    file(GLOB_RECURSE MT_SOURCES 
        RELATIVE ${CMAKE_CURRENT_SOURCE_DIR}/../ *.cpp
        )
    set(MT_SOURCES ${MT_SOURCES} PARENT_SCOPE)
endif()