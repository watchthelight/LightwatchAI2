# LightwatchCompilerFlags.cmake
# Compiler configuration for cross-platform builds

# Detect compiler and set appropriate flags
function(lightwatch_set_compiler_flags target)
    if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
        target_compile_options(${target} PRIVATE
            -Wall
            -Wextra
            -Wpedantic
            -Wconversion
            -Wsign-conversion
            -Wcast-qual
            -Wformat=2
            -Wundef
            -Werror=return-type
            -Wno-unused-parameter
        )

        # Optimization flags for release builds
        target_compile_options(${target} PRIVATE
            $<$<CONFIG:Release>:-O3>
            $<$<CONFIG:Release>:-march=native>
            $<$<CONFIG:Release>:-DNDEBUG>
        )

        # Debug flags
        target_compile_options(${target} PRIVATE
            $<$<CONFIG:Debug>:-O0>
            $<$<CONFIG:Debug>:-g>
            $<$<CONFIG:Debug>:-fno-omit-frame-pointer>
        )

    elseif(MSVC)
        target_compile_options(${target} PRIVATE
            /W4
            /permissive-
            /Zc:__cplusplus
        )

        # Optimization flags for release builds
        target_compile_options(${target} PRIVATE
            $<$<CONFIG:Release>:/O2>
            $<$<CONFIG:Release>:/DNDEBUG>
        )

        # Debug flags
        target_compile_options(${target} PRIVATE
            $<$<CONFIG:Debug>:/Od>
            $<$<CONFIG:Debug>:/Zi>
        )
    endif()

    # Position-independent code for library targets
    set_target_properties(${target} PROPERTIES
        POSITION_INDEPENDENT_CODE ON
    )
endfunction()

# Global compiler detection variables
if(CMAKE_CXX_COMPILER_ID MATCHES "GNU")
    set(LIGHTWATCH_COMPILER_GCC TRUE)
elseif(CMAKE_CXX_COMPILER_ID MATCHES "Clang")
    set(LIGHTWATCH_COMPILER_CLANG TRUE)
elseif(MSVC)
    set(LIGHTWATCH_COMPILER_MSVC TRUE)
endif()
