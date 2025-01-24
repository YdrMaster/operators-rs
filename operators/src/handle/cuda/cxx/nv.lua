target("lib")
    set_kind("shared")
    set_toolchains("cuda")
    set_optimize("aggressive")

    if is_plat("windows") then
        -- See <https://stackoverflow.com/questions/78515942/cuda-compatibility-with-visual-studio-2022-version-17-10>
        add_cuflags("-Xcompiler=/utf-8", "--expt-relaxed-constexpr", "--allow-unsupported-compiler")
        add_defines("_ALLOW_COMPILER_AND_STL_VERSION_MISMATCH")
    end

    set_languages("cxx17")
    add_files("src.cu")
target_end()
