target("lib")
    set_kind("shared")
    set_toolchains("cuda")
    set_optimize("aggressive")

    if is_plat("windows") then
        add_cuflags("-Xcompiler=/utf-8", "--expt-relaxed-constexpr", "--allow-unsupported-compiler")
    end

    set_languages("cxx17")
    add_files("src.cu")
