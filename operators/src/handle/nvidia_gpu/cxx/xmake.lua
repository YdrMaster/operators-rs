option("iluvatar") --目前的作用只是一个标识符，判断是否开启天数编译
    set_default("auto") -- 默认值
    set_showmenu(true) -- 是否显示在菜单中
    set_category("ILUVATAR   SDK Configuration") -- 分类名称
    set_description(" The ILUVATAR  SDK Directory (default: auto)") -- 描述信息

toolchain("iluvatar.toolchain")
    set_toolset("cc"  , "clang"  )
    set_toolset("cxx" , "clang++")
    set_toolset("cu"  , "clang++")
    set_toolset("culd", "clang++")
    set_toolset("cu-ccbin", "$(env CXX)", "$(env CC)")
toolchain_end()
rule("iluvatar.env")
    add_deps("cuda.env", {order = true})
    after_load(function (target)
        local old = target:get("syslinks")
        local new = {}

        for _, link in ipairs(old) do
            if link ~= "cudadevrt" then
                table.insert(new, link)
            end
        end

        if #old > #new then
            target:set("syslinks", new)
            local log = "cudadevrt removed, syslinks = { "
            for _, link in ipairs(new) do
                log = log .. link .. ", "
            end
            log = log:sub(0, -3) .. " }"
            print(log)
        end
    end)
rule_end()


target("lib")
    set_kind("shared")
    set_optimize("aggressive")
    set_languages("cxx17")
    add_files("src.cu")
    -- if has_config("iluvatar") then
        -- 如果配置了 Iluvatar，则按照 Iluvatar 的方式编译
        set_toolchains("iluvatar.toolchain")
        add_rules("iluvatar.env")
        set_values("cuda.rdc", false)
        add_links("cudart")   -- 首选动态链接 cudart 以免链接 cudart_static
    -- end

target_end()

