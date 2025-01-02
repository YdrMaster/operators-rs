fn main() {
    use build_script_cfg::Cfg;
    use search_cl_tools::find_opencl;
    use search_corex_tools::find_corex;
    use search_cuda_tools::{find_cuda_root, find_nccl_root};
    use search_infini_tools::{find_infini_ccl, find_infini_op, find_infini_rt};

    let cpu = Cfg::new("use_cpu");
    let cl = Cfg::new("use_cl");
    let infini = Cfg::new("use_infini");
    let cuda = Cfg::new("use_cuda");
    let nvidia = Cfg::new("use_nvidia");
    let nccl = Cfg::new("use_nccl");
    let iluvatar = Cfg::new("use_iluvatar");

    if cfg!(feature = "common-cpu") {
        cpu.define()
    }
    if cfg!(feature = "opencl") && find_opencl().is_some() {
        cl.define()
    }
    if cfg!(feature = "infini")
        && find_infini_rt().is_some()
        && find_infini_op().is_some()
        && find_infini_ccl().is_some()
    {
        infini.define()
    }
    let use_nvidia = cfg!(feature = "nvidia-gpu") && find_cuda_root().is_some();
    let use_iluvatar = cfg!(feature = "iluvatar-gpu") && find_corex().is_some();
    if use_nvidia {
        nvidia.define();
        if find_nccl_root().is_some() {
            nccl.define()
        }
    }
    if use_iluvatar {
        iluvatar.define()
    }
    if use_nvidia || use_iluvatar {
        cuda.define()
    }
}
