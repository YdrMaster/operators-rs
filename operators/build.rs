fn main() {
    use build_script_cfg::Cfg;
    use search_ascend_tools::find_ascend_toolkit_home;
    use search_cl_tools::find_opencl;
    use search_cuda_tools::{find_cuda_root, find_nccl_root};

    let cpu = Cfg::new("use_cpu");
    let cl = Cfg::new("use_cl");
    let cuda = Cfg::new("use_cuda");
    let nccl = Cfg::new("use_nccl");
    let ascend = Cfg::new("use_ascend");

    if cfg!(feature = "common-cpu") {
        cpu.define();
    }
    if cfg!(feature = "opencl") && find_opencl().is_some() {
        cl.define();
    }
    if cfg!(feature = "nvidia-gpu") && find_cuda_root().is_some() {
        cuda.define();
        if find_nccl_root().is_some() {
            nccl.define();
        }
    }
    if cfg!(feature = "ascend") && find_ascend_toolkit_home().is_some() {
        ascend.define();
    }
}
