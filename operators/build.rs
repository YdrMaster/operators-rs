fn main() {
    use build_script_cfg::Cfg;
    use search_cl_tools::find_opencl;
    use search_cuda_tools::{find_cuda_root, find_nccl_root};
    use search_infini_tools::find_infini_rt;

    let cpu = Cfg::new("use_cpu");
    let cl = Cfg::new("use_cl");
    let infini = Cfg::new("use_infini");
    let cuda = Cfg::new("use_cuda");
    let nccl = Cfg::new("use_nccl");

    if cfg!(feature = "common-cpu") {
        cpu.define();
    }
    if cfg!(feature = "opencl") && find_opencl().is_some() {
        cl.define();
    }
    if cfg!(feature = "infini") && find_infini_rt().is_some() {
        infini.define();
    }
    if cfg!(feature = "nvidia-gpu") && find_cuda_root().is_some() {
        cuda.define();
        if find_nccl_root().is_some() {
            nccl.define();
        }
    }
}
