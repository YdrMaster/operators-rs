fn main() {
    use build_script_cfg::Cfg;
    use search_cuda_tools::{find_cuda_root, find_nccl_root};

    let cpu = Cfg::new("use_cpu");
    let cuda = Cfg::new("use_cuda");
    let nccl = Cfg::new("use_nccl");
    if cfg!(feature = "common-cpu") {
        cpu.define();
    }
    if cfg!(feature = "nvidia-gpu") && find_cuda_root().is_some() {
        cuda.define();
        if find_nccl_root().is_some() {
            nccl.define();
        }
    }
}
