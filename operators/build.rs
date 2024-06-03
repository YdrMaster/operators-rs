fn main() {
    use search_cuda_tools::{find_cuda_root, find_nccl_root, Cfg};

    let cpu = Cfg::new("cpu");
    let cuda = Cfg::new("cuda");
    let nccl = Cfg::new("nccl");
    if cfg!(feature = "common-cpu") {
        cpu.detect();
    }
    if cfg!(feature = "nvidia-gpu") {
        if find_cuda_root().is_some() {
            cuda.detect();
        }
        if find_nccl_root().is_some() {
            nccl.detect();
        }
    }
}
