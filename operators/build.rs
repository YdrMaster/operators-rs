﻿fn main() {
    use build_script_cfg::Cfg;
    use search_ascend_tools::find_ascend_toolkit_home;
    use search_cuda_tools::{find_cuda_root, find_nccl_root};
    use search_neuware_tools::find_neuware_home;

    let cpu = Cfg::new("use_cpu");
    let cuda = Cfg::new("use_cuda");
    let nccl = Cfg::new("use_nccl");
    let neuware = Cfg::new("use_neuware");
    let ascend = Cfg::new("use_ascend");
    if cfg!(feature = "common-cpu") {
        cpu.define();
    }
    if cfg!(feature = "nvidia-gpu") && find_cuda_root().is_some() {
        cuda.define();
        if find_nccl_root().is_some() {
            nccl.define();
        }
    }
    if cfg!(feature = "cambricon-mlu") && find_neuware_home().is_some() {
        neuware.define();
    }
    if cfg!(feature = "ascend-card") && find_ascend_toolkit_home().is_some() {
        ascend.define();
    }
}
