fn main() {
    use build_script_cfg::Cfg;
    use search_cuda_tools::find_cuda_root;
    use search_neuware_tools::find_neuware_home;

    let cpu = Cfg::new("use_cpu");
    let cuda = Cfg::new("use_cuda");
    let neuware = Cfg::new("use_neuware");
    if cfg!(feature = "common-cpu") {
        cpu.define();
    }
    if cfg!(feature = "nvidia-gpu") && find_cuda_root().is_some() {
        cuda.define();
    }
    if cfg!(feature = "cambricon-mlu") && find_neuware_home().is_some() {
        neuware.define();
    }
}
