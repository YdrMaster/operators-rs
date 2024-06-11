fn main() {
    use build_script_cfg::Cfg;
    use search_neuware_tools::find_neuware_home;

    let neuware = Cfg::new("detected_neuware");
    if find_neuware_home().is_some() {
        neuware.define();
    }
}
