use super::Key;
use libloading::Library;
use log::info;
use std::{
    collections::HashMap,
    env::temp_dir,
    fmt, fs,
    io::ErrorKind::NotFound,
    path::{Path, PathBuf},
    process::{Command, Output, Stdio},
    sync::{Arc, Once, OnceLock, RwLock},
};

pub(crate) const EXPORT_H: &str = "#include \"../export.h\"";
pub(crate) const EXPORT: &str = "__C __export ";

pub(super) fn cache_lib(key: &Key, code: impl FnOnce() -> String) -> Arc<Library> {
    static CACHE: OnceLock<RwLock<HashMap<Key, Arc<Library>>>> = OnceLock::new();
    let cache = CACHE.get_or_init(Default::default);

    if let Some(lib) = cache.read().unwrap().get(key) {
        return lib.clone();
    }

    static DIR: OnceLock<PathBuf> = OnceLock::new();
    let dir = DIR
        .get_or_init(|| {
            let root = temp_dir().join(format!("InfiniLM{:08x}", std::process::id()));
            fs::create_dir_all(&root).unwrap();
            fs::write(root.join("export.h"), include_str!("cxx/export.h")).unwrap();
            root
        })
        .join(format!("{}_{}", key.0, key.1));

    fs::create_dir_all(&dir).unwrap();
    fs::write(dir.join("xmake.lua"), include_str!("cxx/xmake.lua")).unwrap();
    fs::write(dir.join("src.cu"), code()).unwrap();

    let arch = format!(
        "-gencode arch=compute_{ver},code=sm_{ver}",
        ver = key.1.to_arch_string()
    );

    static CHECKED: Once = Once::new();
    CHECKED.call_once(xmake_check);

    xmake_config(&dir, std::env::var("CUDA_ROOT").unwrap(), arch);
    xmake_build(&dir);
    xmake_install(&dir);

    let lib_path: PathBuf = if cfg!(windows) {
        dir.join("bin").join("lib")
    } else {
        dir.join("lib").join("liblib.so")
    };
    let lib = Arc::new(unsafe { Library::new(lib_path) }.unwrap());
    cache.write().unwrap().insert(key.clone(), lib.clone());
    lib
}

fn xmake_check() {
    const ERR: &str = "xmake detection failed";
    const MSG: &str = "
This project requires xmake to build cub device-wide code.
See [this page](https://xmake.io/#/getting_started?id=installation) to install xmake.";

    match Command::new("xmake").arg("--version").output() {
        Ok(output) => {
            if !output.status.success() {
                let code = output
                    .status
                    .code()
                    .map_or("-".into(), |code| code.to_string());
                let log = read_output(&output);
                panic!("{ERR}.\n\nstatus code: {code}\n\n{log}\n{MSG}");
            } else {
                // xmake detected, nothing to do
            }
        }
        Err(e) if e.kind() == NotFound => {
            panic!("xmake not found.\n{MSG}");
        }
        Err(e) => {
            panic!("{ERR}: {e}\n{MSG}");
        }
    }
}

fn xmake_config(dir: impl AsRef<Path>, cuda_root: impl fmt::Display, arch: impl fmt::Display) {
    let output = Command::new("xmake")
        .arg("config")
        .arg("--toolchain=cuda")
        .arg(format!("--cuda={cuda_root}"))
        .arg(format!("--cuflags={arch}"))
        .arg(format!("--culdflags={arch}"))
        .current_dir(dir)
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .output()
        .unwrap();
    let log = read_output(&output);
    if output.status.success() {
        info!("{log}");
    } else {
        panic!("xmake config failed: {log}");
    }
}

fn xmake_build(dir: impl AsRef<Path>) {
    let output = Command::new("xmake")
        .arg("build")
        .current_dir(dir)
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .output()
        .unwrap();
    let log = read_output(&output);
    if output.status.success() {
        info!("{log}");
    } else {
        panic!("xmake build failed: {log}");
    }
}

fn xmake_install(dir: impl AsRef<Path>) {
    let output = Command::new("xmake")
        .arg("install")
        .arg("--installdir=.")
        .current_dir(dir)
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .output()
        .unwrap();
    let log = read_output(&output);
    if output.status.success() {
        info!("{log}");
    } else {
        panic!("xmake install failed: {log}");
    }
}

fn read_output(output: &Output) -> String {
    let mut log = String::new();
    if !output.stdout.is_empty() {
        log += "stdout:\n\n";
        log += &String::from_utf8_lossy(&output.stdout);
        if log.ends_with('\n') {
            log.push('\n');
        } else {
            log += "\n\n";
        }
    }
    if !output.stderr.is_empty() {
        log += "stderr:\n\n";
        log += &String::from_utf8_lossy(&output.stderr);
        if log.ends_with('\n') {
            log.push('\n');
        } else {
            log += "\n\n";
        }
    }
    log
}

#[test]
fn test_env() {
    assert!(!std::env!("CUDA_ROOT").is_empty());
}

#[test]
fn test_compile() {
    use cuda::Version;
    use libloading::Symbol;
    use std::ffi::{c_char, CStr};

    let lib = cache_lib(
        &("test_compile".into(), Version { major: 8, minor: 0 }),
        || include_str!("cxx/test_compile_8.0/test_compile.cu").into(),
    );
    type Func<'lib> = Symbol<'lib, unsafe extern "C" fn() -> *const c_char>;
    let func: Func = unsafe { lib.get(b"hello_world\0") }.unwrap();
    assert_eq!(
        unsafe { CStr::from_ptr(func()) }.to_bytes(),
        b"Hello, world!"
    );
}
