use super::Key;
use fslock::LockFile;
use libloading::Library;
use log::{info, warn};
use std::{
    collections::HashMap,
    env::{temp_dir, var},
    fmt, fs,
    io::ErrorKind::NotFound,
    path::{Path, PathBuf},
    process::{Command, Output, Stdio},
    sync::{Arc, LazyLock, Once, OnceLock, RwLock},
};
#[allow(dead_code)]
pub(crate) const EXPORT_H: &str = "#include \"../export.h\"";
#[allow(dead_code)]
pub(crate) const EXPORT: &str = "__C __export ";
#[allow(dead_code)]
pub(super) fn cache_lib(key: &Key, code: impl FnOnce() -> String) -> Arc<Library> {
    static CACHE: OnceLock<RwLock<HashMap<Key, Arc<Library>>>> = OnceLock::new();
    let cache = CACHE.get_or_init(Default::default);

    if let Some(lib) = cache.read().unwrap().get(key) {
        return lib.clone();
    }

    static ROOT: LazyLock<PathBuf> = LazyLock::new(|| {
        let user = var(if cfg!(windows) { "USERNAME" } else { "USER" }).unwrap();
        let root = temp_dir().join(format!("operators-rs-nv-libs-{user}"));
        fs::create_dir_all(&root).unwrap();
        fs::write(root.join("export.h"), include_str!("cxx/export.h")).unwrap();
        root
    });

    let dir = ROOT.join(format!("{}_{}", key.0, key.1));
    let lock = dir.join(".lock");
    let src = dir.join("src.cu");
    let lib: PathBuf = if cfg!(windows) {
        dir.join("bin").join("lib.dll")
    } else {
        dir.join("lib").join("liblib.so")
    };

    fs::create_dir_all(&dir).unwrap();

    let mut guard = LockFile::open(&lock).unwrap();
    if !guard.try_lock_with_pid().unwrap() {
        warn!(
            "{} is locked by {}",
            dir.display(),
            fs::read_to_string(&lock).unwrap(),
        );
        guard.lock_with_pid().unwrap();
    }

    let code = code();
    let compile = if fs::read_to_string(&src).is_ok_and(|s| s == code) {
        !lib.exists()
    } else {
        if cfg!(use_nvidia) {
            fs::write(dir.join("xmake.lua"), include_str!("cxx/nv.lua")).unwrap();
        } else if cfg!(use_iluvatar) {
            fs::write(dir.join("xmake.lua"), include_str!("cxx/iluvatar.lua")).unwrap();
        } else {
            unreachable!()
        }

        fs::write(src, code).unwrap();
        true
    };
    if compile {
        let arch = format!(
            "-gencode arch=compute_{ver},code=sm_{ver}",
            ver = key.1.to_arch_string()
        );

        static CHECKED: Once = Once::new();
        CHECKED.call_once(xmake_check);

        xmake_config(&dir, arch);
        xmake_build(&dir);
        xmake_install(&dir);
    }
    let lib = unsafe { Library::new(lib) };
    guard.unlock().unwrap();

    let lib = Arc::new(lib.unwrap());
    cache.write().unwrap().insert(key.clone(), lib.clone());
    lib
}
#[allow(dead_code)]
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
#[allow(dead_code)]
fn xmake_config(dir: impl AsRef<Path>, arch: impl fmt::Display) {
    let mut cmd = Command::new("xmake");

    let output = if cfg!(use_nvidia) {
        cmd.arg("config").arg("--toolchain=cuda");
        if let Ok(cuda_root) = std::env::var("CUDA_ROOT") {
            cmd.arg(format!("--cuda={cuda_root}"));
        }
        cmd.arg(format!("--cuflags={arch}"))
            .arg(format!("--culdflags={arch}"))
            .current_dir(&dir)
            .stdout(Stdio::piped())
            .stderr(Stdio::piped())
            .output()
            .unwrap()
    } else if cfg!(use_iluvatar) {
        cmd.arg("config")
            .current_dir(&dir)
            .stdout(Stdio::piped())
            .stderr(Stdio::piped())
            .output()
            .unwrap()
    } else {
        unreachable!()
    };
    let log = read_output(&output);
    if output.status.success() {
        info!("{log}");
    } else {
        panic!("xmake config failed at {}: {log}", dir.as_ref().display());
    }
}
#[allow(dead_code)]
fn xmake_build(dir: impl AsRef<Path>) {
    let output = Command::new("xmake")
        .arg("build")
        .current_dir(&dir)
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .output()
        .unwrap();
    let log = read_output(&output);
    if output.status.success() {
        info!("{log}");
    } else {
        panic!("xmake build failed at {}: {log}", dir.as_ref().display());
    }
}

fn xmake_install(dir: impl AsRef<Path>) {
    let output = Command::new("xmake")
        .arg("install")
        .arg("--installdir=.")
        .current_dir(&dir)
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .output()
        .unwrap();
    let log = read_output(&output);
    if output.status.success() {
        info!("{log}");
    } else {
        panic!("xmake install failed at {}: {log}", dir.as_ref().display());
    }
}
#[allow(dead_code)]
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
