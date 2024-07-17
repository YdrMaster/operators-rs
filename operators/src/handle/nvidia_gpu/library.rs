use super::Key;
use common::{locate_error, ErrorPosition};
use libloading::Library;
use log::warn;
use std::{
    collections::HashMap,
    env::temp_dir,
    ffi::OsStr,
    fs,
    io::ErrorKind::NotFound,
    path::{Path, PathBuf},
    process::{Command, Output, Stdio},
    sync::{Arc, Once, OnceLock, RwLock},
};

pub(crate) const EXPORT_H: &str = "#include \"../export.h\"";
pub(crate) const EXPORT: &str = "__C __export ";

pub(super) fn cache_lib(
    key: &Key,
    code: impl FnOnce() -> String,
) -> Result<Arc<Library>, ErrorPosition> {
    static CACHE: OnceLock<RwLock<HashMap<Key, Arc<Library>>>> = OnceLock::new();
    let cache = CACHE.get_or_init(Default::default);

    if let Some(lib) = cache.read().unwrap().get(key) {
        return Ok(lib.clone());
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

    const CUDA: &str = std::env!("CUDA_ROOT");
    let arch = format!(
        "-gencode arch=compute_{ver},code=sm_{ver}",
        ver = key.1.to_arch_string()
    );

    XMake::new("config")
        .arg("--toolchain=cuda")
        .arg(format!("--cuda={CUDA}"))
        .arg(format!("--cuflags={arch}"))
        .arg(format!("--culdflags={arch}"))
        .run(&dir)
        .map_err(|e| locate_error!("xmake config failed: {e}"))?;
    XMake::new("build")
        .run(&dir)
        .map_err(|e| locate_error!("xmake build failed: {e}"))?;
    XMake::new("install")
        .arg("--installdir=.")
        .run(&dir)
        .map_err(|e| locate_error!("xmake install failed: {e}"))?;

    let lib = Arc::new(unsafe { Library::new(dir.join("bin").join("lib")) }.unwrap());
    cache.write().unwrap().insert(key.clone(), lib.clone());
    Ok(lib)
}

struct XMake(Command);

impl XMake {
    fn check() {
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

    fn new(command: &str) -> Self {
        static CHECKED: Once = Once::new();
        CHECKED.call_once(Self::check);

        let mut xmake = Command::new("xmake");
        xmake.arg(command);
        Self(xmake)
    }

    fn arg(mut self, arg: impl AsRef<OsStr>) -> Self {
        self.0.arg(arg);
        self
    }

    fn run(mut self, dir: impl AsRef<Path>) -> Result<(), String> {
        let output = self
            .0
            .current_dir(dir)
            .stdout(Stdio::piped())
            .stderr(Stdio::piped())
            .output()
            .unwrap();
        let log = read_output(&output);
        if output.status.success() {
            warn!("{log}");
            Ok(())
        } else {
            Err(log)
        }
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
    )
    .unwrap();
    type Func<'lib> = Symbol<'lib, unsafe extern "C" fn() -> *const c_char>;
    let func: Func = unsafe { lib.get(b"hello_world\0") }.unwrap();
    assert_eq!(
        unsafe { CStr::from_ptr(func()) }.to_bytes(),
        b"Hello, world!"
    );
}
