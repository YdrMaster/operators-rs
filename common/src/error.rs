use core::fmt;
use std::error::Error;

#[derive(Clone)]
pub struct ErrorPosition {
    file: &'static str,
    line: u32,
    message: String,
}

impl Error for ErrorPosition {}

impl ErrorPosition {
    #[inline]
    pub const fn new(file: &'static str, line: u32, message: String) -> Self {
        Self {
            file,
            line,
            message,
        }
    }
}

impl fmt::Debug for ErrorPosition {
    #[inline]
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "{self}")
    }
}

struct Colored<T>(T, u32);

impl<T: fmt::Display> fmt::Display for Colored<T> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "\x1b[{}m{}\x1b[0m", self.1, self.0)
    }
}

#[inline]
fn red<T>(t: T) -> Colored<T> {
    Colored(t, 31)
}

impl fmt::Display for ErrorPosition {
    #[inline]
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        if self.message.lines().nth(1).is_some() {
            writeln!(f, "{}:{}:", self.file, self.line)?;
            writeln!(f, "{}", red("+--"))?;
            for line in self.message.lines() {
                write!(f, "{} ", red('|'))?;
                writeln!(f, "{line}")?;
            }
            writeln!(f, "{}", red("+--"))
        } else {
            write!(f, "{}:{}: {}", self.file, self.line, self.message)
        }
    }
}

#[macro_export]
macro_rules! locate_error {
    ($msg:expr) => {
        $crate::ErrorPosition::new(file!(), line!(), $msg.to_string())
    };
    () => {
        locate_error!("Error occurred")
    };
}

#[test]
fn test_locate_error() {
    fn error() -> ErrorPosition {
        locate_error!()
    }
    // ...
    let e = error();
    // ...
    println!("{e:?}");
}
