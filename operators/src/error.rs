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
        write!(f, "{}:{}: {:?}", self.file, self.line, self.message)
    }
}

impl fmt::Display for ErrorPosition {
    #[inline]
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "{}:{}: \"{}\"", self.file, self.line, self.message)
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
