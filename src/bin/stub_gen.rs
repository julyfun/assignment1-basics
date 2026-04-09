#![cfg(feature = "pyo3-extension")]

use pyo3_stub_gen::Result;
use std::path::PathBuf;

fn main() -> Result<()> {
    #[cfg(feature = "pyo3-extension")]
    {
        let mut stub = cs336_basics::stub_info()?;
        // Change output directory to cs336_basics/ subdirectory
        let manifest_dir: PathBuf = std::env!("CARGO_MANIFEST_DIR").into();
        stub.python_root = manifest_dir.join("cs336_basics");
        stub.generate()?;
    }
    Ok(())
}
