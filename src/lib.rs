#![allow(unused_variables)]

use pyo3_stub_gen::{derive::gen_stub_pyfunction, define_stub_info_gatherer};

#[pyo3::pymodule]
mod cs336_basics {
    use pyo3::prelude::*;
    use pyo3::types::{PyDict, PyAny, IntoPyDict};
    use super::*;
    
    #[gen_stub_pyfunction]
    #[pyfunction]
    #[pyo3(signature = (a, b), text_signature = "(a: int, b: int) -> str")]
    fn sum_as_string(a: usize, b: usize) -> PyResult<String> {
        Ok((a + b).to_string())
    }
    
    /// This is a test comment.
    #[gen_stub_pyfunction]
    #[pyfunction]
    #[pyo3(signature = (path, vocab_size, special_tokens), text_signature = "(path: str, vocab_size: int, special_tokens: list[str]) -> str")]
    fn train(path: &str, vocab_size: usize, special_tokens: Vec<String>) -> PyResult<String> {
        Ok(path.to_string())
    }
    
}

define_stub_info_gatherer!(stub_info);
