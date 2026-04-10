#![deny(unused_results)]

mod core;

#[pyo3::pymodule]
#[cfg(feature = "pyo3-extension")]
mod cs336_basics {
    use super::core;
    use pyo3::exceptions::PyValueError;
    use pyo3::prelude::*;
    use pyo3_stub_gen::derive::gen_stub_pyfunction;
    use std::collections::HashMap;

    #[gen_stub_pyfunction]
    #[pyfunction]
    #[pyo3(signature = (a, b), text_signature = "(a: int, b: int) -> str")]
    fn sum_as_string(a: usize, b: usize) -> PyResult<String> {
        Ok((a + b).to_string())
    }

    /// This is a test comment.
    #[gen_stub_pyfunction]
    #[pyfunction]
    #[gen_stub(override_return_type(
        type_repr = "tuple[dict[int, bytes], list[tuple[bytes, bytes]]]"
    ))]
    #[pyo3(signature = (path, vocab_size, special_tokens))]
    fn train(
        path: &str,
        vocab_size: usize,
        special_tokens: Vec<String>,
    ) -> PyResult<(HashMap<u64, Vec<u8>>, Vec<(Vec<u8>, Vec<u8>)>)> {
        core::train(path, vocab_size, special_tokens)
            .map(|(vocab, merges)| {
                let vocab_common_hash: HashMap<u64, Vec<u8>> = vocab.into_iter().collect();
                (vocab_common_hash, merges)
            })
            .map_err(|e| PyValueError::new_err(e.to_string()))
    }
}


#[cfg(test)]
mod tests {
    use super::core::*;
    use std::sync::Once;
    use std::time::{Duration, Instant};

    fn init_tracing() {
        static INIT: Once = Once::new();
        INIT.call_once(|| {
            let _ = tracing_subscriber::fmt()
                .with_env_filter(
                    tracing_subscriber::EnvFilter::from_default_env()
                        .add_directive("info".parse().expect("valid directive")),
                )
                .with_target(false)
                .without_time()
                .try_init();
        });
    }

    #[test]
    fn test_regex() -> Result<(), fancy_regex::Error> {
        use fancy_regex::Regex;
        let t = "hello, who are you? <|endoftext|> whoami who";
        let re = Regex::new(PAT)?;
        for m in re.find_iter(t) {
            if let Ok(m) = m {
                print!("{} ", m.as_str());
                println!("{:?}", m.as_str().as_bytes());
            }
        }
        Ok(())
    }

    #[test]
    fn test_re2() -> Result<(), fancy_regex::Error> {
        use fancy_regex::{Regex, escape};
        let special_tokens = vec!["<|endoftext|>", "whoami233"];
        let special_tokens: Vec<String> = special_tokens
            .into_iter()
            .map(|x| escape(x).to_string())
            .collect();
        let pat = " ?(?:".to_string() + special_tokens.join("|").as_str() + ")|" + PAT;
        let t = "hello, who are you? <|endoftext|> whoami233 who";
        let re = Regex::new(pat.as_str())?;
        for m in re.find_iter(t) {
            if let Ok(m) = m {
                println!("{} ", m.as_str());
            }
        }
        Ok(())
    }

    #[test]
    fn test_re3() -> Result<(), fancy_regex::Error> {
        use fancy_regex::{Regex, escape};
        let special_tokens = vec!["<|endoftext|>"];
        let special_tokens1: Vec<String> = special_tokens
            .iter()
            .map(|x| escape(x).to_string())
            .collect();
        let pat = "(".to_string() + special_tokens1.join("|").as_str() + ")|" + PAT;
        let t = "hello, who are you? <|endoftext|> whoami233 who";
        let re = Regex::new(pat.as_str())?;
        for m in re.find_iter(t) {
            if let Ok(m) = m
                && !special_tokens.contains(&m.as_str().trim())
            {
                println!("m: {}", m.as_str());
            }
        }
        Ok(())
    }

    #[test]
    fn test_re4() -> Result<(), std::io::Error> {
        let (vocab, merges) = train("data/owt_train.txt", 32000, vec!["<|endoftext|>".to_string()])?;
        eprintln!("vocab size: {}", vocab.len());
        eprintln!("merges size: {}", merges.len());
        Ok(())
    }
}

#[cfg(feature = "pyo3-extension")]
use pyo3_stub_gen::define_stub_info_gatherer;
#[cfg(feature = "pyo3-extension")]
define_stub_info_gatherer!(stub_info);
