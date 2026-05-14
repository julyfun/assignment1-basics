#![deny(unused_results)]

mod core;

#[pyo3::pymodule]
#[cfg(feature = "pyo3-extension")]
mod cs336_basics {
    use super::core;
    use pyo3::exceptions::PyValueError;
    use pyo3::prelude::*;
    use pyo3::types::{PyAny, PyType};
    use pyo3_stub_gen::derive::gen_stub_pyfunction;
    use serde_json::Value;
    use std::collections::HashMap;
    use std::fs;

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

    #[pyclass]
    struct Tokenizer {
        inner: core::TokenizerCore,
    }

    fn parse_vocab_json_value(
        v: &Value,
    ) -> Result<HashMap<u64, Vec<u8>>, std::io::Error> {
        let obj = v
            .as_object()
            .ok_or_else(|| core::invalid("vocab should be a JSON object".to_string()))?;
        let mut vocab = HashMap::new();
        for (k, v) in obj {
            let id = k
                .parse::<u64>()
                .map_err(|_| core::invalid("vocab key should be u64".to_string()))?;
            let arr = v.as_array().ok_or_else(|| {
                core::invalid("vocab value should be a list of bytes".to_string())
            })?;
            let mut bytes = Vec::with_capacity(arr.len());
            for x in arr {
                let b = x
                    .as_u64()
                    .ok_or_else(|| core::invalid("byte should be integer".to_string()))?;
                if b > 255 {
                    return Err(core::invalid("byte value should be <= 255".to_string()));
                }
                bytes.push(b as u8);
            }
            let _ = vocab.insert(id, bytes);
        }
        Ok(vocab)
    }

    fn parse_merges_json_value(v: &Value) -> Result<Vec<(Vec<u8>, Vec<u8>)>, std::io::Error> {
        let arr = v
            .as_array()
            .ok_or_else(|| core::invalid("merges should be a list".to_string()))?;
        let mut merges = Vec::with_capacity(arr.len());
        for item in arr {
            let pair = item
                .as_array()
                .ok_or_else(|| core::invalid("merge item should be a pair".to_string()))?;
            if pair.len() != 2 {
                return Err(core::invalid("merge pair length should be 2".to_string()));
            }
            let parse_bytes = |x: &Value| -> Result<Vec<u8>, std::io::Error> {
                let bs = x
                    .as_array()
                    .ok_or_else(|| core::invalid("merge token should be byte list".to_string()))?;
                let mut out = Vec::with_capacity(bs.len());
                for b in bs {
                    let b = b
                        .as_u64()
                        .ok_or_else(|| core::invalid("merge byte should be integer".to_string()))?;
                    if b > 255 {
                        return Err(core::invalid("merge byte should be <= 255".to_string()));
                    }
                    out.push(b as u8);
                }
                Ok(out)
            };
            merges.push((parse_bytes(&pair[0])?, parse_bytes(&pair[1])?));
        }
        Ok(merges)
    }

    fn parse_vocab_merges_from_files(
        vocab_filepath: &str,
        merges_filepath: &str,
    ) -> Result<(HashMap<u64, Vec<u8>>, Vec<(Vec<u8>, Vec<u8>)>), std::io::Error> {
        let vocab_content = fs::read_to_string(vocab_filepath)?;
        let vocab_json: Value =
            serde_json::from_str(vocab_content.as_str()).map_err(|e| core::invalid(e.to_string()))?;

        if vocab_json.get("vocab").is_some() && vocab_json.get("merges").is_some() {
            let vocab = parse_vocab_json_value(
                vocab_json
                    .get("vocab")
                    .ok_or_else(|| core::invalid("missing vocab".to_string()))?,
            )?;
            let merges = parse_merges_json_value(
                vocab_json
                    .get("merges")
                    .ok_or_else(|| core::invalid("missing merges".to_string()))?,
            )?;
            return Ok((vocab, merges));
        }

        let vocab = parse_vocab_json_value(&vocab_json)?;
        let merges_content = fs::read_to_string(merges_filepath)?;
        let merges_json: Value = serde_json::from_str(merges_content.as_str())
            .map_err(|e| core::invalid(e.to_string()))?;
        let merges = parse_merges_json_value(&merges_json)?;
        Ok((vocab, merges))
    }

    #[pymethods]
    impl Tokenizer {
        #[new]
        #[pyo3(signature = (vocab, merges, special_tokens=None))]
        fn new(
            vocab: HashMap<u64, Vec<u8>>,
            merges: Vec<(Vec<u8>, Vec<u8>)>,
            special_tokens: Option<Vec<String>>,
        ) -> PyResult<Self> {
            let mut vocab_fixed = HashMap::with_hasher(core::FixedHasher::default());
            for (k, v) in vocab {
                let _ = vocab_fixed.insert(k, v);
            }
            let inner = core::TokenizerCore::new(vocab_fixed, merges, special_tokens)
                .map_err(|e| PyValueError::new_err(e.to_string()))?;
            Ok(Self { inner })
        }

        #[classmethod]
        #[pyo3(signature = (vocab_filepath, merges_filepath, special_tokens=None))]
        fn from_files(
            _cls: &Bound<'_, PyType>,
            vocab_filepath: &str,
            merges_filepath: &str,
            special_tokens: Option<Vec<String>>,
        ) -> PyResult<Self> {
            let (vocab, merges) = parse_vocab_merges_from_files(vocab_filepath, merges_filepath)
                .map_err(|e| PyValueError::new_err(e.to_string()))?;
            Self::new(vocab, merges, special_tokens)
        }

        fn encode(&self, text: &str) -> PyResult<Vec<u64>> {
            self.inner
                .encode(text)
                .map_err(|e| PyValueError::new_err(e.to_string()))
        }

        fn encode_iterable(&self, iterable: &Bound<'_, PyAny>) -> PyResult<Vec<u64>> {
            let mut chunks = Vec::<String>::new();
            let iter = iterable.try_iter()?;
            for item in iter {
                chunks.push(item?.extract::<String>()?);
            }
            self.inner
                .encode_iterable(chunks)
                .map_err(|e| PyValueError::new_err(e.to_string()))
        }

        fn encode_file(&self, path: &str) -> PyResult<Vec<u64>> {
            self.inner
                .encode_file(path)
                .map_err(|e| PyValueError::new_err(e.to_string()))
        }

        fn encode_file_u16(&self, path: &str) -> PyResult<Vec<u16>> {
            self.inner
                .encode_file_u16(path)
                .map_err(|e| PyValueError::new_err(e.to_string()))
        }

        fn decode(&self, ids: Vec<u64>) -> PyResult<String> {
            self.inner
                .decode(ids.as_slice())
                .map_err(|e| PyValueError::new_err(e.to_string()))
        }
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
