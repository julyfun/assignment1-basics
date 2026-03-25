#![allow(unused_variables)]

const NUM_PROCESSES: usize = 8;
const MINI_CHUNK_SIZE: usize = 1 << 12;
const PAT: &str = r#"'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"#;

#[pyo3::pymodule]
#[cfg(feature = "pyo3-extension")]
mod cs336_basics {
    use pyo3_stub_gen::derive::gen_stub_pyfunction;
    use pyo3::exceptions::PyValueError;
    use pyo3::prelude::*;
    use pyo3::types::{IntoPyDict, PyAny, PyDict};
    use std::collections::HashMap;
    use std::fs::File;
    use std::io::{BufReader, Read, Seek, SeekFrom};


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
        // read from path
        let file = File::open(path)?;
        let mut reader = BufReader::new(file);
        let bs = find_chunk_boundaries(&mut reader, NUM_PROCESSES, b"<|endoftext|>")?;
        // let mut buffer = vec![0u8];
        // for (start, end) in bs.windows(2).map(|w| (w[0], w[1])) {
        //     reader.seek(SeekFrom::Start(start as u64))?;
        //     buffer.resize(end - start, 0);
        //     reader.read(&mut buffer)?;
        // }

        // let re = fancy_regex::Regex::new(PAT).map_err(|e|
        //     PyValueError::new_err(e.to_string())
        // )?;
        // for m in re.find_iter("test") {
        // }
        Ok(path.to_string())
    }
    // pretoken
}

mod core {
    use std::collections::HashMap;
    use std::fs::File;
    use std::io::{BufReader, Read, Seek, SeekFrom};
    use super::*;
    
    // &[u8] => WordState
    struct WordState {
        cnt: u64,
        pre: Vec<i16>, // out of bounds means no pre/next
        nxt: Vec<i16>,
    }
    impl WordState {
        fn from(s: &[u8]) -> Self {
            let len = s.len() as i16;
            let pre = (0..len).map(|i| i - 1).collect();
            let nxt = (0..len).map(|i| i + 1).collect();
            Self { cnt: 0, pre, nxt }
        }
    }
    // (id, id) => BytesPairState
    #[derive(PartialOrd, PartialEq, Eq)]
    struct BytesPairState {
        cnt: u64,
        word_index: (u32, i16),
    }
    use std::cmp::Ordering;
    impl Ord for BytesPairState {
        fn cmp(&self, other: &Self) -> Ordering {
            if self.cnt < other.cnt {
                Ordering::Less
            } else if self.cnt > other.cnt {
                Ordering::Greater
            } else {
                Ordering::Equal
            }
        }
    }

    fn find_chunk_boundaries(
        file: &mut (impl Read + Seek),
        desired_num_chunks: usize,
        end_token: &[u8],
    ) -> Result<Vec<usize>, std::io::Error> {
        use std::io::ErrorKind::InvalidInput;
        let end_token_str = str::from_utf8(end_token).map_err(|_| InvalidInput)?;
        // 移动指针到末尾
        let file_size = (file.seek(SeekFrom::End(0))?) as usize;
        let chunk_size = (file_size as usize) / desired_num_chunks;
        // e.g. [0, 12, 24] for 2 processes
        let mut boundaries = (0..=desired_num_chunks)
            .map(|i| i * chunk_size)
            .collect::<Vec<_>>();
        // eg [0, 12, 26]
        *boundaries
            .last_mut()
            .ok_or(std::io::ErrorKind::InvalidInput)? = file_size as usize;
        // slightly longer to make sure no end_token is skipped
        let buf_size = MINI_CHUNK_SIZE + end_token.len() - 1;
        let mut buffer = vec![0u8; buf_size];
        for b in &mut boundaries {
            let mut pos = *b;
            file.seek(SeekFrom::Start(pos as u64))?;
            loop {
                // read a minichunk
                let actual_lens = file.read(&mut buffer)?;
                if actual_lens == 0 {
                    *b = file_size;
                    break;
                }
                let s = str::from_utf8(&buffer[..actual_lens]).map_err(|_| InvalidInput)?;
                if let Some(idx) = s.find(end_token_str) {
                    *b = pos + idx;
                    break;
                }
                pos += MINI_CHUNK_SIZE;
            }
        }
        Ok(boundaries)
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
        let pat = " ?(?:".to_string() + special_tokens1.join("|").as_str() + ")|" + PAT;
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
    fn test_re4() -> Result<(), fancy_regex::Error> {
        use fancy_regex::{Regex, escape};
        let special_tokens = vec!["<|endoftext|>"];
        let special_tokens1: Vec<String> = special_tokens
            .iter()
            .map(|x| escape(x).to_string())
            .collect();
        let pat = " ?(?:".to_string() + special_tokens1.join("|").as_str() + ")|" + PAT;
        let t = "hello, who are you? <|endoftext|> whoami233 who";
        let re = Regex::new(pat.as_str())?;
        for m in re.find_iter(t) {
            if let Ok(m) = m
                && !special_tokens.contains(&m.as_str().trim())
            {
                let bytes = m.as_str().as_bytes();
                for (x, y) in bytes.windows(2).map(|w| (w[0], w[1])) {
                }
            }
        }
        Ok(())
    }
}

#[cfg(feature = "pyo3-extension")]
use pyo3_stub_gen::define_stub_info_gatherer;
#[cfg(feature = "pyo3-extension")]
define_stub_info_gatherer!(stub_info);
