#![allow(unused_variables)]

const NUM_PROCESSES: usize = 8;
const MINI_CHUNK_SIZE: usize = 1 << 12;
const PAT: &str = r#"'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"#;

#[pyo3::pymodule]
#[cfg(feature = "pyo3-extension")]
mod cs336_basics {
    use pyo3::exceptions::PyValueError;
    use pyo3::prelude::*;
    use pyo3::types::{IntoPyDict, PyAny, PyDict};
    use pyo3_stub_gen::derive::gen_stub_pyfunction;
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
    use super::*;
    use std::collections::{HashMap, BinaryHeap};
    use std::fs::File;
    use std::io::{BufReader, Error, Read, Seek, SeekFrom};

    // id => PreState
    /// merges the same pretokens
    struct PreState {
        bytes: Vec<u8>,
        cnt: u64,
        pre: Vec<i16>, // out of bounds means no pre/next
        nxt: Vec<i16>,
        vocab_id_at: Vec<u64>,
    }
    impl PreState {
        fn from(s: &[u8], vocab: &HashMap<Vec<u8>, u64>) -> Option<Self> {
            let len = s.len() as i16;
            let pre = (0..len).map(|i| i - 1).collect();
            let nxt = (0..len).map(|i| i + 1).collect();
            // is Some() only if all in iter are Some()
            let vocab_id_at = s.iter().map(|b| vocab.get([*b].as_slice()).copied()).collect::<Option<Vec<_>>>()?;
            Some(Self {
                bytes: s.to_vec(),
                cnt: 1,
                pre,
                nxt,
                vocab_id_at,
            })
        }
    }
    // &[u8] => pre_id
    //

    // (id, id) => ByteGroupState
    // 可能会 concat 两个 vocab_id 加入 vocab
    #[derive(PartialOrd, PartialEq, Eq, Clone)]
    struct ByteGroupState {
        vocab_id_pair: (u64, u64),
        cnt: u64,
        pre_indices: Vec<(u64, i16)>,
    }
    use std::cmp::Ordering;
    impl Ord for ByteGroupState {
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

    use std::io::ErrorKind::InvalidInput;

    fn find_chunk_boundaries(
        file: &mut (impl Read + Seek),
        desired_num_chunks: usize,
        end_token: &[u8],
    ) -> Result<Vec<usize>, std::io::Error> {
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
    fn test_re4() -> Result<(), std::io::Error> {
        use fancy_regex::{Regex, escape};
        let special_tokens = vec!["<|endoftext|>"];
        let special_tokens1: Vec<String> = special_tokens
            .iter()
            .map(|x| escape(x).to_string())
            .collect();
        let pat = " ?(?:".to_string() + special_tokens1.join("|").as_str() + ")|" + PAT;
        let t = "hello, who are you? <|endoftext|> whoami233 who";
        let re = Regex::new(pat.as_str()).map_err(|_| InvalidInput)?;

        // byte group => its id
        let mut vocab = {
            let mut v = HashMap::new();
            special_tokens.iter().enumerate().for_each(|(i, x)| {
                v.insert(x.as_bytes().to_vec(), i as u64);
            });
            (0..256).for_each(|i| {
                v.insert(vec![i as u8], (special_tokens.len() + i as usize) as u64);
            });
            v
        };
        let mut rev_vocab = {
            let mut r = HashMap::new();
            vocab.iter().for_each(|(k, v)| {
                r.insert(*v, k.clone());
            });
            r
        };
        
        // mapping a pretoken (Vec<u8>) => pretoken_id(u64)
        let (pretoken_dict, pretoken_state) = {
            // mapping Vec<u8> => id(u64)
            let mut d = HashMap::new();
            let mut pre_state: HashMap<u64, PreState> = HashMap::new();
            let mut pre_token_cnt: u64 = 0;
            for m in re.find_iter(t) {
                if let Ok(m) = m
                    && !special_tokens.contains(&m.as_str().trim())
                {
                    let m = m.as_str();
                    if let Some(id) = d.get(m.as_bytes()) {
                        pre_state.get_mut(id).ok_or(InvalidInput)?.cnt += 1;
                    } else {
                        pre_token_cnt += 1;
                        d.insert(m.as_bytes().to_vec(), pre_token_cnt);
                        pre_state.insert(
                            pre_token_cnt, PreState::from(m.as_bytes(), &vocab).ok_or(InvalidInput)?
                        );
                    }
                }
            }
            (d, pre_state)
        };
        let mut byte_group_heap = {
            // vocab_id_pair => ByteGroupState
            let mut byte_group_state: HashMap<(u64, u64), ByteGroupState> = HashMap::new();
            for (pre_id, pre) in &pretoken_state {
                // e.g. pretoken_id: 12, "word"
                for (x_idx, (x, y)) in pre.bytes.windows(2).map(|w| (w[0], w[1])).enumerate() {
                    let id_x = *vocab.get([x].as_slice()).ok_or(InvalidInput)?;
                    let id_y = *vocab.get([y].as_slice()).ok_or(InvalidInput)?;
                    let state = byte_group_state.entry((id_x, id_y)).or_insert(
                        ByteGroupState {
                            vocab_id_pair: (id_x, id_y),
                            cnt: 0,
                            pre_indices: Vec::new(),
                        }
                    );
                    state.cnt += pre.cnt;
                    state.pre_indices.push((*pre_id as u64, x_idx as i16));
                }
            }
            let mut heap: BinaryHeap<ByteGroupState> = BinaryHeap::new();
            for state in byte_group_state.values() {
                heap.push(state.clone());
            }
            heap
        };
        // start merging!
        // 若在 heap 中，则 (id_x, id_y) 必然不在 vocab 中
        while let Some(state) = byte_group_heap.pop() {
            let (id_x, id_y) = state.vocab_id_pair;
            let x = rev_vocab.get(&id_x).ok_or(InvalidInput)?;
            let y = rev_vocab.get(&id_y).ok_or(InvalidInput)?;
        }

        for m in re.find_iter(t) {
            if let Ok(m) = m
                && !special_tokens.contains(&m.as_str().trim())
            {
                let bytes = m.as_str().as_bytes();
                for (x, y) in bytes.windows(2).map(|w| (w[0], w[1])) {}
            }
        }
        Ok(())
    }
}

#[cfg(feature = "pyo3-extension")]
use pyo3_stub_gen::define_stub_info_gatherer;
#[cfg(feature = "pyo3-extension")]
define_stub_info_gatherer!(stub_info);
