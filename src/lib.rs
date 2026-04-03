#![allow(unused_variables)]
#![deny(unused_results)]

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
    use siphasher::sip::SipHasher13;

    use super::*;
    use std::collections::{BinaryHeap, HashMap, HashSet};
    use std::fs::File;
    use std::hash::BuildHasherDefault;
    use std::io::ErrorKind::InvalidInput;
    use std::io::{BufReader, Error, Read, Seek, SeekFrom};

    // id => PreState
    /// merges the same pretokens
    #[derive(Debug, Clone)]
    pub struct PreState {
        pub bytes: Vec<u8>,
        pub cnt: u64,
        pub pre: Vec<i16>, // out of bounds means no pre/next
        pub nxt: Vec<i16>,
        pub vocab_id_at: Vec<Option<u64>>,
    }
    impl PreState {
        pub fn from(s: &[u8], vocab: &HashMap<Vec<u8>, u64, FixedHasher>) -> Option<Self> {
            let len = s.len() as i16;
            let pre = (0..len).map(|i| i - 1).collect();
            let nxt = (0..len).map(|i| i + 1).collect();
            // is Some() only if all in iter are Some()
            let vocab_id_at = s
                .iter()
                .map(|b| vocab.get([*b].as_slice()).copied())
                .collect::<Option<Vec<_>>>()?
                .into_iter()
                .map(Some)
                .collect();
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
    type FixedHasher = BuildHasherDefault<SipHasher13>;
    #[derive(PartialEq, Eq, Clone, Debug)]
    pub struct GroupPairState {
        pub vocab_id_pair: (u64, u64),
        pub cnt: u64,
        pub pre_indices: HashSet<(u64, i16), FixedHasher>,
    }
    use std::cmp::Ordering;
    impl PartialOrd for GroupPairState {
        fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
            Some(self.cmp(other))
        }
    }
    impl Ord for GroupPairState {
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

    pub fn find_chunk_boundaries(
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
            let _ = file.seek(SeekFrom::Start(pos as u64))?;
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

    fn train(
        path: &str,
        vocab_size: usize,
        special_tokens: Vec<String>,
    ) -> Result<String, std::io::Error> {
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
}

#[cfg(test)]
mod tests {
    use super::core::*;
    use super::*;
    use siphasher::sip::SipHasher13;
    use std::collections::{BinaryHeap, HashMap, HashSet};
    use std::fs::File;
    use std::hash::BuildHasherDefault;
    use std::io::ErrorKind::InvalidInput;
    use std::io::{BufReader, Error, Read, Seek, SeekFrom};

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

    fn invalid(s: String) -> std::io::Error {
        std::io::Error::new(InvalidInput, s)
    }

    fn view(a: &Vec<u8>) -> String {
        str::from_utf8(a).unwrap_or("...").to_string()
    }

    fn hasher() -> BuildHasherDefault<SipHasher13> {
        BuildHasherDefault::<SipHasher13>::default()
    }
    #[test]
    fn test_re4() -> Result<(), std::io::Error> {
        use fancy_regex::{Regex, escape};
        const MAX_VOCAB_SIZE: usize = 10000;

        let special_tokens = vec!["<|endoftext|>"];
        let special_tokens1: Vec<String> = special_tokens
            .iter()
            .map(|x| escape(x).to_string())
            .collect();
        let pat = " ?(?:".to_string() + special_tokens1.join("|").as_str() + ")|" + PAT;
        // let t = "who, who, who are you <|endoftext|> who";
        // read from data/tinytrain.txt
        let t = std::fs::read_to_string("data/t.txt")?;
        println!("t.len: {}", t.len());
        let re = Regex::new(pat.as_str()).map_err(|_| InvalidInput)?;

        // byte group => its id
        let mut vocab = {
            let mut v = HashMap::with_hasher(hasher());
            special_tokens.iter().enumerate().for_each(|(i, x)| {
                let _ = v.insert(x.as_bytes().to_vec(), i as u64);
            });
            (0..256).for_each(|i| {
                let _ = v.insert(vec![i as u8], (special_tokens.len() + i as usize) as u64);
            });
            v
        };
        let mut rev_vocab = {
            let mut r = HashMap::with_hasher(hasher());
            vocab.iter().for_each(|(k, v)| {
                let _ = r.insert(*v, k.clone());
            });
            r
        };

        eprintln!("pretoken_dict");

        // mapping a pretoken (Vec<u8>) => pretoken_id(u64)
        let mut cnt = 0;
        let (pretoken_dict, mut pretoken_state) = {
            // mapping Vec<u8> => id(u64)
            let mut d = HashMap::with_hasher(hasher());
            let mut pre_state: HashMap<u64, PreState, _> = HashMap::with_hasher(hasher());
            let mut pre_token_cnt: u64 = 0;
            for m in re.find_iter(t.as_str()) {
                cnt += 1;
                if cnt % 100000 == 0 {
                    eprintln!("cnt: {}", cnt);
                }
                if let Ok(m) = m
                    && !special_tokens.contains(&m.as_str().trim())
                {
                    let m = m.as_str();
                    if let Some(id) = d.get(m.as_bytes()) {
                        pre_state.get_mut(id).ok_or(InvalidInput)?.cnt += 1;
                    } else {
                        let _ = d.insert(m.as_bytes().to_vec(), pre_token_cnt);
                        let _ = pre_state.insert(
                            pre_token_cnt,
                            PreState::from(m.as_bytes(), &vocab).ok_or(InvalidInput)?,
                        );
                        pre_token_cnt += 1;
                    }
                }
            }
            (d, pre_state)
        };

        eprintln!("pretoken_dict size: {}", pretoken_state.len());
        eprintln!("group_pair_heap");

        let mut group_pair_heap = {
            // vocab_id_pair => ByteGroupState
            let mut group_pair_state: HashMap<(u64, u64), GroupPairState, _> =
                HashMap::with_hasher(hasher());
            for (pre_id, pre) in &pretoken_state {
                // e.g. pretoken_id: 12, "word"
                for (x_idx, (x, y)) in pre.bytes.windows(2).map(|w| (w[0], w[1])).enumerate() {
                    let id_x = *vocab.get([x].as_slice()).ok_or(InvalidInput)?;
                    let id_y = *vocab.get([y].as_slice()).ok_or(InvalidInput)?;
                    let state = group_pair_state
                        .entry((id_x, id_y))
                        .or_insert(GroupPairState {
                            vocab_id_pair: (id_x, id_y),
                            cnt: 0,
                            pre_indices: HashSet::with_hasher(hasher()),
                        });
                    state.cnt += pre.cnt;
                    let _ = state.pre_indices.insert((*pre_id as u64, x_idx as i16));
                    let dbg1 = str::from_utf8(&pre.bytes).unwrap_or("...");
                }
            }
            let mut heap: BinaryHeap<GroupPairState> = BinaryHeap::new();
            for state in group_pair_state.values() {
                heap.push(state.clone());
            }
            heap
        };
        // start merging!
        // 若在 heap 中，则 (id_x, id_y) 必然不在 vocab 中
        // vocab_id_pair: (u64, u64),
        // cnt: u64,
        // pre_indices: Vec<(u64, i16)>,
        let mut lazy_deletions =
            HashMap::<(u64, u64), HashSet<(u64, i16), _>, _>::with_hasher(hasher());

        let mut dbg_cnt = 0;
        eprintln!("popping");
        while vocab.len() < MAX_VOCAB_SIZE
            && let Some(mut state) = group_pair_heap.pop()
        {
            let (x_id, y_id) = state.vocab_id_pair;
            let x = rev_vocab.get(&x_id).ok_or(InvalidInput)?.clone();
            let y = rev_vocab.get(&y_id).ok_or(InvalidInput)?.clone();
            let new_group = [x.clone(), y.clone()].concat();
            // let new_id = vocab.len() as u64;
            // let _ = vocab.insert(new_group.clone(), new_id);
            let new_id = vocab.get(&new_group).copied().unwrap_or_else(|| {
                let new_id = vocab.len() as u64;
                let _ = vocab.insert(new_group.clone(), new_id);
                new_id
            });
            let dbg_new_group = str::from_utf8(&new_group).unwrap_or("...");

            if let Some(to_delete) = lazy_deletions.remove(&(x_id, y_id)) {
                for (pre_id, pre_idx) in to_delete {
                    state
                        .pre_indices
                        .remove(&(pre_id, pre_idx))
                        .then_some(())
                        .ok_or(invalid(format!(
                            "remove non existing: <{}> {},{} all: {:?} but tried: {:?}",
                            dbg_new_group,
                            x_id,
                            y_id,
                            state.pre_indices,
                            (pre_id, pre_idx),
                        )))?;
                    state.cnt -= pretoken_state
                        .get(&pre_id)
                        .ok_or(invalid(format!("pretoken_state.get {pre_id}")))?
                        .cnt;
                }
                group_pair_heap.push(state);
                continue;
            }

            if state.cnt == 0 {
                eprintln!("skip cuz cnt == 0");
                continue;
            }

            dbg_cnt += 1;
            if dbg_cnt < 50 {
                eprintln!("inserted <{}> as {}", dbg_new_group, new_id);
            }

            // println!("insert new word: |{dbg_new_group}|");
            let _ = rev_vocab.insert(new_id, new_group);
            // maps vocab_id_pair to GroupPairState
            let mut vocab_id_pair_map = HashMap::with_hasher(hasher());

            for (pretoken_id, x_idx) in state.pre_indices {
                let dbg_keys = pretoken_state.keys().cloned().collect::<Vec<_>>();
                let y_idx = x_idx + x.len() as i16;
                let pretoken = pretoken_state
                    .get_mut(&pretoken_id)
                    .ok_or(invalid(format!("keys {:?} {}", dbg_keys, pretoken_id)))?;
                // vocab_id_at must exist
                // if pretoken.vocab_id_at.get(x_idx as usize).is_none()
                //     || pretoken.vocab_id_at.get(y_idx as usize).is_none()
                // {
                //     continue;
                // }
                if let (Some(actual_x_id_here), Some(actual_y_id_here)) = (
                    pretoken
                        .vocab_id_at
                        .get(x_idx as usize)
                        .ok_or(InvalidInput)?,
                    pretoken
                        .vocab_id_at
                        .get(y_idx as usize)
                        .ok_or(InvalidInput)?,
                ) {
                    if *actual_x_id_here != x_id || *actual_y_id_here != y_id {
                        continue;
                    }
                } else {
                    continue;
                }
                let pre_x_idx = *pretoken
                    .pre
                    .get(x_idx as usize)
                    .ok_or(invalid(format!("pre_x_idx {x_idx}")))?;
                let dbg1 = str::from_utf8(&pretoken.bytes).unwrap_or("...");
                let dbg2 = str::from_utf8(&x).unwrap_or("...");
                let nxt_y_idx = *pretoken.nxt.get(y_idx as usize).ok_or(invalid(format!(
                    "nxt_y_idx {}|{}|{}",
                    x_idx as usize + x.len(),
                    dbg1,
                    dbg2,
                )))?;
                if pre_x_idx > -1 {
                    // |      |     |
                    // pre_x--x--y--nxt_y
                    let pre_id = pretoken
                        .vocab_id_at
                        .get(pre_x_idx as usize)
                        .ok_or(InvalidInput)?
                        .ok_or(invalid(format!(concat!(
                            line!(),
                            ":vocab_id_at[] exists and is None",
                        ))))?;
                    let new_vocab_id_pair = (pre_id, new_id);
                    let dbg_pre = view(rev_vocab.get(&pre_id).ok_or(InvalidInput)?);
                    let dbg_new = view(rev_vocab.get(&new_id).ok_or(InvalidInput)?);
                    // add (pre_x, xy)
                    let state = vocab_id_pair_map
                        .entry(new_vocab_id_pair)
                        .or_insert_with(|| GroupPairState {
                            vocab_id_pair: new_vocab_id_pair,
                            cnt: 0,
                            pre_indices: HashSet::with_hasher(hasher()),
                        });
                    let _ = state.pre_indices.insert((pretoken_id, pre_x_idx));
                    if pretoken_id == 0 && pre_x_idx == 24 {
                        eprintln!(
                            "to-add -------------- {} {} | {} {}",
                            pre_id,
                            new_id,
                            rev_vocab.get(&pre_id).unwrap().len(),
                            rev_vocab.get(&new_id).unwrap().len()
                        );
                    }
                    state.cnt += pretoken.cnt;

                    // lazy delete (pre_x, x)
                    let to_delete = lazy_deletions
                        .entry((pre_id, x_id)) // group
                        .or_insert_with(|| HashSet::with_hasher(hasher()));
                    let _ = to_delete.insert((pretoken_id, pre_x_idx));
                    if pretoken_id == 0 && pre_x_idx == 24 {
                        eprintln!(
                            "to-delete -------------- {} {} | {} {}",
                            pre_id,
                            x_id,
                            rev_vocab.get(&pre_id).unwrap().len(),
                            rev_vocab.get(&x_id).unwrap().len()
                        );
                    }
                }
                if nxt_y_idx < pretoken.bytes.len() as i16 {
                    // |      |     |
                    // pre_x--x--y--nxt_y
                    *pretoken
                        .pre
                        .get_mut(nxt_y_idx as usize)
                        .ok_or(InvalidInput)? = pre_x_idx;
                    let nxt_id = pretoken
                        .vocab_id_at
                        .get(nxt_y_idx as usize)
                        .ok_or(InvalidInput)?
                        .ok_or(invalid(format!(concat!(
                            line!(),
                            ":vocab_id_at[] exists and is None",
                        ))))?;
                    // add (xy, nxt_y)
                    let new_vocab_id_pair = (new_id, nxt_id);
                    let state = vocab_id_pair_map
                        .entry(new_vocab_id_pair)
                        .or_insert_with(|| GroupPairState {
                            vocab_id_pair: new_vocab_id_pair,
                            cnt: 0,
                            pre_indices: HashSet::with_hasher(hasher()),
                        });
                    let _ = state.pre_indices.insert((pretoken_id, x_idx));
                    state.cnt += pretoken.cnt;

                    if pretoken_id == 0 && x_idx == 24 {
                        eprintln!(
                            "to-add -------------- {} {} | {} {}",
                            new_id,
                            nxt_id,
                            rev_vocab.get(&new_id).unwrap().len(),
                            rev_vocab.get(&nxt_id).unwrap().len()
                        );
                    }

                    // lazy delete (y, nxt_y)
                    let to_delete = lazy_deletions
                        .entry((y_id, nxt_id))
                        .or_insert_with(|| HashSet::with_hasher(hasher()));
                    let _ = to_delete.insert((pretoken_id, y_idx));

                    if pretoken_id == 0 && y_idx == 24 {
                        println!(
                            "-----z------ {} {} | {} {}",
                            y_id,
                            nxt_id,
                            rev_vocab.get(&y_id).unwrap().len(),
                            rev_vocab.get(&nxt_id).unwrap().len()
                        );
                    }

                    // maintain pretoken
                    *pretoken
                        .pre
                        .get_mut(nxt_y_idx as usize)
                        .ok_or(InvalidInput)? = x_idx;
                }
                *pretoken
                    .vocab_id_at
                    .get_mut(x_idx as usize)
                    .ok_or(InvalidInput)? = Some(new_id);
                *pretoken
                    .vocab_id_at
                    .get_mut(y_idx as usize)
                    .ok_or(InvalidInput)? = None;
                *pretoken.nxt.get_mut(x_idx as usize).ok_or(InvalidInput)? = nxt_y_idx;

                // let dbg_pretoken= str::from_utf8(&pretoken.bytes).unwrap_or("...");
                // println!("vocab_id_at[{}] = {:?}", dbg_pretoken, pretoken.vocab_id_at);
                // println!(
                //     "vocab_id_at[{}] = {:?}",
                //     dbg_pretoken,
                //     pretoken
                //         .vocab_id_at
                //         .iter()
                //         .map(|v| if let Some(v) = v {
                //             view(rev_vocab.get(v).unwrap())
                //         } else {
                //             "".to_string()
                //         })
                //         .collect::<Vec<_>>()
                // );
            }
            for (_, state) in vocab_id_pair_map.iter() {
                eprintln!("add state {:?}", state);
                group_pair_heap.push(state.clone());
            }
        }
        // for key in vocab.keys().skip(0).take(50) {
        //     let word = str::from_utf8(key).unwrap_or("...");
        //     println!("<{word}>");
        // }
        for m in re.find_iter(t.as_str()) {
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
