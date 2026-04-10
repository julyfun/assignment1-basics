#![allow(dead_code)]

mod dec;

use siphasher::sip::SipHasher13;
use std::collections::{BinaryHeap, HashMap, HashSet};
use std::fs::File;
use std::hash::BuildHasherDefault;
use std::io::ErrorKind::InvalidInput;
use std::io::{BufReader, Read, Seek, SeekFrom};

const NUM_PROCESSES: usize = 32;
const MINI_CHUNK_SIZE: usize = 1 << 12;
pub const PAT: &str =
    r#"'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|[^\S\r\n]|[\r\n]+"#;

pub fn invalid(s: String) -> std::io::Error {
    std::io::Error::new(InvalidInput, s)
}

// id => PreState
/// merges the same pretokens
#[derive(Debug, Clone)]
pub struct PreState {
    pub bytes: Vec<u8>,
    pub cnt: u64,
    pub pre: Vec<i64>, // out of bounds means no pre/next
    pub nxt: Vec<i64>,
    pub vocab_id_at: Vec<Option<u64>>,
}
impl PreState {
    pub fn from(s: &[u8], rev_vocab: &HashMap<Vec<u8>, u64, FixedHasher>) -> Option<Self> {
        let len = s.len() as i64;
        let pre = (0..len).map(|i| i - 1).collect();
        let nxt = (0..len).map(|i| i + 1).collect();
        // is Some() only if all in iter are Some()
        let vocab_id_at = s
            .iter()
            .map(|b| rev_vocab.get([*b].as_slice()).copied())
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

// (id, id) => ByteGroupState
// 可能会 concat 两个 vocab_id 加入 vocab
pub type FixedHasher = BuildHasherDefault<SipHasher13>;
#[derive(Debug, Clone)]
pub struct GroupPairState {
    pub cnt: u64,
    pub pre_indices: HashSet<(u64, i64), FixedHasher>,
    pub latest_ver: u64,
}

#[derive(PartialEq, Eq, Clone, Debug)]
pub struct GroupPairCandidate {
    pub ver: u64,
    pub cnt: u64,
    pub vocab_ids: (u64, u64),
    pub left: Vec<u8>,
    pub right: Vec<u8>,
}
use std::cmp::Ordering;
impl PartialOrd for GroupPairCandidate {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}
impl Ord for GroupPairCandidate {
    fn cmp(&self, other: &Self) -> Ordering {
        if self.cnt < other.cnt {
            Ordering::Less
        } else if self.cnt > other.cnt {
            Ordering::Greater
        } else if self.left < other.left {
            Ordering::Less
        } else if self.left > other.left {
            Ordering::Greater
        } else if self.right < other.right {
            Ordering::Less
        } else if self.right > other.right {
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
    if end_token.is_empty() {
        return Err(invalid("end_token must not be empty".to_string()));
    }
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
        .ok_or(invalid(format!("empty file")))? = file_size as usize;
    // slightly longer to make sure no end_token is skipped
    let buf_size = MINI_CHUNK_SIZE + end_token.len() - 1;
    let mut buffer = vec![0u8; buf_size];
    for b in &mut boundaries.iter_mut().skip(1) {
        let mut pos = *b;
        loop {
            let _ = file.seek(SeekFrom::Start(pos as u64))?;
            // read a minichunk
            let actual_lens = file.read(&mut buffer)?;
            if actual_lens == 0 {
                *b = file_size;
                break;
            }
            if let Some(idx) = buffer[..actual_lens]
                .windows(end_token.len())
                .position(|w| w == end_token)
            {
                *b = pos + idx;
                break;
            }
            pos += MINI_CHUNK_SIZE;
        }
    }
    Ok(boundaries)
}

fn pretokenize(
    path: &str,
    special_tokens: &Vec<String>,
    rev_vocab: &HashMap<Vec<u8>, u64, FixedHasher>,
) -> Result<HashMap<u64, PreState, FixedHasher>, std::io::Error> {
    use fancy_regex::{Regex, escape};
    use rayon::prelude::*;
    
    let file = File::open(path)?;
    let mut boundary_reader = BufReader::new(file);
    let boundaries =
        find_chunk_boundaries(&mut boundary_reader, NUM_PROCESSES, b"<|endoftext|>")?;
    let chunk_ranges = boundaries
        .windows(2)
        .map(|w| (w[0], w[1]))
        .filter(|(start, end)| start < end)
        .collect::<Vec<_>>();
    let spec_esc: Vec<String> = special_tokens
        .iter()
        .map(|x| escape(x).to_string())
        .collect();
    let pat = " ?(?:".to_string() + spec_esc.join("|").as_str() + ")(?:\\s)?|" + PAT;
    
    let local_counts = chunk_ranges
        .par_iter()
        .map(
            |(start, end)| -> Result<HashMap<Vec<u8>, u64, _>, std::io::Error> {
                let mut local = HashMap::<Vec<u8>, u64, _>::with_hasher(hasher());
                let mut reader = BufReader::new(File::open(path)?);
                let _ = reader.seek(SeekFrom::Start(*start as u64))?;
                let mut buffer = vec![0u8; end - start];
                let _ = reader.read(&mut buffer)?;
                let chunk =
                    str::from_utf8(&buffer).map_err(|_| invalid(format!("chunking")))?;
                let local_re = Regex::new(pat.as_str())
                    .map_err(|_| invalid("regex should compile".to_string()))?;
                for m in local_re.find_iter(chunk) {
                    if let Ok(m) = m
                        && !special_tokens.contains(&m.as_str().trim().to_string())
                    {
                        let entry =
                            local.entry(m.as_str().as_bytes().to_vec()).or_insert(0);
                        *entry += 1;
                    }
                }
                Ok(local)
            },
        )
        .collect::<Result<Vec<_>, _>>()?;

    let mut merged_counts = HashMap::<Vec<u8>, u64, _>::with_hasher(hasher());
    for local in local_counts {
        for (tok, tok_cnt) in local {
            let entry = merged_counts.entry(tok).or_insert(0);
            *entry += tok_cnt;
        }
    }

    // mapping Vec<u8> => id(u64)
    let mut pre_state: HashMap<u64, PreState, _> = HashMap::with_hasher(hasher());
    let mut pre_token_cnt: u64 = 0;
    for (tok, tok_cnt) in merged_counts {
        let mut state = PreState::from(tok.as_slice(), rev_vocab)
            .ok_or(invalid(format!("state not found")))?;
        state.cnt = tok_cnt;
        let _ = pre_state.insert(pre_token_cnt, state);
        pre_token_cnt += 1;
    }
    Ok(pre_state)
}

pub fn train(
    path: &str,
    vocab_size: usize,
    special_tokens: Vec<String>,
) -> Result<(HashMap<u64, Vec<u8>, FixedHasher>, Vec<(Vec<u8>, Vec<u8>)>), std::io::Error> {
    // byte group => its id
    let mut rev_vocab = {
        let mut v = HashMap::with_hasher(hasher());
        special_tokens.iter().enumerate().for_each(|(i, x)| {
            let _ = v.insert(x.as_bytes().to_vec(), i as u64);
        });
        (0..256).for_each(|i| {
            let _ = v.insert(vec![i as u8], (special_tokens.len() + i as usize) as u64);
        });
        v
    };
    let mut vocab = {
        let mut r = HashMap::with_hasher(hasher());
        rev_vocab.iter().for_each(|(k, v)| {
            let _ = r.insert(*v, k.clone());
        });
        r
    };

    eprintln!("[stage.pretokenize]");
    // mapping a pretoken (Vec<u8>) => pretoken_id(u64)
    let mut pretoken_state = pretokenize(path, &special_tokens, &rev_vocab)?;

    eprintln!("pretoken_dict size: {}", pretoken_state.len());
    eprintln!("[stage.build-heap]");

    let (mut group_pair_heap, mut group_pair_state) = {
        // vocab_id_pair => ByteGroupState
        let mut group_pair_state: HashMap<(u64, u64), GroupPairState, _> =
            HashMap::with_hasher(hasher());
        for (pre_id, pre) in &pretoken_state {
            // e.g. pretoken_id: 12, "word"
            for (x_idx, (x, y)) in pre.bytes.windows(2).map(|w| (w[0], w[1])).enumerate() {
                let id_x = *rev_vocab
                    .get([x].as_slice())
                    .ok_or(invalid(format!("id_x not found")))?;
                let id_y = *rev_vocab
                    .get([y].as_slice())
                    .ok_or(invalid(format!("id_y not found")))?;
                let state = group_pair_state
                    .entry((id_x, id_y))
                    .or_insert(GroupPairState {
                        cnt: 0,
                        pre_indices: HashSet::with_hasher(hasher()),
                        latest_ver: 0,
                    });
                state.cnt += pre.cnt;
                let _ = state.pre_indices.insert((*pre_id as u64, x_idx as i64));
            }
        }
        let mut heap: BinaryHeap<GroupPairCandidate> = BinaryHeap::new();
        for (vocab_ids, state) in group_pair_state.iter() {
            heap.push(GroupPairCandidate {
                ver: state.latest_ver,
                cnt: state.cnt,
                vocab_ids: *vocab_ids,
                left: vocab
                    .get(&vocab_ids.0)
                    .ok_or(invalid(format!("left not found")))?
                    .clone(),
                right: vocab
                    .get(&vocab_ids.1)
                    .ok_or(invalid(format!("right not found")))?
                    .clone(),
            });
        }
        (heap, group_pair_state)
    };
    // start merging!
    // 若在 heap 中，则 (id_x, id_y) 必然不在 vocab 中
    eprintln!("[stage.merge]");
    let mut merges: Vec<(Vec<u8>, Vec<u8>)> = vec![];
    while rev_vocab.len() < vocab_size {
        let Some(candidate) = group_pair_heap.pop() else {
            break;
        };

        let (x_id, y_id) = candidate.vocab_ids;
        let state = group_pair_state
            .get(&(x_id, y_id))
            .ok_or_else(|| invalid("not in `group_pair_state`".to_string()))?;
        if candidate.ver != state.latest_ver {
            continue;
        }

        let x = vocab.get(&x_id).ok_or(InvalidInput)?.clone();
        let y = vocab.get(&y_id).ok_or(InvalidInput)?.clone();
        let new_group = [x.clone(), y.clone()].concat();
        // let new_id = vocab.len() as u64;
        // let _ = vocab.insert(new_group.clone(), new_id);
        let new_id = rev_vocab.get(&new_group).copied().unwrap_or_else(|| {
            let new_id = rev_vocab.len() as u64;
            let _ = rev_vocab.insert(new_group.clone(), new_id);
            new_id
        });

        if candidate.cnt == 0 {
            eprintln!("skip, cnt == 0");
            continue;
        }

        // println!("insert new word: |{dbg_new_group}|");
        if let Some(_) = vocab.insert(new_id, new_group) {
            eprintln!("skip, inserted");
            continue;
        } else {
            let _ = merges.push((x.clone(), y.clone()));
            if merges.len() % 1 == 0 {
                eprintln!("merged: {}", merges.len());
            }
        }

        let state = state.clone();
        for (pretoken_id, x_idx) in &state.pre_indices {
            let (pretoken_id, x_idx) = (*pretoken_id, *x_idx);
            let y_idx = x_idx + x.len() as i64;
            let pretoken = pretoken_state
                .get_mut(&pretoken_id)
                .ok_or(invalid(format!("keys")))?;
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

            let pre_x_idx = *pretoken.pre.get(x_idx as usize).ok_or_else(|| {
                invalid(format!(
                    "pretoken pre_x_idx {} {}",
                    pretoken.bytes.len(),
                    x_idx
                ))
            })?;
            let nxt_y_idx = *pretoken.nxt.get(y_idx as usize).ok_or_else(|| {
                let dbg1 = str::from_utf8(&pretoken.bytes).unwrap_or("...");
                let dbg2 = str::from_utf8(&x).unwrap_or("...");
                invalid(format!(
                    "nxt_y_idx {}|{}|{}",
                    x_idx as usize + x.len(),
                    dbg1,
                    dbg2
                ))
            })?;

            fn modify_group_pair(
                state: &mut HashMap<(u64, u64), GroupPairState, FixedHasher>,
                heap: &mut BinaryHeap<GroupPairCandidate>,
                vocab: &HashMap<u64, Vec<u8>, FixedHasher>,
                vocab_ids: (u64, u64),
                pretoken_id: u64,
                pretoken_cnt: u64,
                idx_in_pretoken: i64,
                add: bool,
            ) -> Result<(), std::io::Error> {
                let state = state.entry(vocab_ids).or_insert_with(|| GroupPairState {
                    cnt: 0,
                    pre_indices: HashSet::with_hasher(hasher()),
                    latest_ver: 0,
                });
                if add {
                    let _ = state.pre_indices.insert((pretoken_id, idx_in_pretoken));
                    state.cnt += pretoken_cnt;
                } else {
                    state
                        .pre_indices
                        .remove(&(pretoken_id, idx_in_pretoken))
                        .then_some(())
                        .ok_or_else(|| invalid("remove failed".to_string()))?;
                    state.cnt -= pretoken_cnt;
                }
                state.latest_ver += 1;
                let _ = heap.push(GroupPairCandidate {
                    cnt: state.cnt,
                    ver: state.latest_ver,
                    vocab_ids: vocab_ids,
                    left: vocab.get(&vocab_ids.0).ok_or(InvalidInput)?.clone(),
                    right: vocab.get(&vocab_ids.1).ok_or(InvalidInput)?.clone(),
                });
                Ok(())
            }

            if pre_x_idx > -1 {
                // |      |     |
                // pre_x--x--y--nxt_y
                let pre_x_id = pretoken
                    .vocab_id_at
                    .get(pre_x_idx as usize)
                    .ok_or(InvalidInput)?
                    .ok_or(invalid(format!(concat!(
                        line!(),
                        ":vocab_id_at[] exists and is None",
                    ))))?;
                let new_vocab_id_pair = (pre_x_id, new_id);
                // add (pre_x, xy)
                modify_group_pair(
                    &mut group_pair_state,
                    &mut group_pair_heap,
                    &vocab,
                    new_vocab_id_pair,
                    pretoken_id,
                    pretoken.cnt,
                    pre_x_idx,
                    true,
                )?;
                modify_group_pair(
                    &mut group_pair_state,
                    &mut group_pair_heap,
                    &vocab,
                    (pre_x_id, x_id),
                    pretoken_id,
                    pretoken.cnt,
                    pre_x_idx,
                    false,
                )?;
            }
            if nxt_y_idx < pretoken.bytes.len() as i64 {
                // |      |     |
                // pre_x--x--y--nxt_y
                let nxt_y_id = pretoken
                    .vocab_id_at
                    .get(nxt_y_idx as usize)
                    .ok_or(InvalidInput)?
                    .ok_or(invalid(format!(concat!(
                        line!(),
                        ":vocab_id_at[] exists and is None",
                    ))))?;
                // add (xy, nxt_y)
                let new_vocab_id_pair = (new_id, nxt_y_id);
                modify_group_pair(
                    &mut group_pair_state,
                    &mut group_pair_heap,
                    &vocab,
                    new_vocab_id_pair,
                    pretoken_id,
                    pretoken.cnt,
                    x_idx,
                    true,
                )?;
                modify_group_pair(
                    &mut group_pair_state,
                    &mut group_pair_heap,
                    &vocab,
                    (y_id, nxt_y_id),
                    pretoken_id,
                    pretoken.cnt,
                    y_idx,
                    false,
                )?;

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
        }
    }
    Ok((vocab, merges))
}

fn hasher() -> BuildHasherDefault<SipHasher13> {
    BuildHasherDefault::<SipHasher13>::default()
}