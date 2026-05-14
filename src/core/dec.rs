use super::{FixedHasher, PAT, hasher, invalid};
use fancy_regex::Regex;
use std::cmp::Ordering;
use std::collections::{BinaryHeap, HashMap, HashSet};
use std::fs::File;
use std::io::Read;

#[derive(Debug, Clone)]
struct DecodePreState {
    bytes: Vec<u8>,
    pre: Vec<i64>,
    nxt: Vec<i64>,
    vocab_id_at: Vec<Option<u64>>,
}

impl DecodePreState {
    fn from(
        s: &[u8],
        rev_vocab: &HashMap<Vec<u8>, u64, FixedHasher>,
    ) -> Result<Self, std::io::Error> {
        let len = s.len() as i64;
        let pre = (0..len).map(|i| i - 1).collect();
        let nxt = (0..len).map(|i| i + 1).collect();
        let vocab_id_at = s
            .iter()
            .map(|b| rev_vocab.get([*b].as_slice()).copied())
            .collect::<Option<Vec<_>>>()
            .ok_or_else(|| invalid("byte is not in vocab".to_string()))?
            .into_iter()
            .map(Some)
            .collect();
        Ok(Self {
            bytes: s.to_vec(),
            pre,
            nxt,
            vocab_id_at,
        })
    }
}

#[derive(PartialEq, Eq, Clone, Debug)]
struct MergeCandidate {
    rank: usize,
    idx_in_pretoken: i64,
    left_id: u64,
    right_id: u64,
}

impl PartialOrd for MergeCandidate {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for MergeCandidate {
    fn cmp(&self, other: &Self) -> Ordering {
        // Reverse rank for min-heap behavior.
        if self.rank > other.rank {
            Ordering::Less
        } else if self.rank < other.rank {
            Ordering::Greater
        } else if self.idx_in_pretoken > other.idx_in_pretoken {
            Ordering::Less
        } else if self.idx_in_pretoken < other.idx_in_pretoken {
            Ordering::Greater
        } else {
            Ordering::Equal
        }
    }
}

fn build_pattern() -> Result<Regex, std::io::Error> {
    Regex::new(PAT).map_err(|e| invalid(e.to_string()))
}

#[derive(Debug, Clone)]
pub struct TokenizerCore {
    vocab: HashMap<u64, Vec<u8>, FixedHasher>,
    rev_vocab: HashMap<Vec<u8>, u64, FixedHasher>,
    merge_ranks: HashMap<(u64, u64), usize, FixedHasher>,
    special_token_set: HashSet<Vec<u8>, FixedHasher>,
    special_tokens_sorted: Vec<String>,
    pretoken_regex: Regex,
}

impl TokenizerCore {
    pub fn new(
        mut vocab: HashMap<u64, Vec<u8>, FixedHasher>,
        merges: Vec<(Vec<u8>, Vec<u8>)>,
        special_tokens: Option<Vec<String>>,
    ) -> Result<Self, std::io::Error> {
        let mut rev_vocab = HashMap::<Vec<u8>, u64, _>::with_hasher(hasher());
        for (id, token) in &vocab {
            let _ = rev_vocab.insert(token.clone(), *id);
        }

        let special_tokens = special_tokens.unwrap_or_default();
        if !special_tokens.is_empty() {
            let mut max_id = vocab.keys().copied().max().unwrap_or(0);
            for st in &special_tokens {
                let b = st.as_bytes().to_vec();
                if !rev_vocab.contains_key(&b) {
                    max_id += 1;
                    let _ = vocab.insert(max_id, b.clone());
                    let _ = rev_vocab.insert(b, max_id);
                }
            }
        }

        let mut merge_ranks = HashMap::<(u64, u64), usize, _>::with_hasher(hasher());
        for (rank, (left, right)) in merges.into_iter().enumerate() {
            let left_id = *rev_vocab
                .get(&left)
                .ok_or_else(|| invalid("merge left token not in vocab".to_string()))?;
            let right_id = *rev_vocab
                .get(&right)
                .ok_or_else(|| invalid("merge right token not in vocab".to_string()))?;
            let _ = merge_ranks.insert((left_id, right_id), rank);
        }

        let mut special_token_set = HashSet::<Vec<u8>, _>::with_hasher(hasher());
        for st in &special_tokens {
            let _ = special_token_set.insert(st.as_bytes().to_vec());
        }
        let mut special_tokens_sorted = special_tokens.clone();
        special_tokens_sorted.sort_by_key(|x| std::cmp::Reverse(x.len()));
        let pretoken_regex = build_pattern()?;

        Ok(Self {
            vocab,
            rev_vocab,
            merge_ranks,
            special_token_set,
            special_tokens_sorted,
            pretoken_regex,
        })
    }

    fn encode_pretoken_bytes(&self, bytes: &[u8]) -> Result<Vec<u64>, std::io::Error> {
        if bytes.is_empty() {
            return Ok(vec![]);
        }
        let mut pretoken = DecodePreState::from(bytes, &self.rev_vocab)?;
        let mut heap: BinaryHeap<MergeCandidate> = BinaryHeap::new();

        let push_candidate =
            |i: i64, heap: &mut BinaryHeap<MergeCandidate>, pretoken: &DecodePreState| {
                if i < 0 || i as usize >= pretoken.bytes.len() {
                    return;
                }
                let j = pretoken.nxt[i as usize];
                if j < 0 || j as usize >= pretoken.bytes.len() {
                    return;
                }
                let left_id = match pretoken.vocab_id_at[i as usize] {
                    Some(v) => v,
                    None => return,
                };
                let right_id = match pretoken.vocab_id_at[j as usize] {
                    Some(v) => v,
                    None => return,
                };
                if let Some(rank) = self.merge_ranks.get(&(left_id, right_id)) {
                    heap.push(MergeCandidate {
                        rank: *rank,
                        idx_in_pretoken: i,
                        left_id,
                        right_id,
                    });
                }
            };

        for i in 0..(pretoken.bytes.len() as i64 - 1) {
            push_candidate(i, &mut heap, &pretoken);
        }

        while let Some(candidate) = heap.pop() {
            let x_idx = candidate.idx_in_pretoken;
            if x_idx < 0 || x_idx as usize >= pretoken.bytes.len() {
                continue;
            }
            let y_idx = pretoken.nxt[x_idx as usize];
            if y_idx < 0 || y_idx as usize >= pretoken.bytes.len() {
                continue;
            }

            let Some(actual_x) = pretoken.vocab_id_at[x_idx as usize] else {
                continue;
            };
            let Some(actual_y) = pretoken.vocab_id_at[y_idx as usize] else {
                continue;
            };
            if actual_x != candidate.left_id || actual_y != candidate.right_id {
                continue;
            }
            let Some(actual_rank) = self.merge_ranks.get(&(actual_x, actual_y)) else {
                continue;
            };
            if *actual_rank != candidate.rank {
                continue;
            }

            let merged = [
                self.vocab
                    .get(&actual_x)
                    .ok_or_else(|| invalid("left id missing in vocab".to_string()))?
                    .clone(),
                self.vocab
                    .get(&actual_y)
                    .ok_or_else(|| invalid("right id missing in vocab".to_string()))?
                    .clone(),
            ]
            .concat();
            let merged_id = *self
                .rev_vocab
                .get(&merged)
                .ok_or_else(|| invalid("merged token not found in vocab".to_string()))?;

            let pre_x = pretoken.pre[x_idx as usize];
            let nxt_y = pretoken.nxt[y_idx as usize];

            pretoken.vocab_id_at[x_idx as usize] = Some(merged_id);
            pretoken.vocab_id_at[y_idx as usize] = None;
            pretoken.nxt[x_idx as usize] = nxt_y;
            if nxt_y >= 0 && (nxt_y as usize) < pretoken.bytes.len() {
                pretoken.pre[nxt_y as usize] = x_idx;
            }

            if pre_x >= 0 {
                push_candidate(pre_x, &mut heap, &pretoken);
            }
            push_candidate(x_idx, &mut heap, &pretoken);
        }

        let mut out = vec![];
        let mut idx = 0_i64;
        loop {
            if idx < 0 || idx as usize >= pretoken.bytes.len() {
                break;
            }
            if let Some(id) = pretoken.vocab_id_at[idx as usize] {
                out.push(id);
            }
            let next = pretoken.nxt[idx as usize];
            if next <= idx {
                break;
            }
            idx = next;
        }
        Ok(out)
    }

    fn encode_match(&self, token: &str) -> Result<Vec<u64>, std::io::Error> {
        if token.is_empty() {
            return Ok(vec![]);
        }
        if self.special_token_set.contains(token.as_bytes()) {
            let id = *self
                .rev_vocab
                .get(token.as_bytes())
                .ok_or_else(|| invalid("special token missing in vocab".to_string()))?;
            return Ok(vec![id]);
        }
        self.encode_pretoken_bytes(token.as_bytes())
    }

    fn encode_normal_text(&self, text: &str) -> Result<Vec<u64>, std::io::Error> {
        let mut out = Vec::new();
        for m in self.pretoken_regex.find_iter(text) {
            let m = m.map_err(|e| invalid(e.to_string()))?;
            out.extend(self.encode_match(m.as_str())?);
        }
        Ok(out)
    }

    pub fn encode(&self, text: &str) -> Result<Vec<u64>, std::io::Error> {
        if self.special_tokens_sorted.is_empty() {
            return self.encode_normal_text(text);
        }

        let mut out = Vec::new();
        let mut i = 0usize;
        while i < text.len() {
            let mut best: Option<(usize, &str)> = None;
            for st in &self.special_tokens_sorted {
                if let Some(rel) = text[i..].find(st.as_str()) {
                    let pos = i + rel;
                    match best {
                        None => best = Some((pos, st.as_str())),
                        Some((best_pos, best_st)) => {
                            if pos < best_pos || (pos == best_pos && st.len() > best_st.len()) {
                                best = Some((pos, st.as_str()));
                            }
                        }
                    }
                }
            }

            let Some((pos, st)) = best else {
                out.extend(self.encode_normal_text(&text[i..])?);
                break;
            };
            if pos > i {
                out.extend(self.encode_normal_text(&text[i..pos])?);
            }
            let id = *self
                .rev_vocab
                .get(st.as_bytes())
                .ok_or_else(|| invalid("special token missing in vocab".to_string()))?;
            out.push(id);
            i = pos + st.len();
        }
        Ok(out)
    }

    pub fn encode_file(&self, path: &str) -> Result<Vec<u64>, std::io::Error> {
        let mut file = File::open(path)?;
        let mut text = String::new();
        let _ = file.read_to_string(&mut text)?;
        self.encode(text.as_str())
    }

    pub fn encode_file_u16(&self, path: &str) -> Result<Vec<u16>, std::io::Error> {
        let ids = self.encode_file(path)?;
        let mut out = Vec::with_capacity(ids.len());
        for id in ids {
            if id > u16::MAX as u64 {
                return Err(invalid(format!("token id {} exceeds uint16 range", id)));
            }
            out.push(id as u16);
        }
        Ok(out)
    }

    pub fn encode_iterable<I, S>(&self, iterable: I) -> Result<Vec<u64>, std::io::Error>
    where
        I: IntoIterator<Item = S>,
        S: AsRef<str>,
    {
        let mut out = Vec::new();
        let mut carry = String::new();

        for chunk in iterable {
            carry.push_str(chunk.as_ref());
            if carry.is_empty() {
                continue;
            }

            let mut matches: Vec<(usize, usize)> = vec![];
            for m in self.pretoken_regex.find_iter(carry.as_str()) {
                let m = m.map_err(|e| invalid(e.to_string()))?;
                matches.push((m.start(), m.end()));
            }
            if matches.is_empty() {
                continue;
            }
            if matches.len() == 1 && matches[0].0 == 0 && matches[0].1 == carry.len() {
                continue;
            }

            let last = *matches
                .last()
                .ok_or_else(|| invalid("regex match should exist".to_string()))?;
            let mut prev_end = 0usize;
            for (s, e) in matches.into_iter() {
                if s != prev_end {
                    return Err(invalid("pretokenization gap".to_string()));
                }
                if e > carry.len() {
                    return Err(invalid("pretokenization index out-of-bounds".to_string()));
                }
                if (s, e) != last {
                    out.extend(self.encode_match(&carry[s..e])?);
                }
                prev_end = e;
            }
            carry = carry[last.0..last.1].to_string();
        }

        if !carry.is_empty() {
            out.extend(self.encode(carry.as_str())?);
        }
        Ok(out)
    }

    pub fn decode(&self, ids: &[u64]) -> Result<String, std::io::Error> {
        let mut bytes = vec![];
        for id in ids {
            let tok = self
                .vocab
                .get(id)
                .ok_or_else(|| invalid(format!("token id {} not in vocab", id)))?;
            bytes.extend(tok);
        }
        Ok(String::from_utf8_lossy(bytes.as_slice()).to_string())
    }
}

#[test]
fn test_tokenizer_core_encode_decode_roundtrip() -> Result<(), std::io::Error> {
    let mut vocab = HashMap::with_hasher(hasher());
    let _ = vocab.insert(0, b" ".to_vec());
    let _ = vocab.insert(1, b"a".to_vec());
    let _ = vocab.insert(2, b"c".to_vec());
    let _ = vocab.insert(3, b"e".to_vec());
    let _ = vocab.insert(4, b"h".to_vec());
    let _ = vocab.insert(5, b"t".to_vec());
    let _ = vocab.insert(6, b"th".to_vec());
    let _ = vocab.insert(7, b" c".to_vec());
    let _ = vocab.insert(8, b" a".to_vec());
    let _ = vocab.insert(9, b"the".to_vec());
    let _ = vocab.insert(10, b" at".to_vec());
    let merges = vec![
        (b"t".to_vec(), b"h".to_vec()),
        (b" ".to_vec(), b"c".to_vec()),
        (b" ".to_vec(), b"a".to_vec()),
        (b"th".to_vec(), b"e".to_vec()),
        (b" a".to_vec(), b"t".to_vec()),
    ];
    let tk = TokenizerCore::new(vocab, merges, None)?;
    let ids = tk.encode("the cat ate")?;
    assert_eq!(ids, vec![9, 7, 1, 5, 10, 3]);
    assert_eq!(tk.decode(&ids)?, "the cat ate");
    Ok(())
}

#[test]
fn test_tokenizer_core_special_tokens_and_overlaps() -> Result<(), std::io::Error> {
    let mut vocab = HashMap::with_hasher(hasher());
    for i in 0..256_u64 {
        let _ = vocab.insert(i, vec![i as u8]);
    }
    let merges = vec![];
    let tk = TokenizerCore::new(
        vocab,
        merges,
        Some(vec![
            "<|endoftext|>".to_string(),
            "<|endoftext|><|endoftext|>".to_string(),
        ]),
    )?;
    let ids = tk.encode("x<|endoftext|><|endoftext|>y<|endoftext|>")?;
    let decoded = tk.decode(&ids)?;
    assert_eq!(decoded, "x<|endoftext|><|endoftext|>y<|endoftext|>");
    Ok(())
}

#[test]
fn test_tokenizer_core_decode_invalid_utf8_replacement() -> Result<(), std::io::Error> {
    let mut vocab = HashMap::with_hasher(hasher());
    let _ = vocab.insert(0, vec![0xff]);
    let tk = TokenizerCore::new(vocab, vec![], None)?;
    let s = tk.decode(&[0])?;
    assert_eq!(s, "\u{fffd}");
    Ok(())
}
