fn main() {
}

#[test]
fn test_maps() -> Result<(), fancy_regex::Error> {
    use fancy_regex::{Regex, escape};
    let pat = r" ?(?:<\|endoftext\|>|whoami233)|'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+";
    let re = Regex::new(pat)?;
    let t = "hello, who are you? <|endoftext|> whoami233 who";
    for m in re.find_iter(t) {
        if let Ok(m) = m {
            println!("{} ", m.as_str());
        }
    }
    Ok(())
}

#[test]
fn test1() {
    let s = "hello, who are you?  whoami233 who".to_string();
    let t = "1".to_string() + s.as_str();
    println!("t: {t}");
}
