//! The little JSON reader the GLB parser needs (no dependency for one format).
//! Nesting is bounded, so hostile input cannot overflow the stack.

use std::collections::BTreeMap;

use super::{Error, Result};

const MAX_DEPTH: usize = 64;

/// A JSON value.
#[derive(Debug, Clone, PartialEq)]
pub(super) enum Value {
    Null,
    Bool,
    Number(f64),
    String(String),
    Array(Vec<Value>),
    Object(BTreeMap<String, Value>),
}

impl Value {
    /// A member of an object; `None` on anything else.
    pub(super) fn get(&self, key: &str) -> Option<&Value> {
        match self {
            Value::Object(members) => members.get(key),
            _ => None,
        }
    }

    pub(super) fn as_array(&self) -> Option<&[Value]> {
        match self {
            Value::Array(items) => Some(items),
            _ => None,
        }
    }

    pub(super) fn as_f64(&self) -> Option<f64> {
        match self {
            Value::Number(n) => Some(*n),
            _ => None,
        }
    }

    pub(super) fn as_str(&self) -> Option<&str> {
        match self {
            Value::String(s) => Some(s),
            _ => None,
        }
    }

    /// A number that is a whole count or index, exact in `f64`.
    pub(super) fn as_index(&self) -> Option<u64> {
        const LIMIT: f64 = 9_007_199_254_740_992.0; // 2^53
        let n = self.as_f64()?;
        ((0.0..LIMIT).contains(&n) && n.fract() == 0.0).then_some(n as u64)
    }
}

/// Parses a whole JSON text.
pub(super) fn parse(bytes: &[u8]) -> Result<Value> {
    let mut p = Parser { bytes, at: 0 };
    let value = p.value(0)?;
    p.skip_space();
    if p.at != bytes.len() {
        return Err(p.error("trailing characters"));
    }
    Ok(value)
}

struct Parser<'a> {
    bytes: &'a [u8],
    at: usize,
}

impl Parser<'_> {
    fn error(&self, what: &str) -> Error {
        Error::new(format!("{what} at byte {}", self.at))
    }

    fn skip_space(&mut self) {
        while matches!(self.bytes.get(self.at), Some(b' ' | b'\t' | b'\n' | b'\r')) {
            self.at += 1;
        }
    }

    fn literal(&mut self, word: &[u8], value: Value) -> Result<Value> {
        if self.bytes[self.at..].starts_with(word) {
            self.at += word.len();
            Ok(value)
        } else {
            Err(self.error("unexpected token"))
        }
    }

    fn value(&mut self, depth: usize) -> Result<Value> {
        if depth > MAX_DEPTH {
            return Err(self.error("nested too deep"));
        }
        self.skip_space();
        match self.bytes.get(self.at) {
            None => Err(self.error("unexpected end")),
            Some(b'n') => self.literal(b"null", Value::Null),
            Some(b't') => self.literal(b"true", Value::Bool),
            Some(b'f') => self.literal(b"false", Value::Bool),
            Some(b'"') => Ok(Value::String(self.string()?)),
            Some(b'[') => self.array(depth),
            Some(b'{') => self.object(depth),
            Some(_) => self.number(),
        }
    }

    fn array(&mut self, depth: usize) -> Result<Value> {
        self.at += 1;
        let mut items = Vec::new();
        self.skip_space();
        if self.bytes.get(self.at) == Some(&b']') {
            self.at += 1;
            return Ok(Value::Array(items));
        }
        loop {
            items.push(self.value(depth + 1)?);
            self.skip_space();
            match self.bytes.get(self.at) {
                Some(b',') => self.at += 1,
                Some(b']') => {
                    self.at += 1;
                    return Ok(Value::Array(items));
                }
                _ => return Err(self.error("expected , or ]")),
            }
        }
    }

    fn object(&mut self, depth: usize) -> Result<Value> {
        self.at += 1;
        let mut members = BTreeMap::new();
        self.skip_space();
        if self.bytes.get(self.at) == Some(&b'}') {
            self.at += 1;
            return Ok(Value::Object(members));
        }
        loop {
            self.skip_space();
            if self.bytes.get(self.at) != Some(&b'"') {
                return Err(self.error("expected a member name"));
            }
            let key = self.string()?;
            self.skip_space();
            if self.bytes.get(self.at) != Some(&b':') {
                return Err(self.error("expected :"));
            }
            self.at += 1;
            members.insert(key, self.value(depth + 1)?);
            self.skip_space();
            match self.bytes.get(self.at) {
                Some(b',') => self.at += 1,
                Some(b'}') => {
                    self.at += 1;
                    return Ok(Value::Object(members));
                }
                _ => return Err(self.error("expected , or }")),
            }
        }
    }

    fn number(&mut self) -> Result<Value> {
        let start = self.at;
        while matches!(
            self.bytes.get(self.at),
            Some(b'0'..=b'9' | b'-' | b'+' | b'.' | b'e' | b'E')
        ) {
            self.at += 1;
        }
        std::str::from_utf8(&self.bytes[start..self.at])
            .ok()
            .and_then(|s| s.parse().ok())
            .map(Value::Number)
            .ok_or_else(|| self.error("bad number"))
    }

    fn string(&mut self) -> Result<String> {
        self.at += 1;
        let mut out = Vec::new();
        loop {
            let Some(&b) = self.bytes.get(self.at) else {
                return Err(self.error("unterminated string"));
            };
            self.at += 1;
            match b {
                b'"' => {
                    return String::from_utf8(out).map_err(|_| self.error("invalid UTF-8"));
                }
                b'\\' => {
                    let Some(&e) = self.bytes.get(self.at) else {
                        return Err(self.error("unterminated string"));
                    };
                    self.at += 1;
                    match e {
                        b'"' | b'\\' | b'/' => out.push(e),
                        b'b' => out.push(8),
                        b'f' => out.push(12),
                        b'n' => out.push(b'\n'),
                        b'r' => out.push(b'\r'),
                        b't' => out.push(b'\t'),
                        b'u' => {
                            let c = self.unicode_escape()?;
                            out.extend_from_slice(c.encode_utf8(&mut [0; 4]).as_bytes());
                        }
                        _ => return Err(self.error("bad escape")),
                    }
                }
                _ => out.push(b),
            }
        }
    }

    fn hex4(&mut self) -> Result<u32> {
        let digits = self
            .bytes
            .get(self.at..self.at + 4)
            .ok_or_else(|| self.error("bad \\u escape"))?;
        let value = std::str::from_utf8(digits)
            .ok()
            .and_then(|s| u32::from_str_radix(s, 16).ok())
            .ok_or_else(|| self.error("bad \\u escape"))?;
        self.at += 4;
        Ok(value)
    }

    /// A `\uXXXX`, with its low surrogate when it starts a pair; a lone
    /// surrogate becomes U+FFFD (names only matter in messages).
    fn unicode_escape(&mut self) -> Result<char> {
        let first = self.hex4()?;
        if (0xD800..0xDC00).contains(&first) && self.bytes[self.at..].starts_with(b"\\u") {
            let back = self.at;
            self.at += 2;
            let second = self.hex4()?;
            if (0xDC00..0xE000).contains(&second) {
                let code = 0x10000 + ((first - 0xD800) << 10) + (second - 0xDC00);
                return Ok(char::from_u32(code).unwrap_or('\u{FFFD}'));
            }
            self.at = back;
        }
        Ok(char::from_u32(first).unwrap_or('\u{FFFD}'))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn values() {
        let v =
            parse(br#" {"a":[1,-2.5e1,true,false,null],"b":{"c":"x\n\u00e9\ud83d\ude00\ud800"}} "#)
                .unwrap();
        let a = v.get("a").unwrap().as_array().unwrap();
        assert_eq!(a[0].as_index(), Some(1));
        assert_eq!(a[1].as_f64(), Some(-25.0));
        assert_eq!(a[1].as_index(), None);
        assert_eq!(a[4], Value::Null);
        let c = v.get("b").unwrap().get("c").unwrap().as_str().unwrap();
        assert_eq!(c, "x\n\u{e9}\u{1F600}\u{FFFD}");
        assert_eq!(parse(b"[]").unwrap(), Value::Array(vec![]));
        assert_eq!(parse(b"{}").unwrap().get("x"), None);
    }

    #[test]
    fn refusals() {
        for bad in [
            &b""[..],
            b"{",
            b"[1,",
            b"[1 2]",
            b"{\"a\" 1}",
            b"{a:1}",
            b"\"abc",
            b"\"\\q\"",
            b"\"\\u12\"",
            b"nul",
            b"1 2",
            b"--1",
            b"\"\xff\"",
        ] {
            assert!(parse(bad).is_err(), "{bad:?}");
        }
    }

    #[test]
    fn nesting_is_bounded() {
        let deep = vec![b'['; 100_000];
        assert!(parse(&deep).unwrap_err().to_string().contains("deep"));
    }
}
