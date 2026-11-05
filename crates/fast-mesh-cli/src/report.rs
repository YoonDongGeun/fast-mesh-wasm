//! A command's report as a small tree, printed as text or as JSON.

use std::fmt::Write as _;

/// A value in a report.
pub enum Value {
    Null,
    Bool(bool),
    Count(usize),
    Number(f64),
    Text(String),
    Object(Vec<(&'static str, Value)>),
}

impl Value {
    /// The report as one line of JSON.
    pub fn to_json(&self) -> String {
        let mut out = String::new();
        self.write_json(&mut out);
        out
    }

    fn write_json(&self, out: &mut String) {
        match self {
            Value::Null => out.push_str("null"),
            Value::Bool(b) => out.push_str(if *b { "true" } else { "false" }),
            Value::Count(n) => {
                let _ = write!(out, "{n}");
            }
            // JSON has no NaN or infinity.
            Value::Number(x) if !x.is_finite() => out.push_str("null"),
            Value::Number(x) => {
                let _ = write!(out, "{x}");
            }
            Value::Text(s) => write_json_string(s, out),
            Value::Object(members) => {
                out.push('{');
                for (i, (name, value)) in members.iter().enumerate() {
                    if i > 0 {
                        out.push(',');
                    }
                    write_json_string(name, out);
                    out.push(':');
                    value.write_json(out);
                }
                out.push('}');
            }
        }
    }

    /// The report as `name: value` lines, nested members indented.
    pub fn to_text(&self) -> String {
        let mut out = String::new();
        self.write_text(0, &mut out);
        out
    }

    fn write_text(&self, depth: usize, out: &mut String) {
        let Value::Object(members) = self else {
            return;
        };
        for (name, value) in members {
            let _ = write!(out, "{:width$}{name}:", "", width = 2 * depth);
            match value {
                Value::Object(_) => {
                    out.push('\n');
                    value.write_text(depth + 1, out);
                }
                Value::Null => out.push_str(" none\n"),
                Value::Bool(b) => {
                    let _ = writeln!(out, " {b}");
                }
                Value::Count(n) => {
                    let _ = writeln!(out, " {n}");
                }
                Value::Number(x) => {
                    let _ = writeln!(out, " {x}");
                }
                Value::Text(s) => {
                    let _ = writeln!(out, " {s}");
                }
            }
        }
    }
}

fn write_json_string(s: &str, out: &mut String) {
    out.push('"');
    for c in s.chars() {
        match c {
            '"' => out.push_str("\\\""),
            '\\' => out.push_str("\\\\"),
            '\n' => out.push_str("\\n"),
            '\r' => out.push_str("\\r"),
            '\t' => out.push_str("\\t"),
            c if u32::from(c) < 0x20 => {
                let _ = write!(out, "\\u{:04x}", u32::from(c));
            }
            c => out.push(c),
        }
    }
    out.push('"');
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sample() -> Value {
        Value::Object(vec![
            ("path", Value::Text("C:\\a \"b\"\n\u{1}".into())),
            ("n", Value::Count(3)),
            ("x", Value::Number(1.5)),
            ("bad", Value::Number(f64::NAN)),
            ("none", Value::Null),
            ("inner", Value::Object(vec![("ok", Value::Bool(true))])),
        ])
    }

    #[test]
    fn json() {
        assert_eq!(
            sample().to_json(),
            r#"{"path":"C:\\a \"b\"\n\u0001","n":3,"x":1.5,"bad":null,"none":null,"inner":{"ok":true}}"#
        );
    }

    #[test]
    fn text() {
        let text = sample().to_text();
        assert!(text.contains("n: 3\n"), "{text}");
        assert!(text.contains("none: none\n"), "{text}");
        assert!(text.contains("inner:\n  ok: true\n"), "{text}");
    }
}
