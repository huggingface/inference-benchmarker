//! Deterministic answer checks for accuracy and conversation benchmarks.
//!
//! A [`Check`] is attached to a request and scored against the generated text.
//! Every check is a fixed rule: no model is used to judge answers.
use regex::Regex;
use serde::{Deserialize, Serialize};

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum Check {
    /// The whole answer equals `value` (trimmed, case-insensitive).
    Exact { value: String },
    /// The answer mentions `value` (case-insensitive). Used for context retention.
    Contains { value: String },
    /// The answer matches a regular expression.
    Regex { pattern: String },
    /// Multiple choice: the extracted letter equals `answer` (A, B, …).
    Choice { answer: String },
    /// Math: the final number equals `value` within `tolerance`.
    Number {
        value: f64,
        #[serde(default)]
        tolerance: f64,
    },
    /// Instruction following: the answer has at most `max` words.
    MaxWords { max: usize },
    /// Instruction following: the answer is valid JSON.
    Json,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct CheckOutcome {
    pub passed: bool,
    /// What was read from the answer (a letter, a number), when applicable.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub extracted: Option<String>,
}

impl CheckOutcome {
    fn new(passed: bool, extracted: Option<String>) -> Self {
        Self { passed, extracted }
    }
}

impl Check {
    /// Reject checks that can never be evaluated (bad regex, empty answer).
    pub fn validate(&self) -> anyhow::Result<()> {
        match self {
            Check::Regex { pattern } => {
                Regex::new(pattern).map_err(|e| anyhow::anyhow!("invalid regex: {e}"))?;
            }
            Check::Choice { answer } if !is_choice_letter(answer) => {
                return Err(anyhow::anyhow!("choice answer must be one letter A-Z"));
            }
            Check::Exact { value } | Check::Contains { value } if value.trim().is_empty() => {
                return Err(anyhow::anyhow!("check value must not be empty"));
            }
            Check::Number { tolerance, .. } if *tolerance < 0.0 => {
                return Err(anyhow::anyhow!("tolerance must not be negative"));
            }
            _ => {}
        }
        Ok(())
    }

    pub fn evaluate(&self, response: &str) -> CheckOutcome {
        match self {
            Check::Exact { value } => {
                CheckOutcome::new(response.trim().eq_ignore_ascii_case(value.trim()), None)
            }
            Check::Contains { value } => CheckOutcome::new(
                response
                    .to_lowercase()
                    .contains(&value.trim().to_lowercase()),
                None,
            ),
            Check::Regex { pattern } => CheckOutcome::new(
                Regex::new(pattern).is_ok_and(|re| re.is_match(response)),
                None,
            ),
            Check::Choice { answer } => {
                let got = extract_choice(response);
                let passed = got
                    .as_deref()
                    .is_some_and(|g| g.eq_ignore_ascii_case(answer.trim()));
                CheckOutcome::new(passed, got)
            }
            Check::Number { value, tolerance } => {
                let got = extract_number(response);
                let passed = got.is_some_and(|g| (g - value).abs() <= tolerance.max(1e-9));
                CheckOutcome::new(passed, got.map(|g| g.to_string()))
            }
            Check::MaxWords { max } => {
                let words = response.split_whitespace().count();
                CheckOutcome::new(words <= *max, Some(words.to_string()))
            }
            Check::Json => CheckOutcome::new(
                serde_json::from_str::<serde_json::Value>(strip_code_fence(response)).is_ok(),
                None,
            ),
        }
    }
}

fn is_choice_letter(s: &str) -> bool {
    let s = s.trim();
    s.len() == 1 && s.chars().all(|c| c.is_ascii_alphabetic())
}

/// Read the chosen letter from a multiple-choice answer.
///
/// Tries, in order: "answer is (B)" / "Answer: B", the last "(B)", then an
/// answer that starts with the letter ("B." / "B)"). Letters are uppercase so
/// prose such as "I think" or "a tricky one" is not read as a choice.
pub fn extract_choice(response: &str) -> Option<String> {
    let patterns = [
        r"(?i:answer)\s*(?i:is)?\s*[:：]?\s*\(?\s*([A-Z])\s*\)?(?:[^A-Za-z]|$)",
        r"\(([A-Z])\)",
        r"^\s*([A-Z])(?:[).:]|$)",
    ];
    patterns.iter().find_map(|p| {
        Regex::new(p)
            .ok()?
            .captures_iter(response)
            .last()
            .map(|c| c[1].to_string())
    })
}

/// Read the final number of a math answer: after `####` (GSM8K style), else
/// the last number in the text. Thousands separators are ignored.
pub fn extract_number(response: &str) -> Option<f64> {
    const NUM: &str = r"-?\d[\d,]*(?:\.\d+)?";
    let marked = Regex::new(&format!(r"####\s*\$?\s*({NUM})")).ok()?;
    let any = Regex::new(&format!("({NUM})")).ok()?;
    let raw = marked
        .captures_iter(response)
        .last()
        .or_else(|| any.captures_iter(response).last())?;
    raw[1].replace(',', "").parse().ok()
}

fn strip_code_fence(s: &str) -> &str {
    let t = s.trim();
    let Some(body) = t.strip_prefix("```") else {
        return t;
    };
    let body = body.split_once('\n').map_or("", |(_, rest)| rest);
    body.strip_suffix("```").unwrap_or(body).trim()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn choice_extraction() {
        assert_eq!(extract_choice("The answer is (C).").as_deref(), Some("C"));
        assert_eq!(extract_choice("Answer: B").as_deref(), Some("B"));
        assert_eq!(extract_choice("The answer is a tricky one"), None);
        assert_eq!(extract_choice("B. Because...").as_deref(), Some("B"));
        assert_eq!(
            extract_choice("(A) is wrong, so (D) it is").as_deref(),
            Some("D")
        );
        assert_eq!(extract_choice("I don't know"), None);
        let check = Check::Choice { answer: "C".into() };
        assert!(check.evaluate("so the answer is C").passed);
        assert!(!check.evaluate("so the answer is A").passed);
    }

    #[test]
    fn number_extraction() {
        assert_eq!(extract_number("so 3 + 4 = 7\n#### 7"), Some(7.0));
        assert_eq!(extract_number("It costs $1,250.50 in total"), Some(1250.5));
        assert_eq!(extract_number("#### 12 apples, not 15"), Some(12.0));
        assert_eq!(extract_number("no numbers"), None);
        let check = Check::Number {
            value: 18.0,
            tolerance: 0.0,
        };
        assert!(
            check
                .evaluate("She makes 9 * 2 = 18 dollars. #### 18")
                .passed
        );
        assert!(!check.evaluate("#### 19").passed);
    }

    #[test]
    fn text_checks() {
        let contains = Check::Contains {
            value: "Kathmandu".into(),
        };
        assert!(contains.evaluate("You said you live in kathmandu.").passed);
        assert!(!contains.evaluate("I don't remember.").passed);
        assert!(
            Check::Exact {
                value: "yes".into()
            }
            .evaluate(" Yes \n")
            .passed
        );
        assert!(Check::MaxWords { max: 3 }.evaluate("one two three").passed);
        assert!(!Check::MaxWords { max: 2 }.evaluate("one two three").passed);
        assert!(Check::Json.evaluate("```json\n{\"a\": 1}\n```").passed);
        assert!(!Check::Json.evaluate("{a: 1}").passed);
        let re = Check::Regex {
            pattern: r"^\d{4}-\d{2}-\d{2}$".into(),
        };
        assert!(re.evaluate("2026-10-20").passed);
    }

    #[test]
    fn validation() {
        assert!(Check::Regex {
            pattern: "(".into()
        }
        .validate()
        .is_err());
        assert!(Check::Choice {
            answer: "AB".into()
        }
        .validate()
        .is_err());
        assert!(Check::Contains { value: " ".into() }.validate().is_err());
        assert!(Check::Choice { answer: "a".into() }.validate().is_ok());
    }

    #[test]
    fn serde_shape() {
        let c: Check = serde_json::from_str(r#"{"kind":"number","value":7}"#).unwrap();
        assert_eq!(
            c,
            Check::Number {
                value: 7.0,
                tolerance: 0.0
            }
        );
    }
}
