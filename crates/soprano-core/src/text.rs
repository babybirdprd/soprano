//! Text normalization for Soprano TTS
//!
//! Ported from Python's soprano/utils/text.py
//! Normalizes text to a format that Soprano recognizes, including:
//! - Number expansion
//! - Abbreviation expansion
//! - Date/time formatting
//! - Special character replacement

use fancy_regex::Regex;
use lazy_static::lazy_static;

lazy_static! {
    // Abbreviations (case-insensitive with period)
    static ref ABBREVIATIONS: Vec<(Regex, &'static str)> = vec![
        (Regex::new(r"(?i)\bmrs\.").unwrap(), "misuss"),
        (Regex::new(r"(?i)\bms\.").unwrap(), "miss"),
        (Regex::new(r"(?i)\bmr\.").unwrap(), "mister"),
        (Regex::new(r"(?i)\bdr\.").unwrap(), "doctor"),
        (Regex::new(r"(?i)\bst\.").unwrap(), "saint"),
        (Regex::new(r"(?i)\bco\.").unwrap(), "company"),
        (Regex::new(r"(?i)\bjr\.").unwrap(), "junior"),
        (Regex::new(r"(?i)\bmaj\.").unwrap(), "major"),
        (Regex::new(r"(?i)\bgen\.").unwrap(), "general"),
        (Regex::new(r"(?i)\bdrs\.").unwrap(), "doctors"),
        (Regex::new(r"(?i)\brev\.").unwrap(), "reverend"),
        (Regex::new(r"(?i)\blt\.").unwrap(), "lieutenant"),
        (Regex::new(r"(?i)\bhon\.").unwrap(), "honorable"),
        (Regex::new(r"(?i)\bsgt\.").unwrap(), "sergeant"),
        (Regex::new(r"(?i)\bcapt\.").unwrap(), "captain"),
        (Regex::new(r"(?i)\besq\.").unwrap(), "esquire"),
        (Regex::new(r"(?i)\bltd\.").unwrap(), "limited"),
        (Regex::new(r"(?i)\bcol\.").unwrap(), "colonel"),
        (Regex::new(r"(?i)\bft\.").unwrap(), "fort"),
    ];

    // Case-sensitive abbreviations (no period)
    static ref CASED_ABBREVIATIONS: Vec<(Regex, &'static str)> = vec![
        (Regex::new(r"\bTTS\b").unwrap(), "text to speech"),
        (Regex::new(r"\bHz\b").unwrap(), "hertz"),
        (Regex::new(r"\bkHz\b").unwrap(), "kilohertz"),
        (Regex::new(r"\bKBs\b").unwrap(), "kilobytes"),
        (Regex::new(r"\bKB\b").unwrap(), "kilobyte"),
        (Regex::new(r"\bMBs\b").unwrap(), "megabytes"),
        (Regex::new(r"\bMB\b").unwrap(), "megabyte"),
        (Regex::new(r"\bGBs\b").unwrap(), "gigabytes"),
        (Regex::new(r"\bGB\b").unwrap(), "gigabyte"),
        (Regex::new(r"\bTBs\b").unwrap(), "terabytes"),
        (Regex::new(r"\bTB\b").unwrap(), "terabyte"),
        (Regex::new(r"\bAPIs\b").unwrap(), "a p i's"),
        (Regex::new(r"\bAPI\b").unwrap(), "a p i"),
        (Regex::new(r"\bCLIs\b").unwrap(), "c l i's"),
        (Regex::new(r"\bCLI\b").unwrap(), "c l i"),
        (Regex::new(r"\bCPUs\b").unwrap(), "c p u's"),
        (Regex::new(r"\bCPU\b").unwrap(), "c p u"),
        (Regex::new(r"\bGPUs\b").unwrap(), "g p u's"),
        (Regex::new(r"\bGPU\b").unwrap(), "g p u"),
        (Regex::new(r"\bAve\b").unwrap(), "avenue"),
        (Regex::new(r"\betc\b").unwrap(), "etcetera"),
    ];

    // Number patterns
    static ref NUM_PREFIX_RE: Regex = Regex::new(r"#\d").unwrap();
    static ref NUM_SUFFIX_RE: Regex = Regex::new(r"(?i)\d([KMBT])").unwrap();
    static ref NUM_LETTER_SPLIT_RE: Regex = Regex::new(r"(?i)(\d[a-z]|[a-z]\d)").unwrap();
    static ref COMMA_NUMBER_RE: Regex = Regex::new(r"(\d[\d,]+\d)").unwrap();
    static ref PHONE_NUMBER_RE: Regex = Regex::new(r"\(?\d{3}\)?[-.\s]\d{3}[-.\s]?\d{4}").unwrap();
    static ref TIME_RE: Regex = Regex::new(r"(\d\d?:\d\d(?::\d\d)?)").unwrap();
    static ref DOLLARS_RE: Regex = Regex::new(r"\$([\d.,]+)").unwrap();
    static ref POUNDS_RE: Regex = Regex::new(r"£([\d,]+)").unwrap();
    static ref ORDINAL_RE: Regex = Regex::new(r"(\d+)(st|nd|rd|th)").unwrap();
    static ref NUMBER_RE: Regex = Regex::new(r"\d+").unwrap();
    static ref MULTIPLY_RE: Regex = Regex::new(r"(\d)\s?\*\s?(\d)").unwrap();
    static ref DIVIDE_RE: Regex = Regex::new(r"(\d)\s?/\s?(\d)").unwrap();
    static ref ADD_RE: Regex = Regex::new(r"(\d)\s?\+\s?(\d)").unwrap();
    static ref SUBTRACT_RE: Regex = Regex::new(r"(\d)\s?-\s?(\d)").unwrap();

    // Special characters
    static ref SPECIAL_CHARS: Vec<(Regex, &'static str)> = vec![
        (Regex::new(r"@").unwrap(), " at "),
        (Regex::new(r"&").unwrap(), " and "),
        (Regex::new(r"%").unwrap(), " percent "),
        // (Regex::new(r":").unwrap(), "."),  // Disabled: interferes with times
        (Regex::new(r";").unwrap(), ","),
        // (Regex::new(r"\+").unwrap(), " plus "),  // Disabled: handled in numbers
        (Regex::new(r"\\").unwrap(), " backslash "),
        (Regex::new(r"~").unwrap(), " about "),
        (Regex::new(r"<=").unwrap(), " less than or equal to "),
        (Regex::new(r">=").unwrap(), " greater than or equal to "),
        (Regex::new(r"<").unwrap(), " less than "),
        (Regex::new(r">").unwrap(), " greater than "),
        (Regex::new(r"=").unwrap(), " equals "),
        // (Regex::new(r"/").unwrap(), " slash "),  // Disabled: interferes with dates
        (Regex::new(r"_").unwrap(), " "),
    ];

    // Cleanup patterns
    static ref WHITESPACE_RE: Regex = Regex::new(r"\s+").unwrap();
    static ref SPACE_PUNCT_RE: Regex = Regex::new(r" ([.?!,])").unwrap();
    static ref DASH_RE: Regex = Regex::new(r"\. - \.").unwrap();
    static ref DOT_ABBREV_RE: Regex = Regex::new(r"(?i)([A-Z])\.([A-Z])").unwrap();
    static ref ELLIPSIS_RE: Regex = Regex::new(r"\.\.\.+").unwrap();
    static ref MULTI_COMMA_RE: Regex = Regex::new(r",+").unwrap();
    static ref MULTI_PERIOD_RE: Regex = Regex::new(r"[.,]*\.[.,]*").unwrap();
    static ref MULTI_EXCLAIM_RE: Regex = Regex::new(r"[.,!]*![.,!]*").unwrap();
    static ref MULTI_QUESTION_RE: Regex = Regex::new(r"[.,!?]*\?[.,!?]*").unwrap();
}

/// Convert number to words (simplified version)
fn number_to_words(n: u64) -> String {
    let ones = [
        "",
        "one",
        "two",
        "three",
        "four",
        "five",
        "six",
        "seven",
        "eight",
        "nine",
        "ten",
        "eleven",
        "twelve",
        "thirteen",
        "fourteen",
        "fifteen",
        "sixteen",
        "seventeen",
        "eighteen",
        "nineteen",
    ];
    let tens = [
        "", "", "twenty", "thirty", "forty", "fifty", "sixty", "seventy", "eighty", "ninety",
    ];

    if n == 0 {
        return "zero".to_string();
    }

    if n < 20 {
        return ones[n as usize].to_string();
    }

    if n < 100 {
        let t = tens[(n / 10) as usize];
        let o = ones[(n % 10) as usize];
        if n % 10 == 0 {
            return t.to_string();
        }
        return format!("{} {}", t, o);
    }

    if n < 1000 {
        let h = ones[(n / 100) as usize];
        let rest = n % 100;
        if rest == 0 {
            return format!("{} hundred", h);
        }
        return format!("{} hundred {}", h, number_to_words(rest));
    }

    if n < 1_000_000 {
        let thousands = n / 1000;
        let rest = n % 1000;
        let t_str = number_to_words(thousands);
        if rest == 0 {
            return format!("{} thousand", t_str);
        }
        return format!("{} thousand {}", t_str, number_to_words(rest));
    }

    if n < 1_000_000_000 {
        let millions = n / 1_000_000;
        let rest = n % 1_000_000;
        let m_str = number_to_words(millions);
        if rest == 0 {
            return format!("{} million", m_str);
        }
        return format!("{} million {}", m_str, number_to_words(rest));
    }

    // For very large numbers, just spell out digits
    n.to_string()
        .chars()
        .map(|c| match c {
            '0' => "zero",
            '1' => "one",
            '2' => "two",
            '3' => "three",
            '4' => "four",
            '5' => "five",
            '6' => "six",
            '7' => "seven",
            '8' => "eight",
            '9' => "nine",
            _ => "",
        })
        .collect::<Vec<_>>()
        .join(" ")
}

/// Convert ordinal number (1st, 2nd, etc) to words
fn ordinal_to_words(n: u64) -> String {
    let base = number_to_words(n);
    if n % 100 >= 11 && n % 100 <= 13 {
        return format!("{}th", base);
    }
    match n % 10 {
        1 => format!("{}first", &base[..base.len().saturating_sub(3)].trim_end()),
        2 => format!("{}second", &base[..base.len().saturating_sub(3)].trim_end()),
        3 => format!("{}third", &base[..base.len().saturating_sub(5)].trim_end()),
        _ => format!("{}th", base),
    }
}

/// Expand abbreviations
fn expand_abbreviations(text: &str) -> String {
    let mut result = text.to_string();
    for (re, replacement) in ABBREVIATIONS.iter() {
        result = re.replace_all(&result, *replacement).to_string();
    }
    for (re, replacement) in CASED_ABBREVIATIONS.iter() {
        result = re.replace_all(&result, *replacement).to_string();
    }
    result
}

/// Expand special characters
fn expand_special_characters(text: &str) -> String {
    let mut result = text.to_string();
    for (re, replacement) in SPECIAL_CHARS.iter() {
        result = re.replace_all(&result, *replacement).to_string();
    }
    result
}

/// Normalize numbers in text
fn normalize_numbers(text: &str) -> String {
    let mut result = text.to_string();

    // Handle #1, #2 etc -> "number one"
    result = NUM_PREFIX_RE
        .replace_all(&result, |caps: &fancy_regex::Captures| {
            let s = caps.get(0).unwrap().as_str();
            format!("number {}", &s[1..])
        })
        .to_string();

    // Handle 100K, 5M, etc
    result = NUM_SUFFIX_RE
        .replace_all(&result, |caps: &fancy_regex::Captures| {
            let s = caps.get(0).unwrap().as_str();
            let digit = &s[..s.len() - 1];
            let suffix = s.chars().last().unwrap().to_ascii_uppercase();
            let word = match suffix {
                'K' => "thousand",
                'M' => "million",
                'B' => "billion",
                'T' => "trillion",
                _ => return s.to_string(),
            };
            format!("{} {}", digit, word)
        })
        .to_string();

    // Remove commas from numbers (1,234,567 -> 1234567)
    result = COMMA_NUMBER_RE
        .replace_all(&result, |caps: &fancy_regex::Captures| {
            caps.get(1).unwrap().as_str().replace(',', "")
        })
        .to_string();

    // Handle dollars
    result = DOLLARS_RE
        .replace_all(&result, |caps: &fancy_regex::Captures| {
            let amount = caps.get(1).unwrap().as_str().replace(',', "");
            let parts: Vec<&str> = amount.split('.').collect();
            let dollars: u64 = parts[0].parse().unwrap_or(0);
            let cents: u64 = if parts.len() > 1 {
                parts[1].parse().unwrap_or(0)
            } else {
                0
            };

            if dollars > 0 && cents > 0 {
                let d_unit = if dollars == 1 { "dollar" } else { "dollars" };
                let c_unit = if cents == 1 { "cent" } else { "cents" };
                format!(
                    "{} {}, {} {}",
                    number_to_words(dollars),
                    d_unit,
                    number_to_words(cents),
                    c_unit
                )
            } else if dollars > 0 {
                let d_unit = if dollars == 1 { "dollar" } else { "dollars" };
                format!("{} {}", number_to_words(dollars), d_unit)
            } else if cents > 0 {
                let c_unit = if cents == 1 { "cent" } else { "cents" };
                format!("{} {}", number_to_words(cents), c_unit)
            } else {
                "zero dollars".to_string()
            }
        })
        .to_string();

    // Handle pounds
    result = POUNDS_RE
        .replace_all(&result, |caps: &fancy_regex::Captures| {
            let amount: u64 = caps
                .get(1)
                .unwrap()
                .as_str()
                .replace(',', "")
                .parse()
                .unwrap_or(0);
            format!("{} pounds", number_to_words(amount))
        })
        .to_string();

    // Handle math operations - DISABLED for now as they interfere with dates
    // TODO: Implement proper date detection before enabling these
    // result = MULTIPLY_RE.replace_all(&result, "$1 times $2").to_string();
    // result = DIVIDE_RE.replace_all(&result, "$1 over $2").to_string();
    // result = ADD_RE.replace_all(&result, "$1 plus $2").to_string();
    // result = SUBTRACT_RE.replace_all(&result, "$1 minus $2").to_string();

    // Handle ordinals (1st, 2nd, 3rd, etc)
    result = ORDINAL_RE
        .replace_all(&result, |caps: &fancy_regex::Captures| {
            let n: u64 = caps.get(1).unwrap().as_str().parse().unwrap_or(0);
            ordinal_to_words(n)
        })
        .to_string();

    // Handle remaining plain numbers
    result = NUMBER_RE
        .replace_all(&result, |caps: &fancy_regex::Captures| {
            let n: u64 = caps.get(0).unwrap().as_str().parse().unwrap_or(0);
            number_to_words(n)
        })
        .to_string();

    result
}

/// Normalize newlines - ensure each line ends with proper punctuation
fn normalize_newlines(text: &str) -> String {
    text.lines()
        .filter(|line| !line.trim().is_empty())
        .map(|line| {
            let line = line.trim();
            if !line.ends_with('.') && !line.ends_with('!') && !line.ends_with('?') {
                format!("{}.", line)
            } else {
                line.to_string()
            }
        })
        .collect::<Vec<_>>()
        .join(" ")
}

/// Remove unknown characters, keeping only ASCII letters, numbers, and basic punctuation
fn remove_unknown_characters(text: &str) -> String {
    text.chars()
        .filter(|c| c.is_ascii_alphanumeric() || " !$%&'*+,-./0123456789?".contains(*c))
        .collect()
}

/// Collapse multiple whitespace into single space
fn collapse_whitespace(text: &str) -> String {
    let result = WHITESPACE_RE.replace_all(text, " ").to_string();
    // Remove space before punctuation
    SPACE_PUNCT_RE.replace_all(&result, "$1").to_string()
}

/// Deduplicate punctuation
fn dedup_punctuation(text: &str) -> String {
    let mut result = ELLIPSIS_RE.replace_all(text, "[ELLIPSIS]").to_string();
    result = MULTI_COMMA_RE.replace_all(&result, ",").to_string();
    result = MULTI_PERIOD_RE.replace_all(&result, ".").to_string();
    result = MULTI_EXCLAIM_RE.replace_all(&result, "!").to_string();
    result = MULTI_QUESTION_RE.replace_all(&result, "?").to_string();
    result = result.replace("[ELLIPSIS]", "...");
    result
}

/// Convert text to ASCII (basic transliteration)
fn convert_to_ascii(text: &str) -> String {
    // Simple ASCII conversion - remove diacritics
    text.chars()
        .map(|c| {
            if c.is_ascii() {
                c
            } else {
                // Basic transliteration for common characters
                match c {
                    'á' | 'à' | 'ä' | 'â' | 'ã' => 'a',
                    'é' | 'è' | 'ë' | 'ê' => 'e',
                    'í' | 'ì' | 'ï' | 'î' => 'i',
                    'ó' | 'ò' | 'ö' | 'ô' | 'õ' => 'o',
                    'ú' | 'ù' | 'ü' | 'û' => 'u',
                    'ñ' => 'n',
                    'ç' => 'c',
                    '—' | '–' => '-',
                    // Curly quotes (using Unicode escapes)
                    '\u{2018}' | '\u{2019}' => '\'', // ' '
                    '\u{201C}' | '\u{201D}' => '"',  // " "
                    _ => ' ',                        // Replace unknown with space
                }
            }
        })
        .collect()
}

/// Main text cleaning function - full normalization pipeline
/// Matches Python's clean_text() in soprano/utils/text.py
pub fn clean_text(text: &str) -> String {
    let text = convert_to_ascii(text);
    let text = normalize_newlines(&text);
    let text = normalize_numbers(&text);
    let text = expand_abbreviations(&text);
    let text = expand_special_characters(&text);
    let text = text.to_lowercase();
    let text = remove_unknown_characters(&text);
    let text = collapse_whitespace(&text);
    let text = dedup_punctuation(&text);
    text.trim().to_string()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_number_to_words() {
        assert_eq!(number_to_words(0), "zero");
        assert_eq!(number_to_words(1), "one");
        assert_eq!(number_to_words(12), "twelve");
        assert_eq!(number_to_words(21), "twenty one");
        assert_eq!(number_to_words(100), "one hundred");
        assert_eq!(number_to_words(123), "one hundred twenty three");
        assert_eq!(number_to_words(1000), "one thousand");
        assert_eq!(
            number_to_words(1234),
            "one thousand two hundred thirty four"
        );
    }

    #[test]
    fn test_dollars() {
        let result = clean_text("$2.47");
        assert!(result.contains("two dollars"));
        assert!(result.contains("forty seven cents"));
    }

    #[test]
    fn test_abbreviations() {
        let result = clean_text("Mr. Smith went to Dr. Jones");
        assert!(result.contains("mister"));
        assert!(result.contains("doctor"));
    }

    #[test]
    fn test_special_chars() {
        let result = clean_text("100% of users @ home");
        assert!(result.contains("percent"));
        assert!(result.contains("at"));
    }

    #[test]
    fn test_tech_abbreviations() {
        let result = clean_text("The GPU and CPU are fast");
        assert!(result.contains("g p u"));
        assert!(result.contains("c p u"));
    }
}
