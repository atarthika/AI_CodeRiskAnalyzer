#!/usr/bin/env python3
"""
AI-Powered Software Vulnerability & Code Risk Analyzer  (CLI only)
Languages: C/C++, Python, SQL

  python analyzer.py scan <folder_or_file> [--json report.json] [--min-severity low]
  python analyzer.py train <dataset.csv> [--code-col COL] [--label-col COL] [--langs cpp]
  python analyzer.py fetch <kaggle-name | local-folder> [--train] [--langs sql]
  python analyzer.py trainall --cpp <file|folder> --sql <file|folder>

Each finding reports: kind, severity, file:line, code, explanation, remediation.
Risk score (0-100) = 50% AI model + 50% rule engine (rules only if no model yet
or the file's language is not covered by the model).
"""
import argparse
import json
import re
import sys
from pathlib import Path

MODEL_DIR = Path("models")  # one trained model file per language set
MAX_FILE_BYTES = 1_000_000
WINDOW, STRIDE = 30, 15

SKIP_DIRS = {".git", "node_modules", "venv", ".venv", "env", "__pycache__", "dist",
             "build", "target", ".idea", ".vscode", "vendor", "site-packages", "third_party"}
EXT_LANG = {
    ".c": "cpp", ".h": "cpp", ".cpp": "cpp", ".cc": "cpp", ".cxx": "cpp",
    ".hpp": "cpp", ".hxx": "cpp", ".hh": "cpp",
    ".py": "python",
    ".sql": "sql",
}
COMMENT_PREFIX = {"cpp": ("//", "/*", "*"), "python": ("#",), "sql": ("--", "/*", "*")}
SEV_ORDER = {"critical": 4, "high": 3, "medium": 2, "low": 1}
SEV_WEIGHT = {"critical": 1.0, "high": 0.75, "medium": 0.45, "low": 0.2}

ALL = {"cpp", "python", "sql"}
CPP, PY, SQL = {"cpp"}, {"python"}, {"sql"}

# (id, langs, severity, kind, regex, title, CWE, explanation, remediation)
RULES = [
    # ---------------- C / C++ ----------------
    ("CPP_GETS", CPP, "critical", "vulnerability", r"\bgets\s*\(",
     "Use of gets()", "CWE-242",
     "gets() reads input with no length limit, so any long input overflows the buffer.",
     "Use fgets(buf, sizeof(buf), stdin) or std::getline."),
    ("CPP_UNBOUNDED_COPY", CPP, "high", "vulnerability", r"\b(strcpy|strcat|stpcpy|wcscpy|wcscat)\s*\(",
     "Unbounded string copy", "CWE-120",
     "These functions do not check destination size; oversized input causes a buffer overflow (memory corruption, code execution).",
     "Use strncpy/strlcpy/snprintf with sizeof(dest), or std::string."),
    ("CPP_SPRINTF", CPP, "high", "vulnerability", r"\b(sprintf|vsprintf)\s*\(",
     "sprintf without size limit", "CWE-120",
     "sprintf writes as much as the formatted data needs, regardless of buffer size.",
     "Use snprintf(buf, sizeof(buf), ...)."),
    ("CPP_SCANF_S", CPP, "high", "vulnerability", r"\b(scanf|fscanf|sscanf)\s*\(\s*[^;]*%s",
     "scanf %s without width", "CWE-120",
     "%s with no field width lets the user write past the end of the buffer.",
     'Use a width, e.g. scanf("%63s", buf), or fgets/std::cin >> std::string.'),
    ("CPP_FORMAT_STRING", CPP, "high", "vulnerability", r"\b(printf|vprintf|syslog)\s*\(\s*[A-Za-z_]\w*\s*\)",
     "Format string from a variable", "CWE-134",
     "A variable used as the format string lets an attacker inject %x/%n and read or write memory.",
     'Use a constant format: printf("%s", var).'),
    ("CPP_CMD_INJECT", CPP, "critical", "vulnerability", r"\b(system|popen|_popen|WinExec)\s*\(",
     "OS command execution", "CWE-78",
     "Passing input to a shell lets an attacker append commands (e.g. '; rm -rf /').",
     "Avoid the shell; use execv with a fixed argument list and validate/whitelist input."),
    ("CPP_EXEC", CPP, "medium", "suspicious", r"\b(execl|execlp|execle|execv|execvp|execve|ShellExecute[AW]?)\s*\(",
     "Process execution", "CWE-78",
     "Launching external programs is risky if the path or arguments come from user input.",
     "Hard-code the program path and validate every argument."),
    ("CPP_ALLOCA", CPP, "medium", "vulnerability", r"\balloca\s*\(",
     "alloca() stack allocation", "CWE-770",
     "Attacker-controlled sizes can exhaust the stack and crash or hijack the program.",
     "Use heap allocation (std::vector / malloc) with a size limit."),
    ("CPP_TEMPFILE", CPP, "medium", "vulnerability", r"\b(tmpnam|tempnam|mktemp)\s*\(",
     "Insecure temporary file", "CWE-377",
     "Predictable names allow race conditions (symlink attacks).",
     "Use mkstemp() or std::filesystem with a random name."),
    ("CPP_RAND", CPP, "low", "vulnerability", r"\b(rand|srand)\s*\(",
     "Weak random number generator", "CWE-338",
     "rand() is predictable and unsuitable for tokens, keys or passwords.",
     "Use std::random_device / getrandom / OpenSSL RAND_bytes for security uses."),
    ("CPP_ATOI", CPP, "low", "suspicious", r"\b(atoi|atol|atof)\s*\(",
     "atoi without error checking", "CWE-190",
     "No error or overflow detection, which can lead to integer-overflow bugs.",
     "Use strtol/std::stoi with range and error checks."),
    ("CPP_MEMCPY", CPP, "low", "suspicious", r"\b(memcpy|memmove)\s*\(",
     "memcpy/memmove (check the length)", "CWE-119",
     "If the length is derived from user input, it can overflow the destination.",
     "Validate length <= sizeof(destination) before copying."),
    ("CPP_PROC_INJECT", CPP, "high", "harmful", r"\b(CreateRemoteThread|VirtualAllocEx|WriteProcessMemory|NtUnmapViewOfSection)\b",
     "Process-injection API", "CWE-506",
     "Typical of malware that injects code into other processes.",
     "Remove unless this is a documented, audited security tool."),
    ("CPP_KEYLOG", CPP, "medium", "harmful", r"\b(SetWindowsHookEx[AW]?|GetAsyncKeyState|GetKeyboardState)\b",
     "Keyboard hooking / key capture", "CWE-506",
     "Can be used to capture keystrokes (keylogger behaviour).",
     "Remove, or clearly document and restrict the legitimate purpose."),
    ("CPP_SQL_CONCAT", CPP, "high", "vulnerability",
     r'(?i)("[^"]*\b(select|insert\s+into|update|delete\s+from)\b[^"]*"\s*(\+|<<)\s*\w)|(\b(sprintf|snprintf|strcat)\s*\([^;]*\b(select|insert|update|delete)\b)',
     "SQL query built by concatenation", "CWE-89",
     "User input mixed into the query text lets an attacker change the SQL (SQL injection).",
     "Use prepared statements with bound parameters (sqlite3_bind_*, mysql_stmt_*, etc.)."),

    # ---------------- Python ----------------
    ("PY_EVAL", PY, "critical", "vulnerability", r"(?<![\w.])(eval|exec)\s*\(",
     "eval()/exec() code execution", "CWE-95",
     "Executes arbitrary Python; with user-controlled input this is full remote code execution.",
     "Use ast.literal_eval for data, or a dispatch dict for allowed actions."),
    ("PY_CMD_INJECT", PY, "critical", "vulnerability",
     r"(os\.system\s*\(|os\.popen\s*\(|shell\s*=\s*True|commands\.getoutput\s*\()",
     "Shell command execution", "CWE-78",
     "Commands run through a shell can be hijacked by input such as '; rm -rf ~'.",
     "Use subprocess.run([...], shell=False) with an argument list; validate inputs."),
    ("PY_DESERIALIZE", PY, "high", "vulnerability",
     r"(pickle\.loads?\s*\(|cPickle\.loads?\s*\(|marshal\.loads?\s*\(|yaml\.load\s*\((?![^)]*Safe)|shelve\.open\s*\()",
     "Insecure deserialization", "CWE-502",
     "Unpickling or unsafe YAML loading of untrusted data executes attacker code.",
     "Use json, or yaml.safe_load; never unpickle untrusted data."),
    ("PY_SQL_INJECT", PY, "high", "vulnerability",
     r"""\.(execute|executemany|executescript|raw)\s*\(\s*(f["']|["'][^"']*["']\s*(%|\+|\.format)|\w+\s*\+)""",
     "SQL built with string formatting", "CWE-89",
     "Query text assembled from variables allows SQL injection.",
     'Use parameters: cursor.execute("SELECT * FROM t WHERE id = %s", (uid,)).'),
    ("PY_TLS_OFF", PY, "high", "vulnerability", r"(verify\s*=\s*False|_create_unverified_context|CERT_NONE)",
     "TLS certificate verification disabled", "CWE-295",
     "Connections can be intercepted by a man-in-the-middle attacker.",
     "Keep verification on; supply a CA bundle if you use a private CA."),
    ("PY_WEAK_HASH", PY, "medium", "vulnerability", r"hashlib\.(md5|sha1)\s*\(",
     "Weak hash function", "CWE-328",
     "MD5/SHA-1 are broken for security; passwords hashed this way are easy to crack.",
     "Use hashlib.sha256+, and bcrypt/argon2/scrypt for passwords."),
    ("PY_DEBUG", PY, "medium", "vulnerability", r"(\.run\s*\([^)]*debug\s*=\s*True|^\s*DEBUG\s*=\s*True)",
     "Debug mode enabled", "CWE-489",
     "The Werkzeug/Django debugger exposes code and may allow remote execution.",
     "Disable debug in production; read it from an environment variable."),
    ("PY_TEMPFILE", PY, "medium", "vulnerability", r"tempfile\.mktemp\s*\(",
     "Insecure temporary file", "CWE-377",
     "mktemp() returns a name only, creating a race condition.",
     "Use tempfile.NamedTemporaryFile or mkstemp."),
    ("PY_TAR", PY, "medium", "vulnerability", r"\.extractall\s*\(",
     "Archive extraction (path traversal)", "CWE-22",
     "Malicious archives can write files outside the target folder.",
     "Validate member paths, or use filter='data' (Python 3.12+)."),
    ("PY_SSTI", PY, "medium", "vulnerability", r"(render_template_string\s*\(|autoescape\s*=\s*False|Markup\s*\()",
     "Template injection / XSS risk", "CWE-79",
     "Rendering user-controlled strings as templates or disabling escaping enables XSS/SSTI.",
     "Use render_template with files and keep autoescape on."),
    ("PY_RANDOM", PY, "low", "vulnerability", r"\brandom\.(random|randint|choice|randrange)\s*\(",
     "Non-cryptographic random", "CWE-338",
     "The random module is predictable; do not use it for tokens or passwords.",
     "Use the secrets module."),
    ("PY_BIND_ALL", PY, "low", "suspicious", r"""["']0\.0\.0\.0["']""",
     "Binds to all network interfaces", "CWE-1327",
     "The service becomes reachable from every network.",
     "Bind to 127.0.0.1 unless public access is required."),
    ("PY_NO_TIMEOUT", PY, "low", "suspicious", r"requests\.(get|post|put|delete|head)\s*\((?!.*timeout)",
     "HTTP request without timeout", "CWE-400",
     "A slow or hostile server can hang your program forever.",
     "Pass timeout=10 (or similar)."),
    ("PY_DESTRUCTIVE", PY, "medium", "harmful", r"(shutil\.rmtree\s*\(|os\.(remove|unlink|rmdir|removedirs)\s*\()",
     "Destructive file operation", "CWE-73",
     "Deletes files/folders; dangerous if the path is user-controlled or wrong.",
     "Validate and restrict the path, and ask for confirmation / use a trash folder."),
    ("PY_OBFUSCATION", PY, "high", "harmful",
     r"(base64\.b64decode|zlib\.decompress|codecs\.decode|bytes\.fromhex)\s*\(.*\)\s*.*(exec|eval)|(exec|eval)\s*\(.*(b64decode|decompress|codecs\.decode)",
     "Obfuscated code execution", "CWE-506",
     "Decoding a payload and executing it is a common malware technique.",
     "Remove it; ship readable code."),
    ("PY_KEYLOG", PY, "medium", "harmful", r"(from\s+pynput|import\s+pynput|keyboard\.(hook|on_press|record)\s*\()",
     "Keyboard capture library", "CWE-506",
     "Can record keystrokes (keylogger behaviour).",
     "Remove, or document and restrict the legitimate use."),

    # ---------------- SQL ----------------
    ("SQL_XP_CMDSHELL", SQL, "critical", "harmful", r"(?i)\bxp_cmdshell\b",
     "xp_cmdshell (OS commands from SQL)", "CWE-78",
     "Lets SQL Server run operating-system commands; a favourite for attackers.",
     "Disable xp_cmdshell and remove its usage."),
    ("SQL_DYNAMIC", SQL, "high", "vulnerability",
     r"(?i)(\bEXEC(UTE)?\s*\(|\bEXECUTE\s+IMMEDIATE\b|\bsp_executesql\b|\bPREPARE\s+\w+\s+FROM\s+@)",
     "Dynamic SQL execution", "CWE-89",
     "SQL assembled as a string and executed can be injected if any part comes from input.",
     "Use parameterised sp_executesql / bound parameters; avoid string concatenation."),
    ("SQL_GRANT_ALL", SQL, "high", "vulnerability", r"(?i)(\bGRANT\s+ALL\b|\bTO\s+PUBLIC\b|WITH\s+GRANT\s+OPTION)",
     "Over-broad privileges", "CWE-269",
     "Gives users more rights than needed, widening the damage of any compromise.",
     "Apply least privilege: grant only the specific rights required."),
    ("SQL_PLAIN_PASS", SQL, "high", "vulnerability",
     r"""(?i)(IDENTIFIED\s+BY|\bPASSWORD\s*=?)\s*['"][^'"]+['"]""",
     "Plain-text password in SQL", "CWE-798",
     "Credentials stored in scripts leak via version control and backups.",
     "Use secret management / environment variables; rotate this password."),
    ("SQL_PAYLOAD", ALL, "high", "suspicious",
     r"""(?i)(\bOR\s+['"]?1['"]?\s*=\s*['"]?1\b|\bUNION\s+(ALL\s+)?SELECT\b|;\s*DROP\s+TABLE|'\s*--\s*$)""",
     "SQL-injection payload pattern", "CWE-89",
     "Looks like an injection string (tautology/UNION/stacked query), or code vulnerable to one.",
     "Use parameterised queries and validate input; remove test payloads from source."),
    ("SQL_DESTRUCTIVE", SQL, "medium", "harmful",
     r"(?i)\b(DROP\s+(TABLE|DATABASE|SCHEMA)|TRUNCATE\s+TABLE)\b",
     "Destructive statement", "CWE-285",
     "Permanently deletes data; dangerous if run by mistake or in production.",
     "Wrap in a transaction, back up first, use IF EXISTS and restrict who can run it."),
    ("SQL_SELECT_STAR", SQL, "low", "suspicious", r"(?i)\bSELECT\s+\*\s+FROM\b",
     "SELECT * (data over-exposure)", "CWE-200",
     "May return sensitive columns (password hashes, tokens) that callers should not see.",
     "List only the columns you need."),

    # ---------------- All languages ----------------
    ("SECRET", ALL, "high", "vulnerability",
     r"""(?i)\b(password|passwd|pwd|secret|api[_-]?key|access[_-]?token|auth[_-]?token)\b\s*[:=]\s*["'][^"'\s]{4,}["']""",
     "Hard-coded credential/secret", "CWE-798",
     "Secrets in source code leak through repositories and can be reused by anyone.",
     "Load from environment variables or a secrets manager; rotate the exposed value."),
    ("AWS_KEY", ALL, "critical", "vulnerability", r"AKIA[0-9A-Z]{16}",
     "AWS access key in code", "CWE-798",
     "A valid-looking AWS key is committed; attackers scan repos for these.",
     "Revoke the key now and use IAM roles / environment credentials."),
    ("PRIVATE_KEY", ALL, "critical", "vulnerability", r"-----BEGIN (RSA |EC |DSA |OPENSSH )?PRIVATE KEY-----",
     "Private key in source", "CWE-321",
     "Anyone with the code can impersonate the key owner.",
     "Remove it, rotate the key pair, and load keys from secure storage."),
    ("REVERSE_SHELL", ALL, "critical", "harmful",
     r"(/bin/(ba)?sh\s+-i|\bnc\s+(-e|-c)\s|bash\s+-i\s*>&|/dev/tcp/)",
     "Reverse-shell pattern", "CWE-506",
     "Opens an interactive shell to a remote machine, typical of backdoors.",
     "Remove it and investigate how it entered the codebase."),
    ("DESTRUCTIVE_CMD", ALL, "high", "harmful", r"(rm\s+-rf\s+(/|~|\*)|mkfs\.\w+\s+/dev|dd\s+if=.*of=/dev/|format\s+c:)",
     "Destructive shell command", "CWE-506",
     "Can wipe files or disks.",
     "Remove, or guard with strict path validation and confirmation."),
]
COMPILED = [(i, l, s, k, re.compile(p), t, c, e, r) for i, l, s, k, p, t, c, e, r in RULES]

FREE_RE = re.compile(r"\bfree\s*\(\s*(\w+)\s*\)|\bdelete(?:\s*\[\s*\])?\s+(\w+)\s*;")


# ------------------------------------------------------------------ helpers
def detect_lang(p: Path):
    return EXT_LANG.get(p.suffix.lower())


def iter_files(target: Path):
    if target.is_file():
        if detect_lang(target):
            yield target
        return
    for p in target.rglob("*"):
        if any(part in SKIP_DIRS for part in p.parts):
            continue
        if p.is_file() and detect_lang(p):
            try:
                if p.stat().st_size <= MAX_FILE_BYTES:
                    yield p
            except OSError:
                continue


def read_text(p: Path):
    try:
        data = p.read_bytes()
    except OSError:
        return None
    if b"\x00" in data[:2048]:
        return None
    return data.decode("utf-8", errors="ignore")


def make_finding(n, line, rule, kind=None):
    rid, _, sev, k, _, title, cwe, why, fix = rule
    return {"rule": rid, "kind": kind or k, "severity": sev, "cwe": cwe, "title": title,
            "line": n, "code": line.strip()[:140], "explanation": why, "remediation": fix}


def rule_scan(lang, lines):
    findings, seen = [], set()
    in_block = False
    for n, line in enumerate(lines, 1):
        s = line.strip()
        if lang in ("cpp", "sql"):  # block comments
            if in_block:
                if "*/" in s:
                    in_block = False
                continue
            if s.startswith("/*") and "*/" not in s:
                in_block = True
                continue
        if not s or s.startswith(COMMENT_PREFIX[lang]):
            continue
        for rule in COMPILED:
            if lang in rule[1] and rule[4].search(line):
                key = (rule[0], n)
                if key not in seen:
                    seen.add(key)
                    findings.append(make_finding(n, line, rule))

    if lang == "cpp":
        findings += cpp_memory_checks(lines)
    if lang == "sql":
        findings += sql_unsafe_dml(lines)
    return findings


def cpp_memory_checks(lines):
    """Heuristic use-after-free / double-free detection."""
    out, freed = [], {}
    for n, line in enumerate(lines, 1):
        code = line.split("//")[0]
        if code.strip().startswith(("*", "/*")):
            continue
        m = FREE_RE.search(code)
        now = (m.group(1) or m.group(2)) if m else None
        for var in list(freed):
            if n - freed[var] > 20:
                del freed[var]
                continue
            if re.search(rf"\b{re.escape(var)}\s*=(?!=)", code):
                del freed[var]
                continue
            if var == now:
                out.append(_mem("CPP_DOUBLE_FREE", "Possible double free", "CWE-415",
                                "The same pointer is freed twice, corrupting the heap.",
                                "Set the pointer to nullptr after free, or use smart pointers.", n, line))
                del freed[var]
            elif re.search(rf"\b{re.escape(var)}\b", code):
                out.append(_mem("CPP_UAF", "Possible use after free", "CWE-416",
                                f"'{var}' is used after being freed; attackers can exploit this for code execution.",
                                "Set the pointer to nullptr after free and never touch it again; prefer smart pointers.",
                                n, line))
                del freed[var]
        if now and now not in freed:
            freed[now] = n
    return out


def _mem(rid, title, cwe, why, fix, n, line):
    return {"rule": rid, "kind": "vulnerability", "severity": "high", "cwe": cwe, "title": title,
            "line": n, "code": line.strip()[:140], "explanation": why, "remediation": fix}


def sql_unsafe_dml(lines):
    out = []
    rx = re.compile(r"(?i)^\s*(DELETE\s+FROM\s+[\w.\[\]\"`]+|UPDATE\s+[\w.\[\]\"`]+\s+SET\b[^;]*)\s*;?\s*$")
    for n, line in enumerate(lines, 1):
        if rx.match(line) and not re.search(r"(?i)\bWHERE\b", line):
            out.append({"rule": "SQL_NO_WHERE", "kind": "harmful", "severity": "high", "cwe": "CWE-285",
                        "title": "DELETE/UPDATE without WHERE", "line": n, "code": line.strip()[:140],
                        "explanation": "Affects every row in the table; one mistake wipes or corrupts all data.",
                        "remediation": "Add a WHERE clause, test with SELECT first, and run inside a transaction."})
    return out


def rule_score(findings):
    safe = 1.0
    for f in findings:
        safe *= 1 - 0.6 * SEV_WEIGHT[f["severity"]]
    return 1 - safe


def level(score):
    return "HIGH" if score >= 70 else "MEDIUM" if score >= 40 else "LOW"


# ----------------------------------------------------------------- AI model
def ml_scan(model, lines):
    chunks, starts = [], []
    for i in range(0, max(1, len(lines) - WINDOW + STRIDE), STRIDE):
        c = "\n".join(lines[i:i + WINDOW])
        if c.strip():
            chunks.append(c)
            starts.append(i + 1)
    if not chunks:
        return 0.0, []
    probs = model.predict_proba(chunks)[:, 1]
    ranked = sorted(zip(probs, starts), reverse=True)
    return float(ranked[0][0]), [(float(p), s) for p, s in ranked[:3]]


def make_pipeline():
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import Pipeline
    return Pipeline([
        ("tfidf", TfidfVectorizer(token_pattern=r"[A-Za-z_][A-Za-z_0-9]*|[^\sA-Za-z_0-9]",
                                  ngram_range=(1, 3), max_features=200_000,
                                  sublinear_tf=True, lowercase=False)),
        ("clf", LogisticRegression(max_iter=1000, class_weight="balanced")),
    ])


CODE_COLS = ["query", "func", "func_before", "code", "source", "source_code", "function",
             "vulnerable_code", "content", "text", "snippet", "sentence", "statement"]
LABEL_COLS = ["target", "label", "vul", "vulnerable", "is_vulnerable", "vulnerability",
              "malicious", "is_malicious", "class", "y"]
POS_WORDS = {"1", "true", "yes", "vulnerable", "vuln", "bad", "insecure", "unsafe",
             "malicious", "attack", "sqli", "injection", "sql injection"}
DATA_EXTS = {".csv", ".tsv", ".json", ".jsonl", ".parquet"}


def find_col(df, wanted, candidates):
    if wanted:
        return wanted if wanted in df.columns else None
    lower = {str(c).lower(): c for c in df.columns}
    for c in candidates:
        if c in lower:
            return lower[c]
    return None


def guess_columns(df):
    """Find the code column and the label column, even with unusual names."""
    code = find_col(df, None, CODE_COLS)
    if code is None:  # fall back to the column with the longest text
        best, best_len = None, 0
        for c in df.columns:
            if df[c].dtype == object:
                ln = df[c].dropna().astype(str).str.len().mean()
                if ln and ln > best_len:
                    best, best_len = c, ln
        if best is not None and best_len >= 15:
            code = best
    label = find_col(df, None, LABEL_COLS)
    if label is None:  # fall back to a two-valued column
        for c in df.columns:
            if c != code and 1 < df[c].nunique(dropna=True) <= 2:
                label = c
                break
    return code, label


def to_binary(v):
    s = str(v).strip().lower()
    if s in POS_WORDS:
        return 1
    try:
        return 1 if float(s) > 0 else 0
    except ValueError:
        return 0


def read_json_flexible(path):
    """Read a JSON array, JSON Lines, or {"key": [records]} file into a DataFrame."""
    import json
    import pandas as pd
    raw = Path(path).read_text(encoding="utf-8-sig", errors="replace")
    stripped = raw.lstrip()
    if not stripped:
        sys.exit(f"{path} is empty. Download it again.")
    if stripped[:1] == "<":
        sys.exit(f"{path} is an HTML page, not JSON (the download failed). Download it again.")
    try:
        data = json.loads(raw)
    except json.JSONDecodeError:
        data = []
        for n, line in enumerate(raw.splitlines(), 1):
            line = line.strip()
            if not line:
                continue
            try:
                data.append(json.loads(line))
            except json.JSONDecodeError as e:
                sys.exit(f"{path} is not valid JSON (line {n}: {e.msg}). "
                         "The file may be damaged or cut off; download it again.")
    if isinstance(data, dict):
        for v in data.values():
            if isinstance(v, list) and v and isinstance(v[0], dict):
                data = v
                break
    return pd.DataFrame(data)


def load_table(path, nrows=None):
    import pandas as pd
    path = Path(path)
    ext = path.suffix.lower()
    if ext in (".json", ".jsonl"):
        df = read_json_flexible(path)
        return df.head(nrows) if nrows else df
    if ext == ".parquet":
        df = pd.read_parquet(path)
        return df.head(nrows) if nrows else df
    return pd.read_csv(path, sep="\t" if ext == ".tsv" else ",", nrows=nrows,
                       on_bad_lines="skip", encoding_errors="replace")


def load_labeled(path, code_col, label_col, max_rows):
    """Read a table and return a DataFrame with columns: code, label (0/1)."""
    import pandas as pd
    df = load_table(path, max_rows)
    for name in (code_col, label_col):
        if name and name not in df.columns:
            sys.exit(f"Column '{name}' not found in {path}. Columns: {list(df.columns)}")
    g_code, g_label = guess_columns(df)
    code_col, label_col = code_col or g_code, label_col or g_label
    if not code_col or not label_col:
        sys.exit(f"Could not detect code/label columns in {path}. "
                 f"Use --code-col and --label-col. Columns: {list(df.columns)}")
    print(f"File: {path}\nUsing code column '{code_col}' and label column '{label_col}'")
    df = df[[code_col, label_col]].dropna()
    print("Label values:", df[label_col].value_counts().head(6).to_dict())
    return pd.DataFrame({"code": df[code_col].astype(str), "label": df[label_col].map(to_binary)})


def find_usable_tables(root):
    out = []
    for f in sorted(Path(root).rglob("*")):
        if f.is_file() and f.suffix.lower() in DATA_EXTS:
            try:
                code, label = guess_columns(load_table(f, 300))
            except (Exception, SystemExit):
                continue
            if code is not None and label is not None:
                out.append((f, code, label, f.stat().st_size))
    return out


SRC_EXTS = {".c", ".h", ".cpp", ".cc", ".cxx", ".hpp"}
NEG_TOK = {"safe", "secure", "good", "fixed", "patched", "patch", "clean", "benign",
           "nonvulnerable", "after"}
POS_TOK = {"vulnerable", "vulnerability", "vulnerabilities", "vuln", "vulns", "bad",
           "insecure", "unsafe", "before", "exploit"}


def label_from_text(text):
    low = text.lower()
    if re.search(r"(non|not|no)[_\- ]?vuln", low):
        return 0
    toks = set(re.split(r"[^a-z0-9]+", low))
    if toks & NEG_TOK:
        return 0
    if toks & POS_TOK:
        return 1
    return None


def source_files_dataset(root, max_rows):
    """Build a labeled dataset from raw source files, using folder / file names as labels."""
    import random
    import pandas as pd
    files = [f for f in Path(root).rglob("*")
             if f.is_file() and f.suffix.lower() in SRC_EXTS and f.stat().st_size < 500_000]
    if not files:
        sys.exit(f"No labeled table and no C/C++ source files found in {root}.")
    random.Random(42).shuffle(files)
    rows, examples, folders = [], {0: [], 1: []}, set()
    for f in files:
        rel = f.relative_to(root)
        folders.add(rel.parts[0] if len(rel.parts) > 1 else "(top level)")
        lab = None
        for part in rel.parts[:-1]:          # folder names first
            lab = label_from_text(part)
            if lab is not None:
                break
        if lab is None:                      # then the file name
            lab = label_from_text(f.stem)
        if lab is None:
            continue
        text = f.read_text(encoding="utf-8", errors="replace")[:6000]
        if not text.strip():
            continue
        rows.append((text, lab))
        if len(examples[lab]) < 3:
            examples[lab].append(str(rel))
        if len(rows) >= max_rows:
            break
    df = pd.DataFrame(rows, columns=["code", "label"])
    print(f"Built dataset from source files: {len(df)} files | "
          f"vulnerable(1): {int(df['label'].sum()) if len(df) else 0} | "
          f"safe(0): {int((1 - df['label']).sum()) if len(df) else 0}")
    for lab, name in ((1, "vulnerable"), (0, "safe")):
        print(f"  e.g. {name}: {examples[lab]}")
    if df.empty or df["label"].nunique() < 2:
        sys.exit("Could not infer both classes from folder/file names. Folders seen: "
                 f"{sorted(folders)[:15]}. Name them like 'vulnerable'/'safe' (or 'bad'/'good'), "
                 "or send me the folder names and I'll adapt the code.")
    return df


def resolve_dataset(path, code_col, label_col, max_rows):
    path = Path(path)
    if not path.exists():
        sys.exit(f"Not found: {path}")
    if path.is_dir():
        usable = find_usable_tables(path)
        if usable:
            best = max(usable, key=lambda u: ("train" in u[0].name.lower(), u[3]))
            return load_labeled(best[0], code_col or best[1], label_col or best[2], max_rows)
        print("No labeled table found -> building the dataset from source files "
              "(folder/file names give the labels)...")
        return source_files_dataset(path, max_rows)
    return load_labeled(path, code_col, label_col, max_rows)


def train_file(path, code_col, label_col, langs, max_rows):
    import joblib
    from sklearn.metrics import classification_report
    from sklearn.model_selection import train_test_split

    df = resolve_dataset(path, code_col, label_col, max_rows)
    X, y = df["code"], df["label"]
    print(f"Rows: {len(df)} | vulnerable(1): {int(y.sum())} | safe(0): {int((1 - y).sum())}")
    if y.nunique() < 2:
        sys.exit("Dataset needs both vulnerable and safe samples (check the label column).")
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    model = make_pipeline()
    print("Training...")
    model.fit(Xtr, ytr)
    print(classification_report(yte, model.predict(Xte), target_names=["safe", "vulnerable"]))
    langs = [l.strip() for l in langs.split(",")]
    MODEL_DIR.mkdir(exist_ok=True)
    out = MODEL_DIR / ("+".join(sorted(langs)) + ".joblib")
    joblib.dump({"model": model, "langs": langs}, out)
    print(f"Saved model -> {out} (applies to: {', '.join(langs)})")


def cmd_train(args):
    train_file(args.dataset, args.code_col, args.label_col, args.langs, args.max_rows)


def cmd_trainall(args):
    jobs = [(args.cpp, "cpp"), (args.sql, "sql"), (args.python, "python")]
    jobs = [(p, l) for p, l in jobs if p]
    if not jobs:
        sys.exit("Give at least one of --cpp, --sql, --python (each a file or folder).")
    summary = []
    for path, lang in jobs:
        print("\n" + "=" * 70 + f"\nTraining the {lang.upper()} model from: {path}\n" + "=" * 70)
        try:
            train_file(path, None, None, lang, args.max_rows)
            summary.append((lang, "OK"))
        except SystemExit as e:
            print(f"[!] {lang} training failed: {e}")
            summary.append((lang, "FAILED"))
    print("\nSummary:", ", ".join(f"{l}: {r}" for l, r in summary))


def cmd_fetch(args):
    src = Path(args.source)
    if src.exists():
        root = src
    else:
        try:
            import kagglehub
        except ImportError:
            sys.exit("Install kagglehub first:  pip install kagglehub")
        kind = args.kind
        if kind == "auto":
            kind = "dataset" if "/" in args.source else "competition"
        try:
            fn = kagglehub.competition_download if kind == "competition" else kagglehub.dataset_download
            root = Path(fn(args.source))
        except Exception as e:  # network / auth / rules not accepted
            sys.exit(f"Download failed: {e}\n"
                     "Checklist: (1) log in with kagglehub.login() or set KAGGLE_USERNAME and KAGGLE_KEY, "
                     "(2) for a competition, open its Kaggle page and click 'Join / accept rules' first, "
                     "(3) check the name is exact.")
    print(f"Data folder: {root}\n")

    files = sorted(f for f in root.rglob("*") if f.is_file() and f.suffix.lower() in DATA_EXTS)
    if not files:
        sys.exit("No .csv/.tsv/.json/.jsonl/.parquet files found in that folder. "
                 "It may contain raw source files instead (scan them with: analyzer.py scan <folder>).")

    usable = []
    print(f"{'FILE':<55} {'SIZE':>9}  CODE col -> LABEL col")
    for f in files:
        size = f.stat().st_size
        rel = str(f.relative_to(root))
        try:
            head = load_table(f, 300)
            code, label = guess_columns(head)
        except (Exception, SystemExit) as e:
            print(f"{rel[:55]:<55} {size // 1024:>7}KB  (unreadable: {type(e).__name__})")
            continue
        ok = code is not None and label is not None
        print(f"{rel[:55]:<55} {size // 1024:>7}KB  "
              f"{code if ok else 'n/a'} -> {label if ok else 'n/a (no label: not usable for training)'}")
        if ok:
            usable.append((f, code, label, size))

    if args.file:
        usable = [u for u in usable if args.file.lower() in str(u[0]).lower()]
    if not usable:
        sys.exit("\nNo file with both code and label columns was found. "
                 "Check the table above and use: analyzer.py train <file> --code-col X --label-col Y")
    best = max(usable, key=lambda u: ("train" in u[0].name.lower(), u[3]))
    f, code, label, _ = best
    cmd = f'python analyzer.py train "{f}" --code-col "{code}" --label-col "{label}" --langs {args.langs}'
    print(f"\nBest file for training: {f}\nEquivalent command: {cmd}\n")
    if args.train:
        train_file(f, args.code_col or code, args.label_col or label, args.langs, args.max_rows)
    else:
        print("Add --train to train on it now.")


# ------------------------------------------------------------------- scan
def cmd_scan(args):
    target = Path(args.target)
    if not target.exists():
        sys.exit(f"Path not found: {target}")

    models = {}  # language -> trained model
    if MODEL_DIR.exists() and not args.no_ml:
        import joblib
        for mp in sorted(MODEL_DIR.glob("*.joblib")):
            obj = joblib.load(mp)
            for l in obj["langs"]:
                models[l] = obj["model"]
    if models:
        print(f"[i] AI models loaded for: {', '.join(sorted(models))}\n")
    else:
        print("[i] No trained model -> rule-based analysis only "
              "(run 'train' once your dataset is ready).\n")

    min_sev = SEV_ORDER[args.min_severity]
    results = []
    for f in iter_files(target):
        text = read_text(f)
        if not text or not text.strip():
            continue
        lang = detect_lang(f)
        lines = text.splitlines()
        findings = [x for x in rule_scan(lang, lines) if SEV_ORDER[x["severity"]] >= min_sev]
        findings.sort(key=lambda x: (-SEV_ORDER[x["severity"]], x["line"]))
        r = rule_score(findings)

        ml, regions = None, []
        if lang in models:
            ml, hot = ml_scan(models[lang], lines)
            for p, start in hot:
                if p >= args.threshold:
                    regions.append({"start_line": start, "end_line": start + WINDOW - 1,
                                    "probability": round(p, 2),
                                    "explanation": "The AI model finds this block similar to known vulnerable code.",
                                    "remediation": "Manually review input validation, bounds checks and memory handling here."})
        final = r if ml is None else 0.5 * ml + 0.5 * r
        results.append({"file": str(f), "language": lang, "risk_score": round(final * 100, 1),
                        "risk_level": level(final * 100),
                        "ai_score": None if ml is None else round(ml * 100, 1),
                        "rule_score": round(r * 100, 1),
                        "findings": findings, "ai_regions": regions})

    if not results:
        sys.exit("No C/C++, Python or SQL files found.")
    results.sort(key=lambda x: x["risk_score"], reverse=True)

    avg = sum(r["risk_score"] for r in results) / len(results)
    project = round(0.6 * results[0]["risk_score"] + 0.4 * avg, 1)
    counts, kinds = {}, {}
    for r in results:
        for fd in r["findings"]:
            counts[fd["severity"]] = counts.get(fd["severity"], 0) + 1
            kinds[fd["kind"]] = kinds.get(fd["kind"], 0) + 1

    print("=" * 80)
    print(f" PROJECT RISK: {project}/100 ({level(project)})   files scanned: {len(results)}")
    print(f" By severity : {counts or 'none'}")
    print(f" By type     : {kinds or 'none'}")
    print("=" * 80)

    shown = [r for r in results if r["findings"] or r["ai_regions"]][:args.top]
    for r in shown:
        print(f"\n### {r['file']}  [{r['language']}]  risk {r['risk_score']} ({r['risk_level']})"
              f"  | AI: {r['ai_score']}  rules: {r['rule_score']}")
        for fd in r["findings"]:
            print(f"  [{fd['severity'].upper()}] {fd['title']}  ({fd['cwe']}, {fd['kind']})")
            print(f"    Location   : line {fd['line']}")
            print(f"    Code       : {fd['code']}")
            print(f"    Why        : {fd['explanation']}")
            print(f"    Remediation: {fd['remediation']}")
        for g in r["ai_regions"]:
            print(f"  [AI] Suspicious region lines {g['start_line']}-{g['end_line']} "
                  f"(probability {g['probability']})")
            print(f"    Why        : {g['explanation']}")
            print(f"    Remediation: {g['remediation']}")
    if not shown:
        print("\nNo issues found at this severity level.")

    if args.json:
        Path(args.json).write_text(json.dumps({"project_risk": project, "files": results},
                                              indent=2), encoding="utf-8")
        print(f"\nFull report saved -> {args.json}")


def main():
    ap = argparse.ArgumentParser(description="Vulnerability & Code Risk Analyzer (C/C++, Python, SQL)")
    sub = ap.add_subparsers(dest="cmd", required=True)

    s = sub.add_parser("scan", help="scan a folder or file")
    s.add_argument("target")
    s.add_argument("--json", help="save the full report as JSON")
    s.add_argument("--min-severity", choices=list(SEV_ORDER), default="low")
    s.add_argument("--threshold", type=float, default=0.5, help="AI flag threshold (0-1)")
    s.add_argument("--top", type=int, default=25, help="max files to print")
    s.add_argument("--no-ml", action="store_true", help="ignore the trained model")
    s.set_defaults(fn=cmd_scan)

    t = sub.add_parser("train", help="train the AI model from a dataset")
    t.add_argument("dataset", help=".csv / .json / .jsonl")
    t.add_argument("--code-col")
    t.add_argument("--label-col")
    t.add_argument("--langs", default="cpp", help="languages the dataset covers: cpp,python,sql")
    t.add_argument("--max-rows", type=int, default=100_000)
    t.set_defaults(fn=cmd_train)

    a = sub.add_parser("trainall", help="train several language models in one go")
    a.add_argument("--cpp", help="C/C++ dataset: a file OR a folder (table, or raw .c files in vulnerable/safe folders)")
    a.add_argument("--sql", help="SQL dataset: a file or folder")
    a.add_argument("--python", help="Python dataset: a file or folder")
    a.add_argument("--max-rows", type=int, default=100_000)
    a.set_defaults(fn=cmd_trainall)

    f = sub.add_parser("fetch", help="download from Kaggle (or use a local folder), find the right file, optionally train")
    f.add_argument("source", help="Kaggle name (competition 'source-code-vulnerability' or dataset 'owner/name') or a local folder")
    f.add_argument("--kind", choices=["auto", "competition", "dataset"], default="auto")
    f.add_argument("--file", help="only consider files whose path contains this text")
    f.add_argument("--train", action="store_true", help="train on the best file found")
    f.add_argument("--langs", default="cpp", help="languages the data covers: cpp,python,sql")
    f.add_argument("--code-col")
    f.add_argument("--label-col")
    f.add_argument("--max-rows", type=int, default=100_000)
    f.set_defaults(fn=cmd_fetch)

    args = ap.parse_args()
    args.fn(args)


if __name__ == "__main__":
    main()

# #!/usr/bin/env python3
# """
# AI-Powered Software Vulnerability & Code Risk Analyzer  (CLI only)
# Languages: C/C++, Python, SQL

#   python analyzer.py scan <folder_or_file> [--json report.json] [--min-severity low]
#   python analyzer.py train <dataset.csv> [--code-col COL] [--label-col COL] [--langs cpp]
#   python analyzer.py fetch <kaggle-name | local-folder> [--train] [--langs sql]

# Each finding reports: kind, severity, file:line, code, explanation, remediation.
# Risk score (0-100) = 50% AI model + 50% rule engine (rules only if no model yet
# or the file's language is not covered by the model).
# """
# import argparse
# import json
# import re
# import sys
# from pathlib import Path

# MODEL_DIR = Path("models")  # one trained model file per language set
# MAX_FILE_BYTES = 1_000_000
# WINDOW, STRIDE = 30, 15

# SKIP_DIRS = {".git", "node_modules", "venv", ".venv", "env", "__pycache__", "dist",
#              "build", "target", ".idea", ".vscode", "vendor", "site-packages", "third_party"}
# EXT_LANG = {
#     ".c": "cpp", ".h": "cpp", ".cpp": "cpp", ".cc": "cpp", ".cxx": "cpp",
#     ".hpp": "cpp", ".hxx": "cpp", ".hh": "cpp",
#     ".py": "python",
#     ".sql": "sql",
# }
# COMMENT_PREFIX = {"cpp": ("//", "/*", "*"), "python": ("#",), "sql": ("--", "/*", "*")}
# SEV_ORDER = {"critical": 4, "high": 3, "medium": 2, "low": 1}
# SEV_WEIGHT = {"critical": 1.0, "high": 0.75, "medium": 0.45, "low": 0.2}

# ALL = {"cpp", "python", "sql"}
# CPP, PY, SQL = {"cpp"}, {"python"}, {"sql"}

# # (id, langs, severity, kind, regex, title, CWE, explanation, remediation)
# RULES = [
#     # ---------------- C / C++ ----------------
#     ("CPP_GETS", CPP, "critical", "vulnerability", r"\bgets\s*\(",
#      "Use of gets()", "CWE-242",
#      "gets() reads input with no length limit, so any long input overflows the buffer.",
#      "Use fgets(buf, sizeof(buf), stdin) or std::getline."),
#     ("CPP_UNBOUNDED_COPY", CPP, "high", "vulnerability", r"\b(strcpy|strcat|stpcpy|wcscpy|wcscat)\s*\(",
#      "Unbounded string copy", "CWE-120",
#      "These functions do not check destination size; oversized input causes a buffer overflow (memory corruption, code execution).",
#      "Use strncpy/strlcpy/snprintf with sizeof(dest), or std::string."),
#     ("CPP_SPRINTF", CPP, "high", "vulnerability", r"\b(sprintf|vsprintf)\s*\(",
#      "sprintf without size limit", "CWE-120",
#      "sprintf writes as much as the formatted data needs, regardless of buffer size.",
#      "Use snprintf(buf, sizeof(buf), ...)."),
#     ("CPP_SCANF_S", CPP, "high", "vulnerability", r"\b(scanf|fscanf|sscanf)\s*\(\s*[^;]*%s",
#      "scanf %s without width", "CWE-120",
#      "%s with no field width lets the user write past the end of the buffer.",
#      'Use a width, e.g. scanf("%63s", buf), or fgets/std::cin >> std::string.'),
#     ("CPP_FORMAT_STRING", CPP, "high", "vulnerability", r"\b(printf|vprintf|syslog)\s*\(\s*[A-Za-z_]\w*\s*\)",
#      "Format string from a variable", "CWE-134",
#      "A variable used as the format string lets an attacker inject %x/%n and read or write memory.",
#      'Use a constant format: printf("%s", var).'),
#     ("CPP_CMD_INJECT", CPP, "critical", "vulnerability", r"\b(system|popen|_popen|WinExec)\s*\(",
#      "OS command execution", "CWE-78",
#      "Passing input to a shell lets an attacker append commands (e.g. '; rm -rf /').",
#      "Avoid the shell; use execv with a fixed argument list and validate/whitelist input."),
#     ("CPP_EXEC", CPP, "medium", "suspicious", r"\b(execl|execlp|execle|execv|execvp|execve|ShellExecute[AW]?)\s*\(",
#      "Process execution", "CWE-78",
#      "Launching external programs is risky if the path or arguments come from user input.",
#      "Hard-code the program path and validate every argument."),
#     ("CPP_ALLOCA", CPP, "medium", "vulnerability", r"\balloca\s*\(",
#      "alloca() stack allocation", "CWE-770",
#      "Attacker-controlled sizes can exhaust the stack and crash or hijack the program.",
#      "Use heap allocation (std::vector / malloc) with a size limit."),
#     ("CPP_TEMPFILE", CPP, "medium", "vulnerability", r"\b(tmpnam|tempnam|mktemp)\s*\(",
#      "Insecure temporary file", "CWE-377",
#      "Predictable names allow race conditions (symlink attacks).",
#      "Use mkstemp() or std::filesystem with a random name."),
#     ("CPP_RAND", CPP, "low", "vulnerability", r"\b(rand|srand)\s*\(",
#      "Weak random number generator", "CWE-338",
#      "rand() is predictable and unsuitable for tokens, keys or passwords.",
#      "Use std::random_device / getrandom / OpenSSL RAND_bytes for security uses."),
#     ("CPP_ATOI", CPP, "low", "suspicious", r"\b(atoi|atol|atof)\s*\(",
#      "atoi without error checking", "CWE-190",
#      "No error or overflow detection, which can lead to integer-overflow bugs.",
#      "Use strtol/std::stoi with range and error checks."),
#     ("CPP_MEMCPY", CPP, "low", "suspicious", r"\b(memcpy|memmove)\s*\(",
#      "memcpy/memmove (check the length)", "CWE-119",
#      "If the length is derived from user input, it can overflow the destination.",
#      "Validate length <= sizeof(destination) before copying."),
#     ("CPP_PROC_INJECT", CPP, "high", "harmful", r"\b(CreateRemoteThread|VirtualAllocEx|WriteProcessMemory|NtUnmapViewOfSection)\b",
#      "Process-injection API", "CWE-506",
#      "Typical of malware that injects code into other processes.",
#      "Remove unless this is a documented, audited security tool."),
#     ("CPP_KEYLOG", CPP, "medium", "harmful", r"\b(SetWindowsHookEx[AW]?|GetAsyncKeyState|GetKeyboardState)\b",
#      "Keyboard hooking / key capture", "CWE-506",
#      "Can be used to capture keystrokes (keylogger behaviour).",
#      "Remove, or clearly document and restrict the legitimate purpose."),
#     ("CPP_SQL_CONCAT", CPP, "high", "vulnerability",
#      r'(?i)("[^"]*\b(select|insert\s+into|update|delete\s+from)\b[^"]*"\s*(\+|<<)\s*\w)|(\b(sprintf|snprintf|strcat)\s*\([^;]*\b(select|insert|update|delete)\b)',
#      "SQL query built by concatenation", "CWE-89",
#      "User input mixed into the query text lets an attacker change the SQL (SQL injection).",
#      "Use prepared statements with bound parameters (sqlite3_bind_*, mysql_stmt_*, etc.)."),

#     # ---------------- Python ----------------
#     ("PY_EVAL", PY, "critical", "vulnerability", r"(?<![\w.])(eval|exec)\s*\(",
#      "eval()/exec() code execution", "CWE-95",
#      "Executes arbitrary Python; with user-controlled input this is full remote code execution.",
#      "Use ast.literal_eval for data, or a dispatch dict for allowed actions."),
#     ("PY_CMD_INJECT", PY, "critical", "vulnerability",
#      r"(os\.system\s*\(|os\.popen\s*\(|shell\s*=\s*True|commands\.getoutput\s*\()",
#      "Shell command execution", "CWE-78",
#      "Commands run through a shell can be hijacked by input such as '; rm -rf ~'.",
#      "Use subprocess.run([...], shell=False) with an argument list; validate inputs."),
#     ("PY_DESERIALIZE", PY, "high", "vulnerability",
#      r"(pickle\.loads?\s*\(|cPickle\.loads?\s*\(|marshal\.loads?\s*\(|yaml\.load\s*\((?![^)]*Safe)|shelve\.open\s*\()",
#      "Insecure deserialization", "CWE-502",
#      "Unpickling or unsafe YAML loading of untrusted data executes attacker code.",
#      "Use json, or yaml.safe_load; never unpickle untrusted data."),
#     ("PY_SQL_INJECT", PY, "high", "vulnerability",
#      r"""\.(execute|executemany|executescript|raw)\s*\(\s*(f["']|["'][^"']*["']\s*(%|\+|\.format)|\w+\s*\+)""",
#      "SQL built with string formatting", "CWE-89",
#      "Query text assembled from variables allows SQL injection.",
#      'Use parameters: cursor.execute("SELECT * FROM t WHERE id = %s", (uid,)).'),
#     ("PY_TLS_OFF", PY, "high", "vulnerability", r"(verify\s*=\s*False|_create_unverified_context|CERT_NONE)",
#      "TLS certificate verification disabled", "CWE-295",
#      "Connections can be intercepted by a man-in-the-middle attacker.",
#      "Keep verification on; supply a CA bundle if you use a private CA."),
#     ("PY_WEAK_HASH", PY, "medium", "vulnerability", r"hashlib\.(md5|sha1)\s*\(",
#      "Weak hash function", "CWE-328",
#      "MD5/SHA-1 are broken for security; passwords hashed this way are easy to crack.",
#      "Use hashlib.sha256+, and bcrypt/argon2/scrypt for passwords."),
#     ("PY_DEBUG", PY, "medium", "vulnerability", r"(\.run\s*\([^)]*debug\s*=\s*True|^\s*DEBUG\s*=\s*True)",
#      "Debug mode enabled", "CWE-489",
#      "The Werkzeug/Django debugger exposes code and may allow remote execution.",
#      "Disable debug in production; read it from an environment variable."),
#     ("PY_TEMPFILE", PY, "medium", "vulnerability", r"tempfile\.mktemp\s*\(",
#      "Insecure temporary file", "CWE-377",
#      "mktemp() returns a name only, creating a race condition.",
#      "Use tempfile.NamedTemporaryFile or mkstemp."),
#     ("PY_TAR", PY, "medium", "vulnerability", r"\.extractall\s*\(",
#      "Archive extraction (path traversal)", "CWE-22",
#      "Malicious archives can write files outside the target folder.",
#      "Validate member paths, or use filter='data' (Python 3.12+)."),
#     ("PY_SSTI", PY, "medium", "vulnerability", r"(render_template_string\s*\(|autoescape\s*=\s*False|Markup\s*\()",
#      "Template injection / XSS risk", "CWE-79",
#      "Rendering user-controlled strings as templates or disabling escaping enables XSS/SSTI.",
#      "Use render_template with files and keep autoescape on."),
#     ("PY_RANDOM", PY, "low", "vulnerability", r"\brandom\.(random|randint|choice|randrange)\s*\(",
#      "Non-cryptographic random", "CWE-338",
#      "The random module is predictable; do not use it for tokens or passwords.",
#      "Use the secrets module."),
#     ("PY_BIND_ALL", PY, "low", "suspicious", r"""["']0\.0\.0\.0["']""",
#      "Binds to all network interfaces", "CWE-1327",
#      "The service becomes reachable from every network.",
#      "Bind to 127.0.0.1 unless public access is required."),
#     ("PY_NO_TIMEOUT", PY, "low", "suspicious", r"requests\.(get|post|put|delete|head)\s*\((?!.*timeout)",
#      "HTTP request without timeout", "CWE-400",
#      "A slow or hostile server can hang your program forever.",
#      "Pass timeout=10 (or similar)."),
#     ("PY_DESTRUCTIVE", PY, "medium", "harmful", r"(shutil\.rmtree\s*\(|os\.(remove|unlink|rmdir|removedirs)\s*\()",
#      "Destructive file operation", "CWE-73",
#      "Deletes files/folders; dangerous if the path is user-controlled or wrong.",
#      "Validate and restrict the path, and ask for confirmation / use a trash folder."),
#     ("PY_OBFUSCATION", PY, "high", "harmful",
#      r"(base64\.b64decode|zlib\.decompress|codecs\.decode|bytes\.fromhex)\s*\(.*\)\s*.*(exec|eval)|(exec|eval)\s*\(.*(b64decode|decompress|codecs\.decode)",
#      "Obfuscated code execution", "CWE-506",
#      "Decoding a payload and executing it is a common malware technique.",
#      "Remove it; ship readable code."),
#     ("PY_KEYLOG", PY, "medium", "harmful", r"(from\s+pynput|import\s+pynput|keyboard\.(hook|on_press|record)\s*\()",
#      "Keyboard capture library", "CWE-506",
#      "Can record keystrokes (keylogger behaviour).",
#      "Remove, or document and restrict the legitimate use."),

#     # ---------------- SQL ----------------
#     ("SQL_XP_CMDSHELL", SQL, "critical", "harmful", r"(?i)\bxp_cmdshell\b",
#      "xp_cmdshell (OS commands from SQL)", "CWE-78",
#      "Lets SQL Server run operating-system commands; a favourite for attackers.",
#      "Disable xp_cmdshell and remove its usage."),
#     ("SQL_DYNAMIC", SQL, "high", "vulnerability",
#      r"(?i)(\bEXEC(UTE)?\s*\(|\bEXECUTE\s+IMMEDIATE\b|\bsp_executesql\b|\bPREPARE\s+\w+\s+FROM\s+@)",
#      "Dynamic SQL execution", "CWE-89",
#      "SQL assembled as a string and executed can be injected if any part comes from input.",
#      "Use parameterised sp_executesql / bound parameters; avoid string concatenation."),
#     ("SQL_GRANT_ALL", SQL, "high", "vulnerability", r"(?i)(\bGRANT\s+ALL\b|\bTO\s+PUBLIC\b|WITH\s+GRANT\s+OPTION)",
#      "Over-broad privileges", "CWE-269",
#      "Gives users more rights than needed, widening the damage of any compromise.",
#      "Apply least privilege: grant only the specific rights required."),
#     ("SQL_PLAIN_PASS", SQL, "high", "vulnerability",
#      r"""(?i)(IDENTIFIED\s+BY|\bPASSWORD\s*=?)\s*['"][^'"]+['"]""",
#      "Plain-text password in SQL", "CWE-798",
#      "Credentials stored in scripts leak via version control and backups.",
#      "Use secret management / environment variables; rotate this password."),
#     ("SQL_PAYLOAD", ALL, "high", "suspicious",
#      r"""(?i)(\bOR\s+['"]?1['"]?\s*=\s*['"]?1\b|\bUNION\s+(ALL\s+)?SELECT\b|;\s*DROP\s+TABLE|'\s*--\s*$)""",
#      "SQL-injection payload pattern", "CWE-89",
#      "Looks like an injection string (tautology/UNION/stacked query), or code vulnerable to one.",
#      "Use parameterised queries and validate input; remove test payloads from source."),
#     ("SQL_DESTRUCTIVE", SQL, "medium", "harmful",
#      r"(?i)\b(DROP\s+(TABLE|DATABASE|SCHEMA)|TRUNCATE\s+TABLE)\b",
#      "Destructive statement", "CWE-285",
#      "Permanently deletes data; dangerous if run by mistake or in production.",
#      "Wrap in a transaction, back up first, use IF EXISTS and restrict who can run it."),
#     ("SQL_SELECT_STAR", SQL, "low", "suspicious", r"(?i)\bSELECT\s+\*\s+FROM\b",
#      "SELECT * (data over-exposure)", "CWE-200",
#      "May return sensitive columns (password hashes, tokens) that callers should not see.",
#      "List only the columns you need."),

#     # ---------------- All languages ----------------
#     ("SECRET", ALL, "high", "vulnerability",
#      r"""(?i)\b(password|passwd|pwd|secret|api[_-]?key|access[_-]?token|auth[_-]?token)\b\s*[:=]\s*["'][^"'\s]{4,}["']""",
#      "Hard-coded credential/secret", "CWE-798",
#      "Secrets in source code leak through repositories and can be reused by anyone.",
#      "Load from environment variables or a secrets manager; rotate the exposed value."),
#     ("AWS_KEY", ALL, "critical", "vulnerability", r"AKIA[0-9A-Z]{16}",
#      "AWS access key in code", "CWE-798",
#      "A valid-looking AWS key is committed; attackers scan repos for these.",
#      "Revoke the key now and use IAM roles / environment credentials."),
#     ("PRIVATE_KEY", ALL, "critical", "vulnerability", r"-----BEGIN (RSA |EC |DSA |OPENSSH )?PRIVATE KEY-----",
#      "Private key in source", "CWE-321",
#      "Anyone with the code can impersonate the key owner.",
#      "Remove it, rotate the key pair, and load keys from secure storage."),
#     ("REVERSE_SHELL", ALL, "critical", "harmful",
#      r"(/bin/(ba)?sh\s+-i|\bnc\s+(-e|-c)\s|bash\s+-i\s*>&|/dev/tcp/)",
#      "Reverse-shell pattern", "CWE-506",
#      "Opens an interactive shell to a remote machine, typical of backdoors.",
#      "Remove it and investigate how it entered the codebase."),
#     ("DESTRUCTIVE_CMD", ALL, "high", "harmful", r"(rm\s+-rf\s+(/|~|\*)|mkfs\.\w+\s+/dev|dd\s+if=.*of=/dev/|format\s+c:)",
#      "Destructive shell command", "CWE-506",
#      "Can wipe files or disks.",
#      "Remove, or guard with strict path validation and confirmation."),
# ]
# COMPILED = [(i, l, s, k, re.compile(p), t, c, e, r) for i, l, s, k, p, t, c, e, r in RULES]

# FREE_RE = re.compile(r"\bfree\s*\(\s*(\w+)\s*\)|\bdelete(?:\s*\[\s*\])?\s+(\w+)\s*;")


# # ------------------------------------------------------------------ helpers
# def detect_lang(p: Path):
#     return EXT_LANG.get(p.suffix.lower())


# def iter_files(target: Path):
#     if target.is_file():
#         if detect_lang(target):
#             yield target
#         return
#     for p in target.rglob("*"):
#         if any(part in SKIP_DIRS for part in p.parts):
#             continue
#         if p.is_file() and detect_lang(p):
#             try:
#                 if p.stat().st_size <= MAX_FILE_BYTES:
#                     yield p
#             except OSError:
#                 continue


# def read_text(p: Path):
#     try:
#         data = p.read_bytes()
#     except OSError:
#         return None
#     if b"\x00" in data[:2048]:
#         return None
#     return data.decode("utf-8", errors="ignore")


# def make_finding(n, line, rule, kind=None):
#     rid, _, sev, k, _, title, cwe, why, fix = rule
#     return {"rule": rid, "kind": kind or k, "severity": sev, "cwe": cwe, "title": title,
#             "line": n, "code": line.strip()[:140], "explanation": why, "remediation": fix}


# def rule_scan(lang, lines):
#     findings, seen = [], set()
#     in_block = False
#     for n, line in enumerate(lines, 1):
#         s = line.strip()
#         if lang in ("cpp", "sql"):  # block comments
#             if in_block:
#                 if "*/" in s:
#                     in_block = False
#                 continue
#             if s.startswith("/*") and "*/" not in s:
#                 in_block = True
#                 continue
#         if not s or s.startswith(COMMENT_PREFIX[lang]):
#             continue
#         for rule in COMPILED:
#             if lang in rule[1] and rule[4].search(line):
#                 key = (rule[0], n)
#                 if key not in seen:
#                     seen.add(key)
#                     findings.append(make_finding(n, line, rule))

#     if lang == "cpp":
#         findings += cpp_memory_checks(lines)
#     if lang == "sql":
#         findings += sql_unsafe_dml(lines)
#     return findings


# def cpp_memory_checks(lines):
#     """Heuristic use-after-free / double-free detection."""
#     out, freed = [], {}
#     for n, line in enumerate(lines, 1):
#         code = line.split("//")[0]
#         if code.strip().startswith(("*", "/*")):
#             continue
#         m = FREE_RE.search(code)
#         now = (m.group(1) or m.group(2)) if m else None
#         for var in list(freed):
#             if n - freed[var] > 20:
#                 del freed[var]
#                 continue
#             if re.search(rf"\b{re.escape(var)}\s*=(?!=)", code):
#                 del freed[var]
#                 continue
#             if var == now:
#                 out.append(_mem("CPP_DOUBLE_FREE", "Possible double free", "CWE-415",
#                                 "The same pointer is freed twice, corrupting the heap.",
#                                 "Set the pointer to nullptr after free, or use smart pointers.", n, line))
#                 del freed[var]
#             elif re.search(rf"\b{re.escape(var)}\b", code):
#                 out.append(_mem("CPP_UAF", "Possible use after free", "CWE-416",
#                                 f"'{var}' is used after being freed; attackers can exploit this for code execution.",
#                                 "Set the pointer to nullptr after free and never touch it again; prefer smart pointers.",
#                                 n, line))
#                 del freed[var]
#         if now and now not in freed:
#             freed[now] = n
#     return out


# def _mem(rid, title, cwe, why, fix, n, line):
#     return {"rule": rid, "kind": "vulnerability", "severity": "high", "cwe": cwe, "title": title,
#             "line": n, "code": line.strip()[:140], "explanation": why, "remediation": fix}


# def sql_unsafe_dml(lines):
#     out = []
#     rx = re.compile(r"(?i)^\s*(DELETE\s+FROM\s+[\w.\[\]\"`]+|UPDATE\s+[\w.\[\]\"`]+\s+SET\b[^;]*)\s*;?\s*$")
#     for n, line in enumerate(lines, 1):
#         if rx.match(line) and not re.search(r"(?i)\bWHERE\b", line):
#             out.append({"rule": "SQL_NO_WHERE", "kind": "harmful", "severity": "high", "cwe": "CWE-285",
#                         "title": "DELETE/UPDATE without WHERE", "line": n, "code": line.strip()[:140],
#                         "explanation": "Affects every row in the table; one mistake wipes or corrupts all data.",
#                         "remediation": "Add a WHERE clause, test with SELECT first, and run inside a transaction."})
#     return out


# def rule_score(findings):
#     safe = 1.0
#     for f in findings:
#         safe *= 1 - 0.6 * SEV_WEIGHT[f["severity"]]
#     return 1 - safe


# def level(score):
#     return "HIGH" if score >= 70 else "MEDIUM" if score >= 40 else "LOW"


# # ----------------------------------------------------------------- AI model
# def ml_scan(model, lines):
#     chunks, starts = [], []
#     for i in range(0, max(1, len(lines) - WINDOW + STRIDE), STRIDE):
#         c = "\n".join(lines[i:i + WINDOW])
#         if c.strip():
#             chunks.append(c)
#             starts.append(i + 1)
#     if not chunks:
#         return 0.0, []
#     probs = model.predict_proba(chunks)[:, 1]
#     ranked = sorted(zip(probs, starts), reverse=True)
#     return float(ranked[0][0]), [(float(p), s) for p, s in ranked[:3]]


# def make_pipeline():
#     from sklearn.feature_extraction.text import TfidfVectorizer
#     from sklearn.linear_model import LogisticRegression
#     from sklearn.pipeline import Pipeline
#     return Pipeline([
#         ("tfidf", TfidfVectorizer(token_pattern=r"[A-Za-z_][A-Za-z_0-9]*|[^\sA-Za-z_0-9]",
#                                   ngram_range=(1, 3), max_features=200_000,
#                                   sublinear_tf=True, lowercase=False)),
#         ("clf", LogisticRegression(max_iter=1000, class_weight="balanced")),
#     ])


# CODE_COLS = ["query", "func", "func_before", "code", "source", "source_code", "function",
#              "vulnerable_code", "content", "text", "snippet", "sentence", "statement"]
# LABEL_COLS = ["target", "label", "vul", "vulnerable", "is_vulnerable", "vulnerability",
#               "malicious", "is_malicious", "class", "y"]
# POS_WORDS = {"1", "true", "yes", "vulnerable", "vuln", "bad", "insecure", "unsafe",
#              "malicious", "attack", "sqli", "injection", "sql injection"}
# DATA_EXTS = {".csv", ".tsv", ".json", ".jsonl", ".parquet"}


# def find_col(df, wanted, candidates):
#     if wanted:
#         return wanted if wanted in df.columns else None
#     lower = {str(c).lower(): c for c in df.columns}
#     for c in candidates:
#         if c in lower:
#             return lower[c]
#     return None


# def guess_columns(df):
#     """Find the code column and the label column, even with unusual names."""
#     code = find_col(df, None, CODE_COLS)
#     if code is None:  # fall back to the column with the longest text
#         best, best_len = None, 0
#         for c in df.columns:
#             if df[c].dtype == object:
#                 ln = df[c].dropna().astype(str).str.len().mean()
#                 if ln and ln > best_len:
#                     best, best_len = c, ln
#         if best is not None and best_len >= 15:
#             code = best
#     label = find_col(df, None, LABEL_COLS)
#     if label is None:  # fall back to a two-valued column
#         for c in df.columns:
#             if c != code and 1 < df[c].nunique(dropna=True) <= 2:
#                 label = c
#                 break
#     return code, label


# def to_binary(v):
#     s = str(v).strip().lower()
#     if s in POS_WORDS:
#         return 1
#     try:
#         return 1 if float(s) > 0 else 0
#     except ValueError:
#         return 0


# def load_table(path, nrows=None):
#     import pandas as pd
#     path = Path(path)
#     ext = path.suffix.lower()
#     if ext == ".jsonl":
#         return pd.read_json(path, lines=True, nrows=nrows)
#     if ext in (".json", ".parquet"):
#         df = pd.read_json(path) if ext == ".json" else pd.read_parquet(path)
#         return df.head(nrows) if nrows else df
#     return pd.read_csv(path, sep="\t" if ext == ".tsv" else ",", nrows=nrows,
#                        on_bad_lines="skip", encoding_errors="replace")


# def train_file(path, code_col, label_col, langs, max_rows):
#     import joblib
#     from sklearn.metrics import classification_report
#     from sklearn.model_selection import train_test_split

#     df = load_table(path, max_rows)
#     for name in (code_col, label_col):
#         if name and name not in df.columns:
#             sys.exit(f"Column '{name}' not found. Columns: {list(df.columns)}")
#     g_code, g_label = guess_columns(df)
#     code_col, label_col = code_col or g_code, label_col or g_label
#     if not code_col or not label_col:
#         sys.exit(f"Could not detect code/label columns. Use --code-col and --label-col. "
#                  f"Columns: {list(df.columns)}")
#     print(f"File: {path}\nUsing code column '{code_col}' and label column '{label_col}'")
#     df = df[[code_col, label_col]].dropna()
#     print("Label values:", df[label_col].value_counts().head(6).to_dict())
#     X, y = df[code_col].astype(str), df[label_col].map(to_binary)
#     print(f"Rows: {len(df)} | vulnerable(1): {int(y.sum())} | safe(0): {int((1 - y).sum())}")
#     if y.nunique() < 2:
#         sys.exit("Dataset needs both vulnerable and safe samples (check the label column).")
#     Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
#     model = make_pipeline()
#     print("Training...")
#     model.fit(Xtr, ytr)
#     print(classification_report(yte, model.predict(Xte), target_names=["safe", "vulnerable"]))
#     langs = [l.strip() for l in langs.split(",")]
#     MODEL_DIR.mkdir(exist_ok=True)
#     out = MODEL_DIR / ("+".join(sorted(langs)) + ".joblib")
#     joblib.dump({"model": model, "langs": langs}, out)
#     print(f"Saved model -> {out} (applies to: {', '.join(langs)})")


# def cmd_train(args):
#     train_file(args.dataset, args.code_col, args.label_col, args.langs, args.max_rows)


# def cmd_fetch(args):
#     src = Path(args.source)
#     if src.exists():
#         root = src
#     else:
#         try:
#             import kagglehub
#         except ImportError:
#             sys.exit("Install kagglehub first:  pip install kagglehub")
#         kind = args.kind
#         if kind == "auto":
#             kind = "dataset" if "/" in args.source else "competition"
#         try:
#             fn = kagglehub.competition_download if kind == "competition" else kagglehub.dataset_download
#             root = Path(fn(args.source))
#         except Exception as e:  # network / auth / rules not accepted
#             sys.exit(f"Download failed: {e}\n"
#                      "Checklist: (1) log in with kagglehub.login() or set KAGGLE_USERNAME and KAGGLE_KEY, "
#                      "(2) for a competition, open its Kaggle page and click 'Join / accept rules' first, "
#                      "(3) check the name is exact.")
#     print(f"Data folder: {root}\n")

#     files = sorted(f for f in root.rglob("*") if f.is_file() and f.suffix.lower() in DATA_EXTS)
#     if not files:
#         sys.exit("No .csv/.tsv/.json/.jsonl/.parquet files found in that folder. "
#                  "It may contain raw source files instead (scan them with: analyzer.py scan <folder>).")

#     usable = []
#     print(f"{'FILE':<55} {'SIZE':>9}  CODE col -> LABEL col")
#     for f in files:
#         size = f.stat().st_size
#         rel = str(f.relative_to(root))
#         try:
#             head = load_table(f, 300)
#             code, label = guess_columns(head)
#         except Exception as e:
#             print(f"{rel[:55]:<55} {size // 1024:>7}KB  (unreadable: {type(e).__name__})")
#             continue
#         ok = code is not None and label is not None
#         print(f"{rel[:55]:<55} {size // 1024:>7}KB  "
#               f"{code if ok else 'n/a'} -> {label if ok else 'n/a (no label: not usable for training)'}")
#         if ok:
#             usable.append((f, code, label, size))

#     if args.file:
#         usable = [u for u in usable if args.file.lower() in str(u[0]).lower()]
#     if not usable:
#         sys.exit("\nNo file with both code and label columns was found. "
#                  "Check the table above and use: analyzer.py train <file> --code-col X --label-col Y")
#     best = max(usable, key=lambda u: ("train" in u[0].name.lower(), u[3]))
#     f, code, label, _ = best
#     cmd = f'python analyzer.py train "{f}" --code-col "{code}" --label-col "{label}" --langs {args.langs}'
#     print(f"\nBest file for training: {f}\nEquivalent command: {cmd}\n")
#     if args.train:
#         train_file(f, args.code_col or code, args.label_col or label, args.langs, args.max_rows)
#     else:
#         print("Add --train to train on it now.")


# # ------------------------------------------------------------------- scan
# def cmd_scan(args):
#     target = Path(args.target)
#     if not target.exists():
#         sys.exit(f"Path not found: {target}")

#     models = {}  # language -> trained model
#     if MODEL_DIR.exists() and not args.no_ml:
#         import joblib
#         for mp in sorted(MODEL_DIR.glob("*.joblib")):
#             obj = joblib.load(mp)
#             for l in obj["langs"]:
#                 models[l] = obj["model"]
#     if models:
#         print(f"[i] AI models loaded for: {', '.join(sorted(models))}\n")
#     else:
#         print("[i] No trained model -> rule-based analysis only "
#               "(run 'train' once your dataset is ready).\n")

#     min_sev = SEV_ORDER[args.min_severity]
#     results = []
#     for f in iter_files(target):
#         text = read_text(f)
#         if not text or not text.strip():
#             continue
#         lang = detect_lang(f)
#         lines = text.splitlines()
#         findings = [x for x in rule_scan(lang, lines) if SEV_ORDER[x["severity"]] >= min_sev]
#         findings.sort(key=lambda x: (-SEV_ORDER[x["severity"]], x["line"]))
#         r = rule_score(findings)

#         ml, regions = None, []
#         if lang in models:
#             ml, hot = ml_scan(models[lang], lines)
#             for p, start in hot:
#                 if p >= args.threshold:
#                     regions.append({"start_line": start, "end_line": start + WINDOW - 1,
#                                     "probability": round(p, 2),
#                                     "explanation": "The AI model finds this block similar to known vulnerable code.",
#                                     "remediation": "Manually review input validation, bounds checks and memory handling here."})
#         final = r if ml is None else 0.5 * ml + 0.5 * r
#         results.append({"file": str(f), "language": lang, "risk_score": round(final * 100, 1),
#                         "risk_level": level(final * 100),
#                         "ai_score": None if ml is None else round(ml * 100, 1),
#                         "rule_score": round(r * 100, 1),
#                         "findings": findings, "ai_regions": regions})

#     if not results:
#         sys.exit("No C/C++, Python or SQL files found.")
#     results.sort(key=lambda x: x["risk_score"], reverse=True)

#     avg = sum(r["risk_score"] for r in results) / len(results)
#     project = round(0.6 * results[0]["risk_score"] + 0.4 * avg, 1)
#     counts, kinds = {}, {}
#     for r in results:
#         for fd in r["findings"]:
#             counts[fd["severity"]] = counts.get(fd["severity"], 0) + 1
#             kinds[fd["kind"]] = kinds.get(fd["kind"], 0) + 1

#     print("=" * 80)
#     print(f" PROJECT RISK: {project}/100 ({level(project)})   files scanned: {len(results)}")
#     print(f" By severity : {counts or 'none'}")
#     print(f" By type     : {kinds or 'none'}")
#     print("=" * 80)

#     shown = [r for r in results if r["findings"] or r["ai_regions"]][:args.top]
#     for r in shown:
#         print(f"\n### {r['file']}  [{r['language']}]  risk {r['risk_score']} ({r['risk_level']})"
#               f"  | AI: {r['ai_score']}  rules: {r['rule_score']}")
#         for fd in r["findings"]:
#             print(f"  [{fd['severity'].upper()}] {fd['title']}  ({fd['cwe']}, {fd['kind']})")
#             print(f"    Location   : line {fd['line']}")
#             print(f"    Code       : {fd['code']}")
#             print(f"    Why        : {fd['explanation']}")
#             print(f"    Remediation: {fd['remediation']}")
#         for g in r["ai_regions"]:
#             print(f"  [AI] Suspicious region lines {g['start_line']}-{g['end_line']} "
#                   f"(probability {g['probability']})")
#             print(f"    Why        : {g['explanation']}")
#             print(f"    Remediation: {g['remediation']}")
#     if not shown:
#         print("\nNo issues found at this severity level.")

#     if args.json:
#         Path(args.json).write_text(json.dumps({"project_risk": project, "files": results},
#                                               indent=2), encoding="utf-8")
#         print(f"\nFull report saved -> {args.json}")


# def main():
#     ap = argparse.ArgumentParser(description="Vulnerability & Code Risk Analyzer (C/C++, Python, SQL)")
#     sub = ap.add_subparsers(dest="cmd", required=True)

#     s = sub.add_parser("scan", help="scan a folder or file")
#     s.add_argument("target")
#     s.add_argument("--json", help="save the full report as JSON")
#     s.add_argument("--min-severity", choices=list(SEV_ORDER), default="low")
#     s.add_argument("--threshold", type=float, default=0.5, help="AI flag threshold (0-1)")
#     s.add_argument("--top", type=int, default=25, help="max files to print")
#     s.add_argument("--no-ml", action="store_true", help="ignore the trained model")
#     s.set_defaults(fn=cmd_scan)

#     t = sub.add_parser("train", help="train the AI model from a dataset")
#     t.add_argument("dataset", help=".csv / .json / .jsonl")
#     t.add_argument("--code-col")
#     t.add_argument("--label-col")
#     t.add_argument("--langs", default="cpp", help="languages the dataset covers: cpp,python,sql")
#     t.add_argument("--max-rows", type=int, default=100_000)
#     t.set_defaults(fn=cmd_train)

#     f = sub.add_parser("fetch", help="download from Kaggle (or use a local folder), find the right file, optionally train")
#     f.add_argument("source", help="Kaggle name (competition 'source-code-vulnerability' or dataset 'owner/name') or a local folder")
#     f.add_argument("--kind", choices=["auto", "competition", "dataset"], default="auto")
#     f.add_argument("--file", help="only consider files whose path contains this text")
#     f.add_argument("--train", action="store_true", help="train on the best file found")
#     f.add_argument("--langs", default="cpp", help="languages the data covers: cpp,python,sql")
#     f.add_argument("--code-col")
#     f.add_argument("--label-col")
#     f.add_argument("--max-rows", type=int, default=100_000)
#     f.set_defaults(fn=cmd_fetch)

#     args = ap.parse_args()
#     args.fn(args)


# if __name__ == "__main__":
#     main()