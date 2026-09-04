import hashlib
import json
import re
from pathlib import Path

import pandas as pd


# Canonical aliases. Keep this list intentionally conservative: only fields that have
# unambiguous semantic meaning are normalized.
KEY_MAPPING = {
    # identity / execution
    "run": "run_id", "run_id": "run_id", "execution_id": "run_id", "id": "run_id",
    "test": "test", "test_name": "test",
    "seed": "random_seed", "random_seed": "random_seed", "ntb_random_seed": "random_seed",
    "group": "seed_group", "seed_group": "seed_group",
    # configuration
    "cache": "cache_policy", "cache_policy": "cache_policy",
    "sched": "scheduler", "scheduler": "scheduler",
    "mem_mode": "memory_mode", "memory_mode": "memory_mode",
    "opt": "compiler_opt", "compiler_opt": "compiler_opt",
    "fx": "feature_x", "feature_x": "feature_x",
    "fy": "feature_y", "feature_y": "feature_y",
    "clk": "clock_mhz", "clock": "clock_mhz", "clock_mhz": "clock_mhz",
    "timeout": "timeout_ms", "timeout_ms": "timeout_ms",
    "traffic": "traffic_pattern", "traffic_pattern": "traffic_pattern",
    "workload": "workload",
    "environment": "environment", "env": "environment",
    "noc_vc": "noc_virtual_channels", "noc_virtual_channels": "noc_virtual_channels",
    # telemetry
    "throughput": "throughput_mb_s", "throughput_mb_s": "throughput_mb_s", "throughput_mbps": "throughput_mb_s",
    "latency": "average_latency_ns", "avg_latency_ns": "average_latency_ns", "average_latency_ns": "average_latency_ns",
    "execution_time": "execution_time_s", "execution_time_s": "execution_time_s", "runtime_s": "execution_time_s",
    "runtime": "execution_time_ms", "runtime_ms": "execution_time_ms", "execution_ms": "execution_time_ms", "exec": "execution_time_ms",
    "cpu": "cpu_util_pct", "cpu_util": "cpu_util_pct", "cpu_util_pct": "cpu_util_pct",
    "memory_mb": "memory_mb", "mem_mb": "memory_mb",
    "temperature": "temperature_c", "temp": "temperature_c", "temp_c": "temperature_c", "temperature_c": "temperature_c",
    "coverage": "functional_coverage_pct", "func_cov": "functional_coverage_pct", "functional_coverage_pct": "functional_coverage_pct",
    # status/failure
    "outcome": "outcome", "result": "outcome", "status": "outcome", "test_status": "outcome", "final_status": "outcome",
    "failure_type": "failure_type", "fail_type": "failure_type", "failure": "failure_type",
    "failure_signature": "failure_signature", "signature": "failure_signature", "sig": "failure_signature",
    "pattern": "traffic_pattern", "functional": "functional_coverage_pct",
}

# Generic keys which are too ambiguous and should not create schema columns.
BLOCKED_KEYS = {
    "state", "time", "timestamp", "warning", "error", "info", "debug",
    "uvm_info", "uvm_warning", "uvm_error", "uvm_fatal",
}

# Ordered from specific to general. A single failure may match several signatures;
# the first specific canonical class becomes failure_type while signatures are retained.
FAILURE_RULES = [
    ("THERMAL_TIMEOUT", [r"THERMAL[_ -]?TIMEOUT", r"THERMAL.*WATCHDOG", r"OVER.?TEMP.*TIMEOUT"]),
    ("TIMEOUT", [r"WATCHDOG[_ -]?TIMEOUT", r"DMA[_ -]?TIMEOUT", r"WATCHDOG_TRIGGER", r"WATCHDOG\s+EXPIRED", r"TRANSACTION\s+DID\s+NOT\s+COMPLETE"]),
    ("ASSERTION", [r"\bASSERT(?:ION)?(?:\s+FAILED|[_ -]?FAIL(?:ED)?)\b", r"UVM_(?:ERROR|FATAL).*ASSERT", r"SVA.*FAIL"]),
    ("PROTOCOL", [r"AXI[_ -]?PROTOCOL", r"PROTOCOL[_ -]?(?:ERROR|VIOLATION)", r"HANDSHAKE[_ -]?(?:ERROR|VIOLATION)", r"ERROR\s+CHECKER=[A-Z0-9_]*(?:PROTOCOL|BURST|TRANSITION)[A-Z0-9_]*"]),
    ("RESOURCE", [r"OUT OF MEMORY", r"OOM\b", r"RESOURCE[_ -]?(?:ERROR|EXHAUST)", r"QUEUE[_ -]?FULL", r"RESOURCE_MONITOR\s*=\s*[A-Z0-9_]+"]),
    ("DEADLOCK", [r"NOC[_ -]?DEADLOCK", r"DEADLOCK[_ -]?DETECTED", r"VC[_ -]?STARVATION"]),
    ("CACHE_COHERENCY_ERROR", [r"CACHE[_ -]?COHERENCY", r"SNOOP[_ -]?MISS"]),
    ("ECC_ERROR", [r"ECC[_ -]?ERROR", r"DOUBLE[_ -]?BIT"]),
]

EXPLICIT_STATUS_PATTERNS = [
    # Prefer terminal/final status forms. finditer + last match handles reordered logs.
    re.compile(r"(?im)^\s*\[?(?:FINAL[_ ]?STATUS|TEST[_ ]?STATUS|RESULT|OUTCOME)\]?\s*[:= ]\s*(PASS(?:ED)?|SUCCESS|OK|FAIL(?:ED)?|ERROR|ABORT(?:ED)?|TIMEOUT)\b"),
    re.compile(r"(?im)^\s*(?:TEST\s+)?(?:PASSED|FAILED)\s*$"),
    re.compile(r"(?im)\bUVM_(?:TEST_)?(?:PASS|FAIL)\b"),
]

FAIL_EVIDENCE_PATTERNS = [
    re.compile(r"(?im)^\s*UVM_FATAL\b"),
    re.compile(r"(?im)^\s*FATAL\s*[:\]]"),
    re.compile(r"(?im)\b(?:TEST|SIMULATION)\s+FAILED\b"),
    re.compile(r"(?im)\b(?:ASSERTION|PROTOCOL).*\bFAILED\b"),
    re.compile(r"(?im)^\s*(?:UVM_ERROR:|ERROR\s+)\s*(?:assertion|checker=)"),
    re.compile(r"(?im)\bWATCHDOG[_ -]?TIMEOUT\b"),
]

PASS_EVIDENCE_PATTERNS = [
    re.compile(r"(?im)\b(?:TEST|SIMULATION)\s+PASSED\b"),
    re.compile(r"(?im)\bCOMPLETED\s+SUCCESSFULLY\b"),
    re.compile(r"(?im)UVM_ERROR\s+count\s*[:=]\s*0\b"),
    re.compile(r"(?im)scoreboard.*\b0\s+errors?\b"),
]

SIGNATURE_PATTERNS = [
    re.compile(r"(?im)\bchecker\s*=\s*([A-Za-z0-9_.:/-]+)"),
    re.compile(r"(?im)\bresource_monitor\s*=\s*([A-Za-z0-9_.:/-]+)"),
    re.compile(r"(?im)\b(?:failure[_ ]?signature|signature|sig)\s*[:=]\s*([A-Za-z0-9_.:/-]+)"),
    re.compile(r"(?im)\bassert(?:ion)?\s+([A-Za-z][A-Za-z0-9_.:/-]{2,})\s+(?:failed|failure)\b"),
    re.compile(r"(?im)\b(?:property|check)\s+([A-Za-z][A-Za-z0-9_.:/-]{2,})\s+(?:failed|failure)\b"),
    re.compile(r"(?im)UVM_FATAL.*\(([A-Za-z][A-Za-z0-9_.:/-]+)\)"),
    re.compile(r"(?im)\b(UVM_FATAL|WATCHDOG_TIMEOUT|DMA_TIMEOUT|AXI_PROTOCOL_ERROR|NOC_DEADLOCK|ECC_ERROR)\b"),
]

NUMERIC_FIELDS = {
    "random_seed", "seed_group", "clock_mhz", "timeout_ms", "noc_virtual_channels",
    "throughput_mb_s", "average_latency_ns", "execution_time_s", "execution_time_ms",
    "cpu_util_pct", "memory_mb", "temperature_c", "functional_coverage_pct",
    "l2_cache_kb", "dma_channels", "axi_data_width", "axi_burst_len", "outstanding_txns",
    "voltage_mv", "packet_size_bytes", "irq_rate_khz",
}

CONFIG_ID_FIELDS = [
    "test", "cache_policy", "scheduler", "memory_mode", "compiler_opt", "feature_x", "feature_y",
    "clock_mhz", "timeout_ms", "traffic_pattern", "workload", "environment", "noc_virtual_channels",
    "cpu_mode", "l2_cache_kb", "dma_channels", "axi_data_width", "axi_burst_len", "outstanding_txns",
    "snoop_enable", "ecc_enable", "prefetch_enable", "iommu_enable", "voltage_mv",
    "packet_size_bytes", "irq_rate_khz", "reset_n", "reset_sequence",
]


ALLOWED_KEYS = set(CONFIG_ID_FIELDS) | NUMERIC_FIELDS | {
    "run_id", "random_seed", "seed_group", "outcome", "failure_type", "failure_signature",
    "throughput_mb_s", "average_latency_ns", "execution_time_s", "execution_time_ms",
    "cpu_util_pct", "memory_mb", "temperature_c", "functional_coverage_pct", "config_id",
}


def _norm_key(key):
    key = re.sub(r"[^a-zA-Z0-9]+", "_", str(key).strip()).strip("_").lower()
    return KEY_MAPPING.get(key, key)


def _clean_value(value):
    value = str(value).strip().strip('"\'').rstrip(",;")
    # Strip common units only for numeric conversion; semantic field determines units.
    numeric_candidate = re.sub(r"(?i)(?:%|mb/s|mbps|ns|ms|sec|secs|seconds|s|mhz|mv|mb|c)$", "", value).strip()
    if re.fullmatch(r"[-+]?\d+", numeric_candidate):
        try:
            return int(numeric_candidate)
        except ValueError:
            pass
    if re.fullmatch(r"[-+]?(?:\d+\.\d*|\d*\.\d+)", numeric_candidate):
        try:
            return float(numeric_candidate)
        except ValueError:
            pass
    return value


def _extract_pairs(content):
    """Extract well-formed key/value assignments without turning prefixes/timestamps into keys."""
    pairs = []
    # Key must begin at line start or after whitespace, |, +. This prevents "INFO test"
    # from becoming a key and still supports compact pipe-delimited logs and +args.
    pair_re = re.compile(
        r"(?:^|[\s|+])([A-Za-z_][A-Za-z0-9_]*)\s*[:=]\s*"
        r"(\"[^\"]*\"|'[^']*'|[A-Za-z0-9_+./%-]+)"
    )
    for raw_line in content.splitlines():
        raw_upper = raw_line.upper()
        is_perf = bool(re.search(r"\b(?:PERF|PERFORMANCE)\b", raw_upper))
        is_cov = bool(re.search(r"\b(?:COV|COVERAGE)\b", raw_upper))
        is_cfg = bool(re.search(r"\b(?:CONFIG|CFG)\b", raw_upper))
        line = raw_line.strip()
        # Strip bracketed timestamp/severity prefixes but leave the actual assignments.
        line = re.sub(r"^\s*\[[^\]]+\]\s*", "", line)
        # Questa-style lines use "vsim INFO:" as a message prefix. Remove it so INFO
        # is not mistaken for the key and the following run_id assignment remains visible.
        line = re.sub(r"(?i)^\s*(?:vsim\s+)?(?:INFO|WARNING|DEBUG|FATAL)\s*:\s*", "", line)
        for match in pair_re.finditer(line):
            key, value = match.group(1), match.group(2)
            kl = key.lower()
            # Context disambiguates overloaded compact aliases.
            if kl == "mem":
                key = "memory_mb" if is_perf else "memory_mode"
            elif kl == "functional" and is_cov:
                key = "functional_coverage_pct"
            elif kl in {"runtime", "exec"} and is_perf:
                key = "execution_time_ms"
            elif kl == "temp" and is_perf:
                key = "temperature_c"
            pairs.append((key, value))
    return pairs


def _canonicalize_pairs(pairs):
    values = {}
    conflicts = {}
    aliases = {}
    extras = {}
    for raw_key, raw_value in pairs:
        original_key = re.sub(r"[^a-zA-Z0-9]+", "_", raw_key.strip()).strip("_").lower()
        key = _norm_key(raw_key)
        if key in BLOCKED_KEYS or not key:
            continue
        value = _clean_value(raw_value)
        if key in NUMERIC_FIELDS and isinstance(value, str):
            continue
        if key not in ALLOWED_KEYS:
            # Preserve unfamiliar data compactly without exploding the CSV schema.
            extras.setdefault(original_key, []).append(value)
            continue
        aliases.setdefault(key, set()).add(original_key)
        if key not in values or pd.isna(values[key]):
            values[key] = value
        elif str(values[key]) != str(value):
            conflicts.setdefault(key, set()).update([str(values[key]), str(value)])
    return values, conflicts, aliases, extras


def _normalize_explicit_status(token):
    token = token.upper().strip()
    if token in {"PASS", "PASSED", "SUCCESS", "OK"} or "PASS" in token:
        return "Pass"
    if token in {"FAIL", "FAILED", "ERROR", "ABORT", "ABORTED", "TIMEOUT"} or "FAIL" in token:
        return "Fail"
    return "Unknown"


def _detect_outcome(content, parsed_outcome=None):
    # Explicit final RESULT/OUTCOME wins over ERROR/WARNING text elsewhere in a log.
    explicit = []
    for pattern in EXPLICIT_STATUS_PATTERNS:
        for m in pattern.finditer(content):
            token = m.group(1) if m.lastindex else m.group(0)
            explicit.append((m.start(), _normalize_explicit_status(token)))
    if explicit:
        explicit.sort(key=lambda x: x[0])
        return explicit[-1][1], "explicit_final_status", 1.0

    if parsed_outcome is not None:
        parsed = _normalize_explicit_status(str(parsed_outcome))
        if parsed != "Unknown":
            return parsed, "parsed_status_field", 0.95

    fail_hits = sum(bool(p.search(content)) for p in FAIL_EVIDENCE_PATTERNS)
    pass_hits = sum(bool(p.search(content)) for p in PASS_EVIDENCE_PATTERNS)
    if fail_hits and not pass_hits:
        return "Fail", "strong_failure_evidence", 0.75
    if pass_hits and not fail_hits:
        return "Pass", "strong_pass_evidence", 0.75
    return "Unknown", "insufficient_or_conflicting_evidence", 0.25


def _extract_failure(content, outcome, existing_type=None, existing_signature=None):
    if outcome != "Fail":
        return "NONE", pd.NA

    content_upper = content.upper()
    failure_type = None
    if existing_type and str(existing_type).upper() not in {"NONE", "NAN", "UNKNOWN", "UNKNOWN_FAILURE"}:
        failure_type = str(existing_type).strip().upper()
    if not failure_type:
        for canonical, patterns in FAILURE_RULES:
            if any(re.search(p, content_upper, re.IGNORECASE) for p in patterns):
                failure_type = canonical
                break
    if not failure_type:
        failure_type = "UNKNOWN_FAILURE"

    signature = None
    if existing_signature and str(existing_signature).upper() not in {"NAN", "NONE", "UNKNOWN"}:
        signature = str(existing_signature).strip()
    if not signature:
        for pattern in SIGNATURE_PATTERNS:
            m = pattern.search(content)
            if m:
                signature = m.group(1).strip()
                break
    if not signature and failure_type != "UNKNOWN_FAILURE":
        signature = failure_type
    return failure_type, signature if signature else pd.NA


def _looks_truncated(content, outcome_source):
    if outcome_source in {"explicit_final_status", "parsed_status_field"}:
        return False
    tail = content[-800:].upper()
    start_markers = bool(re.search(r"\b(?:TEST|SIMULATION|RUN)[_ ]?(?:START|BEGIN)\b", content.upper()))
    terminal_markers = bool(re.search(r"\b(?:RESULT|OUTCOME|FINAL[_ ]?STATUS)\s*[:=]|\b(?:TEST|SIMULATION)\s+(?:PASSED|FAILED)\b", tail))
    return bool(start_markers and not terminal_markers)


def _make_config_id(row):
    existing = row.get("config_id")
    if existing is not None and not pd.isna(existing) and str(existing).strip():
        return str(existing).strip()
    parts = []
    for key in CONFIG_ID_FIELDS:
        val = row.get(key)
        if val is not None and not pd.isna(val):
            parts.append(f"{key}={val}")
    if not parts:
        return pd.NA
    digest = hashlib.sha1("|".join(parts).encode("utf-8")).hexdigest()[:12]
    return f"CFG_{digest}"


def parse_logs(folder_path, output_path="parsed_logs.csv"):
    """Parse raw verification logs into one canonical row per execution."""
    data = []
    log_files = sorted(list(Path(folder_path).rglob("*.log")) + list(Path(folder_path).rglob("*.txt")))
    if not log_files:
        return False, f"❌ ERROR: No .log or .txt files found in '{folder_path}'"

    read_errors = 0
    for file_path in log_files:
        try:
            content = file_path.read_text(errors="ignore")
            pairs = _extract_pairs(content)
            row, conflicts, aliases, extras = _canonicalize_pairs(pairs)

            # More permissive seed extraction for simulator forms such as +ntb_random_seed=123.
            if "random_seed" not in row:
                m = re.search(r"(?i)(?:\+?ntb_random_seed|random[_ ]?seed|--seed|\bseed)\s*[:= ]\s*(\d+)", content)
                if m:
                    row["random_seed"] = int(m.group(1))

            parsed_outcome = row.pop("outcome", None)
            outcome, outcome_source, outcome_confidence = _detect_outcome(content, parsed_outcome)
            failure_type, failure_signature = _extract_failure(
                content, outcome, row.pop("failure_type", None), row.pop("failure_signature", None)
            )

            row["filename"] = file_path.name
            row["Outcome"] = outcome
            row["Failure_Type"] = failure_type
            row["failure_signature"] = failure_signature
            row["outcome_source"] = outcome_source
            row["outcome_confidence"] = outcome_confidence
            row["parse_conflict"] = bool(conflicts)
            row["conflict_fields"] = ";".join(sorted(conflicts)) if conflicts else pd.NA
            row["conflict_details"] = "; ".join(
                f"{k}=[{'|'.join(sorted(v))}]" for k, v in sorted(conflicts.items())
            ) if conflicts else pd.NA
            row["is_truncated"] = _looks_truncated(content, outcome_source)
            row["parse_warning"] = (
                "conflicting_duplicate_fields" if conflicts else
                "possible_truncated_log" if row["is_truncated"] else
                "insufficient_outcome_evidence" if outcome == "Unknown" else pd.NA
            )
            row["config_id"] = _make_config_id(row)
            if extras:
                row["extra_fields"] = json.dumps(extras, sort_keys=True, default=str)
            data.append(row)
        except Exception as exc:
            read_errors += 1
            print(f"⚠️ Could not read {file_path.name}: {exc}")

    if not data:
        return False, "❌ ERROR: No valid data extracted."

    df = pd.DataFrame(data).dropna(axis=1, how="all")

    # Canonical column order first; any legitimate extra keys are retained after these.
    canonical_order = [
        "run_id", "filename", "test", "random_seed", "seed_group", "traffic_pattern", "workload", "environment",
        "cache_policy", "scheduler", "memory_mode", "compiler_opt", "feature_x", "feature_y", "clock_mhz", "timeout_ms",
        "noc_virtual_channels", "throughput_mb_s", "average_latency_ns", "execution_time_s", "execution_time_ms",
        "cpu_util_pct", "memory_mb", "temperature_c", "functional_coverage_pct", "Outcome", "Failure_Type",
        "failure_signature", "outcome_source", "outcome_confidence", "parse_conflict", "conflict_fields",
        "conflict_details", "is_truncated", "parse_warning", "config_id", "extra_fields",
    ]
    ordered = [c for c in canonical_order if c in df.columns] + [c for c in df.columns if c not in canonical_order]
    df = df[ordered]
    df.to_csv(output_path, index=False)

    unknown = int((df["Outcome"] == "Unknown").sum()) if "Outcome" in df else 0
    conflicts = int(df.get("parse_conflict", pd.Series(False, index=df.index)).fillna(False).sum())
    return True, (
        f"✅ Parsed {len(df)} logs into {output_path}. "
        f"Unknown outcomes: {unknown}; conflict-flagged logs: {conflicts}; read errors: {read_errors}."
    )
