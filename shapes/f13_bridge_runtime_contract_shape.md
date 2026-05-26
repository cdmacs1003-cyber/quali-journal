# F13 Bridge Runtime Contract Shape

`shapes/f13_bridge_runtime_contract_shape.json` is the canonical
machine-readable Bridge/F13 runtime contract shape.

This Markdown file is a human-readable documentation wrapper for the JSON
shape. It exists to satisfy the expected documentation path without replacing
the JSON contract artifact.

## Static Contract Summary

- Routes: retrieve-evidence, check-policy, explain-trace.
- Required external flags: `raw_text_included=false` and `internal_path_included=false`.
- Safe evidence fields: evidence identifiers, safe summaries, pointer metadata,
  raw text policy, rights status, and validation shape identifiers.
- Fail-closed marker groups: raw text, internal path, secret/token, and DB/DSN.

## Deferred Verification

- Bridge functional 200 behavior remains NOT_VERIFIED.
- Runtime smoke remains NOT_EXECUTED.
- Authenticated functional smoke remains NOT_EXECUTED.
- Track A, Beta, F13, and release approval remain NOT_GRANTED.
