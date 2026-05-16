# Prompt Template: Change Impact Analysis from Patches

You are analyzing the impact of a software change.

## Rules

- Stay within the issue and provided patch evidence.
- Do not invent files, modules, APIs, tests, user workflows, or risks.
- Prefer concrete file paths and test files when evidence is available.
- If patch evidence is not provided, mark uncertain predictions as assumptions.
- Keep the requirement impact summary concise and behavior-focused.
- Use at most 3 affected functionalities, 5 impacted files, 5 test files, and 3 risk notes.
- Keep every list item under 20 words and the summary under 60 words.

## Output Format

Return JSON only:

```json
{
  "affected_functionality": [
    "Short behavior or capability affected by the change"
  ],
  "impacted_files": [
    "path/to/file.py"
  ],
  "regression_test_focus": [
    "path/to/test_file.py"
  ],
  "risk_notes": [
    "Specific regression risk grounded in the issue or patch"
  ],
  "requirement_impact_summary": "One concise paragraph describing the requirement-level impact.",
  "assumptions": [],
  "unsupported_details_avoided": []
}
```

## Task

Repository: `{repo}`

Language: `{language}`

Source: `{source}`

Task category: `{task_category}`

Problem statement:

```text
{problem_statement}
```

Patch evidence:

```diff
{patch_excerpt}
```

Test patch evidence:

```diff
{test_patch_excerpt}
```

Repository tree / path context:

```text
{repo_tree_excerpt}
```
