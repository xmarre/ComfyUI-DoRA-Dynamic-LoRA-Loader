# State Manager sequence transport design

Status: investigation checkpoint; not yet an implementation specification. No production code changes.

## Verified boundaries

On 2026-09-19 the production PR heads are DoRA #79 `6c1ce0c6d812f854048dc19529163f71d68e186b`; Prompt Writer Plus #19 `74e3b66c0bba238f3213a216b0f9f06a5c5ebfd2` and #20 `1152064cb622fc2df6a886622225bf1936ed2a82`; Continuum Plus #20 `ccf50cddf83124707b866e4df3956e85d41dda4b`, #23 `eb197769ff856bc5d0768cfdaa6017c127741ced`, #24 `e9f8006e4d2d3a2381d26b6dfacf04f90da84416`. All remain open, with their existing stacked bases. Do not modify these implementation PRs for this design task.

State Manager Python and JS normalizers retain text boxes with role, slot, label and scalar text. They do not retain a sequence AST or format discriminator. Scalar STRING transport can preserve Timeline semantics if the string actually contains accepted headers; scalar typing alone does not prove information loss. The exact authored 1023-character value has not yet been recovered.

The reproduced direct path has queue_metadata identity, queued_match=True, timeline=False; Continuum PT209 resolves Auto to Fixed and physical_compiler_eligible=False. No PT208 is present. This proves the Timeline compiler was not invoked; it does not establish where intended boundaries were lost. Do not require Prompt Writer Apply to repair direct operation.

Impact 8.28.3 is reported by the runtime. Matching upstream source shows reproduce skips queue-time regeneration, but ImpactWildcardProcessor.doit still calls wildcard processing on populated_text. The installed file bytes/commit and actual handler registration order have not been obtained. Current DoRA bridge appends its handler and writes both Impact text fields; handler ordering can change populate behavior and must be resolved in the design.

Continuum AGENTS.md requires malformed/unknown prompt syntax to remain a diagnostic Fixed fallback, and prohibits new prompt-content execution stops. Preserve this contract. Continuum main has advanced to a5b8943844594545301b20d01af5d9e3fa38ae29 (repository-reference rename), while #20 still bases on bf25353d8bec44afea22c89717c4301ce13c2036. Inspect layered main plus PR deltas rather than assuming #24 equals the Patcher runtime.

Next: finish source/data-model audit, choose explicit persistent intent and wildcard boundary, record exact code map and acceptance fixtures, then replace this checkpoint with the full design and create a documentation-only PR.
