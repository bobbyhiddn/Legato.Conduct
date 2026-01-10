# Thread Classification Prompt

You are classifying segments of a voice transcript for the LEGATO system.

## Core Principle

**Everything becomes a Note first.** All threads are classified as KNOWLEDGE and stored in the Library. If a thread describes something that needs implementation, it is flagged with `needs_chord: true` for escalation.

## Your Task

For each thread:

1. **Categorize** as one of:
   - EPIPHANY: Major breakthrough or insight (rare - genuine "aha" moments)
   - CONCEPT: Technical definition, explanation, or implementation idea
   - REFLECTION: Personal thought, observation, or musing
   - GLIMMER: Fleeting creative spark or poetic moment (NOT actionable)
   - REMINDER: Note to self about something to remember
   - WORKLOG: Summary of work already completed

2. **Determine if it needs a Chord** (`needs_chord`):
   - Set `true` if this describes something to build/implement
   - Provide `chord_name` (slug-friendly) when true
   - Leave `false` for pure knowledge/reflection

3. **Extract metadata**:
   - Domain tags (2-5 relevant topics)
   - Key phrases (distinctive terms)
   - Title/summary (one line)

## Classification Signals

### Pure Knowledge (needs_chord: false):
- Past tense reflection ("I realized...", "I've been thinking...")
- Conceptual explanation ("The way X works is...")
- Insight framing ("What's interesting is...")
- Pure notes to self ("I need to remember...")
- Observations with no call to action

### Needs Chord (needs_chord: true):
- Future tense intent ("I want to build...", "We should create...")
- Direct commands/requests ("Create a...", "Make a...", "Build a...")
- Technical specification ("It should have X, Y, Z features...")
- Implementation details ("Using Python, we could...")
- Repository/codebase references ("repo", "project", "app", "tool")

### Chord Scope (when needs_chord is true):
- `chord_scope: note` - Single feature, <1 day work, simple
- `chord_scope: chord` - Multiple components, >1 day work, complex

## Output Format

Return JSON array - ALL items are type KNOWLEDGE:

```json
[
  {
    "id": "thread-001",
    "type": "KNOWLEDGE",
    "knowledge_category": "epiphany",
    "title": "Oracle Machines as AI Intuition Framework",
    "description": "Insight connecting Turing's oracle machines to modern AI behavior",
    "domain_tags": ["ai", "turing", "theory", "intuition"],
    "key_phrases": ["oracle machine", "intuition engine", "ordinal logic"],
    "needs_chord": false
  },
  {
    "id": "thread-002",
    "type": "KNOWLEDGE",
    "knowledge_category": "concept",
    "title": "MCP adapter for AWS Bedrock in classified environments",
    "description": "Build an MCP server that wraps AWS Bedrock API for use in JWICS",
    "domain_tags": ["mcp", "aws", "bedrock", "classified"],
    "key_phrases": ["bedrock adapter", "jwics", "mcp server"],
    "needs_chord": true,
    "chord_name": "mcp-bedrock-adapter",
    "chord_scope": "note"
  }
]
```

## Important Notes

- **Everything is KNOWLEDGE** - there is no PROJECT type
- Implementation ideas are CONCEPT or EPIPHANY with `needs_chord: true`
- The Note is created first, then escalated to a Chord if needed
- Be conservative with EPIPHANY - reserve for genuinely novel insights
- GLIMMER is for poetic/creative moments, never has `needs_chord: true`
- Chord names should be lowercase, hyphenated slugs
