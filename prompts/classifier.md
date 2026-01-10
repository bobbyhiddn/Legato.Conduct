# Thread Classification Prompt

You are classifying segments of a voice transcript for the LEGATO system.

## Your Task

For each thread, determine:

1. **Type**: Is this KNOWLEDGE (an insight, concept, reflection) or PROJECT (something to build/implement)?

2. **If KNOWLEDGE**, categorize as:
   - EPIPHANY: Major breakthrough or insight (rare - genuine "aha" moments)
   - CONCEPT: Technical definition or explanation of how something works
   - REFLECTION: Personal thought, observation, or musing about a topic
   - GLIMMER: A fleeting creative spark or poetic moment - NOT a request, NOT actionable
   - REMINDER: Note to self about something to remember (no implementation)
   - WORKLOG: Summary of work already completed

   **GLIMMER is NOT for:**
   - Requests to build/create/make something (those are PROJECTS)
   - Ideas that have clear implementation (those are PROJECTS)
   - Action items with deliverables (those are PROJECTS)

3. **If PROJECT**, determine:
   - Scope: NOTE (single feature, quick implementation) or CHORD (multi-phase, complex)
   - Name: Suggest a slug-friendly project name

4. **For all**, extract:
   - Domain tags (2-5 relevant topics)
   - Key phrases (distinctive terms that identify this content)
   - Title/summary (one line)

## Classification Signals

### KNOWLEDGE indicators:
- Past tense reflection ("I realized...", "I've been thinking...")
- Conceptual explanation ("The way X works is...")
- Insight framing ("What's interesting is...")
- Pure notes to self ("I need to remember...", "Don't forget...")
- Observations with no call to action

**KNOWLEDGE is passive** - it captures thoughts, not requests. If someone is *asking* for something to be done, it's a PROJECT.

### PROJECT indicators:
- Future tense intent ("I want to build...", "We should create...")
- **Direct commands/requests** ("Create a...", "Make a...", "Set up a...", "Build a...", "Add a...")
- **Repository/codebase references** ("repo", "repository", "codebase", "project", "app", "tool", "script")
- Technical specification ("It should have X, Y, Z features...")
- Implementation details ("Using Python, we could...")
- Scope estimation ("This would take about...")
- **Action verbs with artifacts** (create + readme, build + API, make + component, add + feature)

### Scope determination:
- NOTE: Single feature, <1 day work, no architecture needed
- CHORD: Multiple components, >1 day work, needs design doc

## Output Format

Return JSON array:
```json
[
  {
    "id": "thread-001",
    "type": "KNOWLEDGE",
    "knowledge_category": "epiphany",
    "title": "Oracle Machines as AI Intuition Framework",
    "domain_tags": ["ai", "turing", "theory", "intuition"],
    "key_phrases": ["oracle machine", "intuition engine", "ordinal logic"],
    "description": "Insight connecting Turing's oracle machines to modern AI behavior"
  },
  {
    "id": "thread-002",
    "type": "PROJECT",
    "project_scope": "note",
    "project_name": "mcp-bedrock-adapter",
    "title": "MCP adapter for AWS Bedrock in classified environments",
    "description": "Build an MCP server that wraps AWS Bedrock API for use in JWICS",
    "domain_tags": ["mcp", "aws", "bedrock", "classified"],
    "key_phrases": ["bedrock adapter", "jwics", "mcp server"]
  }
]
```

## Important Notes

- **Direct requests to create/build something are PROJECTS** - even short requests like "create a repo with a readme" should be classified as PROJECT, not KNOWLEDGE
- If a thread contains both knowledge and project elements, classify it as "MIXED" and provide both sets of fields
- Be conservative with EPIPHANY - reserve for genuinely novel insights
- GLIMMER is for poetic/creative moments, not for actionable ideas
- Project names should be lowercase, hyphenated slugs
- When in doubt between KNOWLEDGE and PROJECT: if the user is asking for something to be *made*, it's a PROJECT
