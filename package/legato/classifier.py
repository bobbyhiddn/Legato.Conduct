"""
LEGATO Classification Engine.

Transforms raw voice transcripts into structured, actionable items.
Operates in two phases: parsing (segmentation) and classification (categorization + routing).
"""

import os
import sys
import json
import argparse
import math
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional
from datetime import datetime
from pathlib import Path


class ThreadType(Enum):
    """Type of content in a thread. Everything is KNOWLEDGE in the new ontology."""
    KNOWLEDGE = "KNOWLEDGE"


class KnowledgeCategory(Enum):
    """Default categories for knowledge artifacts."""
    EPIPHANY = "epiphany"      # Major breakthrough or insight
    CONCEPT = "concept"        # Technical definition or explanation
    REFLECTION = "reflection"  # Personal thought or observation
    GLIMMER = "glimmer"        # Quick idea seed for future exploration
    REMINDER = "reminder"      # Action item or follow-up
    WORKLOG = "worklog"        # Summary of work done


# Default category definitions (used when none provided)
DEFAULT_CATEGORIES = [
    {"name": "epiphany", "display_name": "Epiphany", "description": "Major breakthrough or insight - genuine 'aha' moments"},
    {"name": "concept", "display_name": "Concept", "description": "Technical definition, explanation, or implementation idea"},
    {"name": "reflection", "display_name": "Reflection", "description": "Personal thought, observation, or musing"},
    {"name": "glimmer", "display_name": "Glimmer", "description": "A captured moment - photographing a feeling. Poetic, evocative, sensory"},
    {"name": "reminder", "display_name": "Reminder", "description": "Note to self about something to remember"},
    {"name": "worklog", "display_name": "Worklog", "description": "Summary of work already completed"},
]


class ChordScope(Enum):
    """Scope for chord implementations (when needs_chord is true)."""
    NOTE = "note"    # Single PR, quick implementation
    CHORD = "chord"  # Multi-PR, complex implementation


@dataclass
class ClassifiedThread:
    """A classified thread from a transcript.

    In the new ontology, everything is KNOWLEDGE. Items that need
    implementation have needs_chord=True with chord escalation fields.
    """

    id: str
    raw_text: str
    thread_type: ThreadType  # Always KNOWLEDGE now

    # Knowledge fields (always present)
    knowledge_category: Optional[KnowledgeCategory] = None
    knowledge_title: Optional[str] = None
    description: Optional[str] = None

    # Chord escalation fields
    needs_chord: bool = False
    chord_name: Optional[str] = None
    chord_scope: Optional[ChordScope] = None

    # Correlation results
    correlation_score: float = 0.0
    correlation_matches: list = field(default_factory=list)
    correlation_action: str = "CREATE"  # CREATE, APPEND, QUEUE

    # Metadata
    domain_tags: list = field(default_factory=list)
    key_phrases: list = field(default_factory=list)
    source_id: Optional[str] = None
    parsed_at: Optional[str] = None

    def to_dict(self) -> dict:
        """Convert to dictionary for JSON serialization."""
        # Handle both enum and string categories
        if self.knowledge_category is None:
            category_value = None
        elif isinstance(self.knowledge_category, str):
            category_value = self.knowledge_category
        else:
            category_value = self.knowledge_category.value

        result = {
            "id": self.id,
            "raw_text": self.raw_text,
            "type": self.thread_type.value,
            "knowledge_category": category_value,
            "knowledge_title": self.knowledge_title,
            "description": self.description,
            "needs_chord": self.needs_chord,
            "correlation_score": self.correlation_score,
            "correlation_matches": self.correlation_matches,
            "correlation_action": self.correlation_action,
            "domain_tags": self.domain_tags,
            "key_phrases": self.key_phrases,
            "source_id": self.source_id,
            "parsed_at": self.parsed_at,
        }

        # Only include chord fields if needs_chord is true
        if self.needs_chord:
            result["chord_name"] = self.chord_name
            result["chord_scope"] = self.chord_scope.value if self.chord_scope else None

        return result

    @classmethod
    def from_dict(cls, data: dict, valid_categories: set = None) -> "ClassifiedThread":
        """Create from dictionary.

        Args:
            data: Thread data dictionary
            valid_categories: Set of valid category names (for dynamic categories)

        Raises:
            TypeError: If data is not a dictionary
        """
        # Validate input type, with fallback for stringified JSON
        if isinstance(data, str):
            # Try to parse as JSON in case it's a stringified object
            try:
                data = json.loads(data)
            except json.JSONDecodeError:
                raise TypeError(
                    f"ClassifiedThread.from_dict() expects a dict, got unparseable string: {repr(data)[:100]}"
                )

        if not isinstance(data, dict):
            raise TypeError(
                f"ClassifiedThread.from_dict() expects a dict, got {type(data).__name__}: {repr(data)[:100]}"
            )

        # Always KNOWLEDGE in new ontology
        thread_type = ThreadType.KNOWLEDGE

        knowledge_category = None
        if data.get("knowledge_category"):
            # Normalize to lowercase to handle Claude returning EPIPHANY vs epiphany
            category_value = data["knowledge_category"].lower()

            # Try enum first, then allow any valid category string
            try:
                knowledge_category = KnowledgeCategory(category_value)
            except ValueError:
                # Not a default enum value - check if it's a valid dynamic category
                if valid_categories and category_value in valid_categories:
                    # Store as string value wrapped in a simple object for compatibility
                    knowledge_category = category_value  # Will be handled specially
                else:
                    # Unknown category - fall back to concept
                    print(f"Warning: Unknown category '{category_value}', using 'concept'", file=sys.stderr)
                    knowledge_category = KnowledgeCategory.CONCEPT

        chord_scope = None
        if data.get("chord_scope"):
            # Normalize to lowercase
            scope_value = data["chord_scope"].lower()
            chord_scope = ChordScope(scope_value)

        return cls(
            id=data.get("id", ""),
            raw_text=data.get("raw_text", data.get("text", "")),
            thread_type=thread_type,
            knowledge_category=knowledge_category,
            knowledge_title=data.get("knowledge_title", data.get("title")),
            description=data.get("description"),
            needs_chord=data.get("needs_chord", False),
            chord_name=data.get("chord_name"),
            chord_scope=chord_scope,
            correlation_score=data.get("correlation_score", 0.0),
            correlation_matches=data.get("correlation_matches", []),
            correlation_action=data.get("correlation_action", "CREATE"),
            domain_tags=data.get("domain_tags", []),
            key_phrases=data.get("key_phrases", []),
            source_id=data.get("source_id"),
            parsed_at=data.get("parsed_at"),
        )


def load_prompt(prompt_name: str) -> str:
    """Load a prompt from the prompts directory."""
    prompts_dir = Path(__file__).parent.parent.parent / "prompts"
    prompt_file = prompts_dir / f"{prompt_name}.md"

    if prompt_file.exists():
        return prompt_file.read_text()

    raise FileNotFoundError(f"Prompt not found: {prompt_file}")


def call_claude(system_prompt: str, user_input: str, max_tokens: int = 4096) -> tuple[str, str]:
    """Call Claude API with the given prompts.

    Returns:
        Tuple of (response_text, stop_reason)
    """
    try:
        import anthropic
    except ImportError:
        raise RuntimeError("anthropic package not installed")

    client = anthropic.Anthropic()

    message = client.messages.create(
        model="claude-opus-4-5-20251101",
        max_tokens=max_tokens,
        system=system_prompt,
        messages=[{"role": "user", "content": user_input}]
    )

    return message.content[0].text, message.stop_reason


def repair_truncated_json(text: str) -> str:
    """Attempt to repair truncated JSON array."""
    text = text.strip()

    # Count open brackets/braces
    open_brackets = text.count('[') - text.count(']')
    open_braces = text.count('{') - text.count('}')

    # Check if we're in a string (odd number of unescaped quotes)
    in_string = False
    i = len(text) - 1
    while i >= 0:
        if text[i] == '"' and (i == 0 or text[i-1] != '\\'):
            in_string = not in_string
            break
        i -= 1

    # If in string, close it
    if in_string:
        text += '"'

    # Close any open braces/brackets
    text += '}' * open_braces
    text += ']' * open_brackets

    return text


# =============================================================================
# Embedding-based Classification Utilities
# =============================================================================

def get_embeddings(texts: list[str], model: str = "text-embedding-3-small") -> list[list[float]]:
    """Generate embeddings for a list of texts using OpenAI API.

    Args:
        texts: List of texts to embed
        model: OpenAI embedding model to use

    Returns:
        List of embedding vectors (each is a list of floats)
    """
    try:
        import openai
    except ImportError:
        print("Warning: openai package not installed, skipping embeddings", file=sys.stderr)
        return []

    api_key = os.environ.get('OPENAI_API_KEY')
    if not api_key:
        print("Warning: No OPENAI_API_KEY, skipping embeddings", file=sys.stderr)
        return []

    try:
        client = openai.OpenAI(api_key=api_key)
        response = client.embeddings.create(
            model=model,
            input=texts
        )
        return [item.embedding for item in response.data]
    except Exception as e:
        print(f"Warning: Embedding generation failed: {e}", file=sys.stderr)
        return []


def cosine_similarity(vec1: list[float], vec2: list[float]) -> float:
    """Compute cosine similarity between two vectors.

    Args:
        vec1: First vector
        vec2: Second vector

    Returns:
        Cosine similarity score between -1 and 1
    """
    if not vec1 or not vec2 or len(vec1) != len(vec2):
        return 0.0

    dot_product = sum(a * b for a, b in zip(vec1, vec2))
    norm1 = math.sqrt(sum(a * a for a in vec1))
    norm2 = math.sqrt(sum(b * b for b in vec2))

    if norm1 == 0 or norm2 == 0:
        return 0.0

    return dot_product / (norm1 * norm2)


def compute_category_similarities(
    thread_text: str,
    category_definitions: list[dict],
    category_embeddings: list[list[float]] = None
) -> list[dict]:
    """Compute similarity scores between a thread and all categories.

    Args:
        thread_text: The text content of the thread
        category_definitions: List of category dicts with name, description
        category_embeddings: Pre-computed category embeddings (optional, will compute if not provided)

    Returns:
        List of dicts with category name and similarity score, sorted by score descending
    """
    if not category_definitions:
        return []

    # Get or compute category embeddings
    if not category_embeddings:
        category_texts = [
            f"{cat.get('display_name', cat.get('name', ''))}: {cat.get('description', '')}"
            for cat in category_definitions
        ]
        category_embeddings = get_embeddings(category_texts)

    if not category_embeddings:
        return []

    # Get thread embedding
    thread_embeddings = get_embeddings([thread_text])
    if not thread_embeddings:
        return []

    thread_embedding = thread_embeddings[0]

    # Compute similarities
    similarities = []
    for i, cat in enumerate(category_definitions):
        if i < len(category_embeddings):
            score = cosine_similarity(thread_embedding, category_embeddings[i])
            similarities.append({
                "name": cat.get("name", ""),
                "display_name": cat.get("display_name", cat.get("name", "")),
                "score": round(score, 4)
            })

    # Sort by score descending
    similarities.sort(key=lambda x: x["score"], reverse=True)
    return similarities


def parse_threads(transcript: str, source_id: Optional[str] = None) -> list[dict]:
    """
    Parse transcript into threads using Claude.

    Args:
        transcript: Raw transcript text
        source_id: Optional source identifier

    Returns:
        List of parsed thread dictionaries
    """
    prompt = """Parse this voice transcript into logical threads.
Each thread should be a coherent segment that discusses one topic.
Return a JSON array of threads with id, text, and summary fields.

Return format:
```json
[
  {"id": "thread-001", "text": "...", "summary": "Brief summary"}
]
```"""

    # Use higher token limit for parsing since output includes full transcript text
    response, stop_reason = call_claude(prompt, transcript, max_tokens=16384)

    # Extract JSON from response
    text = response.strip()
    if "```json" in text:
        text = text.split("```json")[1].split("```")[0]
    elif "```" in text:
        text = text.split("```")[1].split("```")[0]

    # Try to parse JSON, with repair attempt if truncated
    try:
        threads = json.loads(text)
    except json.JSONDecodeError as e:
        if stop_reason == "max_tokens":
            print(f"Warning: Response truncated, attempting JSON repair...", file=sys.stderr)
            repaired = repair_truncated_json(text)
            try:
                threads = json.loads(repaired)
                print(f"JSON repair successful, recovered {len(threads)} threads", file=sys.stderr)
            except json.JSONDecodeError:
                raise ValueError(
                    f"Failed to parse truncated JSON response. "
                    f"Original error: {e}. Consider splitting the transcript into smaller chunks."
                )
        else:
            raise ValueError(f"Failed to parse Claude response as JSON: {e}")

    # Add metadata
    for i, thread in enumerate(threads):
        thread["id"] = f"thread-{i+1:03d}"
        thread["source_id"] = source_id
        thread["parsed_at"] = datetime.utcnow().isoformat() + "Z"

    return threads


def check_correlation(thread: dict, pit_url: str = None, token: str = None) -> dict:
    """Check for similar existing entries in Pit.

    Args:
        thread: Parsed thread dictionary
        pit_url: Base URL for Pit API
        token: Authentication token

    Returns:
        Correlation result with action recommendation
    """
    import os
    import requests

    pit_url = pit_url or os.environ.get('PIT_URL', 'https://legato-pit.fly.dev')
    token = token or os.environ.get('SYSTEM_PAT')

    if not token:
        print("Warning: No SYSTEM_PAT, skipping correlation check", file=sys.stderr)
        return {'action': 'CREATE', 'score': 0.0, 'matches': []}

    try:
        response = requests.post(
            f'{pit_url}/memory/api/correlate',
            headers={
                'Authorization': f'Bearer {token}',
                'Content-Type': 'application/json',
            },
            json={
                'title': thread.get('summary', ''),
                'content': thread.get('text', thread.get('raw_text', '')),
                'key_phrases': thread.get('key_phrases', []),
                'needs_chord': thread.get('needs_chord', False),
            },
            timeout=30,
        )

        if response.ok:
            return response.json()
        else:
            print(f"Correlation check failed: {response.status_code}", file=sys.stderr)
            return {'action': 'CREATE', 'score': 0.0, 'matches': []}

    except Exception as e:
        print(f"Correlation check error: {e}", file=sys.stderr)
        return {'action': 'CREATE', 'score': 0.0, 'matches': []}


def report_pre_classify_stage(
    run_id: str,
    threads: list[dict],
    category_definitions: list[dict] = None,
    pit_url: str = None,
    token: str = None
) -> bool:
    """Report pre-classify stage to Pit's pipeline status endpoint.

    Posts available categories and motif previews before classification begins.
    This enables debug visibility in Pit for the classification pipeline.

    Args:
        run_id: GitHub workflow run ID
        threads: List of parsed thread dictionaries
        category_definitions: Category definitions (uses DEFAULT_CATEGORIES if None)
        pit_url: Base URL for Pit API
        token: Authentication token (SYSTEM_PAT)

    Returns:
        True if report was sent successfully, False otherwise
    """
    import requests

    pit_url = pit_url or os.environ.get('PIT_URL', 'https://legato-pit.fly.dev')
    token = token or os.environ.get('SYSTEM_PAT')

    if not token:
        print("Warning: No SYSTEM_PAT, skipping pre-classify stage report", file=sys.stderr)
        return False

    if not run_id:
        print("Warning: No run_id provided, skipping pre-classify stage report", file=sys.stderr)
        return False

    # Use provided categories or fall back to defaults
    categories = category_definitions if category_definitions else DEFAULT_CATEGORIES

    # Format categories for the API
    formatted_categories = [
        {
            "name": cat.get("name", ""),
            "display_name": cat.get("display_name", cat.get("name", "")),
            "description": cat.get("description", "")
        }
        for cat in categories
    ]

    # Format motifs (threads) with previews
    motifs = []
    for thread in threads:
        thread_id = thread.get("id", "")
        content = thread.get("text", thread.get("raw_text", ""))
        preview = content[:200] if content else ""

        motif = {
            "id": thread_id,
            "preview": preview
        }

        # Add title if available (from summary field)
        if thread.get("summary"):
            motif["title"] = thread["summary"]

        motifs.append(motif)

    # Build the request payload
    payload = {
        "run_id": str(run_id),
        "stage": "pre-classify",
        "status": "started",
        "details": {
            "categories": formatted_categories,
            "motifs": motifs
        }
    }

    try:
        response = requests.post(
            f'{pit_url}/memory/api/pipeline/status',
            headers={
                'Authorization': f'Bearer {token}',
                'Content-Type': 'application/json',
            },
            json=payload,
            timeout=30,
        )

        if response.ok:
            result = response.json()
            if result.get('success'):
                print(f"Pre-classify stage reported: {len(formatted_categories)} categories, {len(motifs)} motifs", file=sys.stderr)
                return True
            else:
                print(f"Pre-classify stage report failed: {result.get('message', 'Unknown error')}", file=sys.stderr)
                return False
        else:
            print(f"Pre-classify stage report failed: HTTP {response.status_code}", file=sys.stderr)
            return False

    except Exception as e:
        print(f"Pre-classify stage report error: {e}", file=sys.stderr)
        return False


def build_dynamic_classifier_prompt(
    category_definitions: list[dict],
    similarity_scores: list[dict] = None
) -> str:
    """Build a classifier prompt with dynamic category definitions and similarity scores.

    Args:
        category_definitions: List of category dicts with name, display_name, description
        similarity_scores: Optional list of dicts with category name and similarity score

    Returns:
        Complete classifier prompt string
    """
    # Build category list for prompt
    category_lines = []
    for cat in category_definitions:
        name = cat.get('name', '').upper()
        desc = cat.get('description', cat.get('display_name', ''))
        category_lines.append(f"   - {name}: {desc}")

    categories_block = "\n".join(category_lines)
    category_names = [cat['name'].lower() for cat in category_definitions]

    # Build similarity guidance block if scores provided
    similarity_block = ""
    if similarity_scores:
        similarity_lines = []
        for s in similarity_scores[:5]:  # Top 5 matches
            score_pct = int(s['score'] * 100)
            similarity_lines.append(f"   - {s['name'].upper()}: {score_pct}% semantic match")
        similarity_block = f"""
## Semantic Analysis (Embedding Similarity)

The following shows how semantically similar this thread is to each category based on embedding comparison:
{chr(10).join(similarity_lines)}

**Use these scores as a strong signal.** The highest-scoring category is likely the best fit unless the content clearly contradicts it.
"""

    return f"""# Thread Classification Prompt

You are classifying segments of a voice transcript for the LEGATO system.

## Core Principle

**Everything becomes a Note first.** All threads are classified as KNOWLEDGE and stored in the Library. If a thread describes something that needs implementation, it is flagged with `needs_chord: true` for escalation.
{similarity_block}
## Your Task

For each thread:

1. **Categorize** as one of:
{categories_block}

2. **Determine if it needs a Chord** (`needs_chord`):
   - Set `true` if this describes something to build/implement
   - Provide `chord_name` (slug-friendly) when true
   - Leave `false` for pure knowledge/reflection

3. **Extract metadata**:
   - Domain tags (2-5 relevant topics)
   - Key phrases (distinctive terms)
   - Title/summary (one line)

## Output Format

Return JSON - ALL items are type KNOWLEDGE:

```json
{{
  "type": "KNOWLEDGE",
  "knowledge_category": "{category_names[0] if category_names else 'concept'}",
  "title": "Title here",
  "description": "Brief description",
  "domain_tags": ["tag1", "tag2"],
  "key_phrases": ["phrase1", "phrase2"],
  "needs_chord": false
}}
```

Valid categories are: {', '.join(category_names)}
"""


def classify_threads(threads: list[dict], skip_correlation: bool = False,
                     category_definitions: list[dict] = None) -> list[ClassifiedThread]:
    """
    Classify parsed threads into KNOWLEDGE types with correlation checking.

    Uses embedding-based similarity scores to guide Claude's classification when
    category definitions are provided.

    Args:
        threads: List of parsed thread dictionaries
        skip_correlation: If True, skip Pit correlation check (for offline/testing)
        category_definitions: Optional list of category definitions from Pit

    Returns:
        List of ClassifiedThread objects
    """
    # Pre-compute category embeddings once (for efficiency)
    category_embeddings = None
    if category_definitions:
        valid_categories = {cat['name'].lower() for cat in category_definitions}
        print(f"Using dynamic categories: {', '.join(valid_categories)}", file=sys.stderr)

        # Generate embeddings for all categories
        category_texts = [
            f"{cat.get('display_name', cat.get('name', ''))}: {cat.get('description', '')}"
            for cat in category_definitions
        ]
        category_embeddings = get_embeddings(category_texts)
        if category_embeddings:
            print(f"Generated embeddings for {len(category_embeddings)} categories", file=sys.stderr)
        else:
            print("Warning: Could not generate category embeddings, proceeding without similarity scores", file=sys.stderr)
    else:
        valid_categories = {c.value for c in KnowledgeCategory}

    results = []

    for thread in threads:
        thread_text = thread.get("text", thread.get("raw_text", ""))

        # Compute similarity scores if we have category embeddings
        similarity_scores = None
        if category_definitions and category_embeddings:
            similarity_scores = compute_category_similarities(
                thread_text,
                category_definitions,
                category_embeddings
            )
            if similarity_scores:
                top_match = similarity_scores[0]
                print(f"  Thread {thread.get('id', '?')}: Top match = {top_match['name']} ({int(top_match['score']*100)}%)", file=sys.stderr)

        # Build prompt with similarity guidance
        if category_definitions:
            classifier_prompt = build_dynamic_classifier_prompt(category_definitions, similarity_scores)
        else:
            classifier_prompt = load_prompt("classifier")

        response, _ = call_claude(classifier_prompt, thread_text)

        # Parse classification response
        text = response.strip()
        if "```json" in text:
            text = text.split("```json")[1].split("```")[0]
        elif "```" in text:
            text = text.split("```")[1].split("```")[0]

        try:
            classification = json.loads(text)
            if isinstance(classification, list):
                classification = classification[0]
        except json.JSONDecodeError:
            # Default to glimmer if parsing fails
            classification = {
                "type": "KNOWLEDGE",
                "knowledge_category": "glimmer",
                "title": thread.get("summary", "Untitled"),
                "domain_tags": [],
                "key_phrases": []
            }

        # Merge with original thread data
        classification["raw_text"] = thread_text
        classification["id"] = thread.get("id", "")
        classification["source_id"] = thread.get("source_id")
        classification["parsed_at"] = thread.get("parsed_at")

        classified = ClassifiedThread.from_dict(classification, valid_categories)

        # Run correlation check against Pit
        if not skip_correlation:
            correlation = check_correlation({
                'summary': classified.knowledge_title,
                'text': classified.raw_text,
                'key_phrases': classified.key_phrases,
                'needs_chord': classified.needs_chord,
            })

            classified.correlation_score = correlation.get('score', 0.0)
            classified.correlation_matches = correlation.get('matches', [])
            classified.correlation_action = correlation.get('action', 'CREATE')

            # Store recommendation for knowledge.py to handle
            if correlation.get('recommendation'):
                classified.correlation_matches.insert(0, {
                    'recommendation': correlation['recommendation']
                })

        results.append(classified)

    return results


def compute_tag_similarity(tags1: list, tags2: list) -> float:
    """Compute Jaccard similarity between two tag lists."""
    if not tags1 or not tags2:
        return 0.0
    set1, set2 = set(t.lower() for t in tags1), set(t.lower() for t in tags2)
    intersection = len(set1 & set2)
    union = len(set1 | set2)
    return intersection / union if union > 0 else 0.0


def find_chord_groups(classified_threads: list[ClassifiedThread],
                      similarity_threshold: float = 0.3) -> list[list[int]]:
    """Find groups of threads that might belong to the same chord.

    Uses domain tag similarity to identify related threads.

    Args:
        classified_threads: List of classified threads
        similarity_threshold: Minimum tag similarity to consider grouping

    Returns:
        List of groups, where each group is a list of thread indices
    """
    # Find threads that need chords
    chord_indices = [i for i, t in enumerate(classified_threads) if t.needs_chord]

    if len(chord_indices) <= 1:
        return [[i] for i in chord_indices]  # No grouping possible

    # Build adjacency based on tag similarity
    adjacency = {i: [] for i in chord_indices}
    for i, idx1 in enumerate(chord_indices):
        for idx2 in chord_indices[i+1:]:
            t1, t2 = classified_threads[idx1], classified_threads[idx2]
            similarity = compute_tag_similarity(t1.domain_tags, t2.domain_tags)
            if similarity >= similarity_threshold:
                adjacency[idx1].append(idx2)
                adjacency[idx2].append(idx1)

    # Find connected components (groups)
    visited = set()
    groups = []

    for start in chord_indices:
        if start in visited:
            continue
        # BFS to find connected component
        group = []
        queue = [start]
        while queue:
            node = queue.pop(0)
            if node in visited:
                continue
            visited.add(node)
            group.append(node)
            queue.extend(n for n in adjacency[node] if n not in visited)
        if group:
            groups.append(group)

    return groups


def ask_claude_about_grouping(threads: list[ClassifiedThread], group_indices: list[int]) -> dict:
    """Ask Claude if grouped threads should form a single multi-note chord.

    Args:
        threads: All classified threads
        group_indices: Indices of threads in this potential group

    Returns:
        Dict with 'should_group', 'chord_name', 'reasoning'
    """
    if len(group_indices) <= 1:
        idx = group_indices[0]
        return {
            'should_group': False,
            'chord_name': threads[idx].chord_name,
            'thread_ids': [threads[idx].id],
        }

    # Build context for Claude
    thread_summaries = []
    for idx in group_indices:
        t = threads[idx]
        thread_summaries.append(f"""
Thread {t.id}:
- Title: {t.knowledge_title}
- Description: {t.description}
- Domain Tags: {', '.join(t.domain_tags)}
- Proposed Chord Name: {t.chord_name}
""")

    prompt = f"""You are analyzing multiple knowledge threads that have been flagged for chord (project) creation.

These threads have overlapping domain tags and may be related. Determine if they should:
1. Be grouped into a SINGLE multi-note chord (one project with multiple linked notes)
2. Remain as separate individual chords

Consider:
- Do they describe parts of the same system/feature?
- Would implementing them together make sense?
- Are they conceptually unified enough for one project?

Threads to analyze:
{"".join(thread_summaries)}

Return JSON:
```json
{{
  "should_group": true/false,
  "chord_name": "unified-chord-name-if-grouped",
  "reasoning": "Brief explanation"
}}
```
"""

    try:
        response, _ = call_claude(prompt, "Analyze these threads for grouping.", max_tokens=1024)

        text = response.strip()
        if "```json" in text:
            text = text.split("```json")[1].split("```")[0]
        elif "```" in text:
            text = text.split("```")[1].split("```")[0]

        result = json.loads(text)
        result['thread_ids'] = [threads[idx].id for idx in group_indices]
        return result

    except Exception as e:
        print(f"Warning: Grouping analysis failed: {e}", file=sys.stderr)
        # Default to not grouping
        return {
            'should_group': False,
            'chord_name': threads[group_indices[0]].chord_name,
            'thread_ids': [threads[idx].id for idx in group_indices],
        }


def group_chords(classified_threads: list[ClassifiedThread]) -> list[dict]:
    """Analyze classified threads and group related ones into multi-note chords.

    Args:
        classified_threads: List of individually classified threads

    Returns:
        List of chord group dicts with:
        - chord_name: Name for the chord
        - thread_ids: List of thread IDs in this chord
        - is_multi_note: Whether this is a multi-note chord
    """
    # Find potential groups based on tag similarity
    groups = find_chord_groups(classified_threads)

    if not groups:
        return []

    chord_groups = []

    for group_indices in groups:
        if len(group_indices) == 1:
            # Single thread, no grouping needed
            t = classified_threads[group_indices[0]]
            chord_groups.append({
                'chord_name': t.chord_name,
                'thread_ids': [t.id],
                'is_multi_note': False,
            })
        else:
            # Multiple threads - ask Claude if they should be grouped
            result = ask_claude_about_grouping(classified_threads, group_indices)

            if result.get('should_group', False):
                chord_groups.append({
                    'chord_name': result['chord_name'],
                    'thread_ids': result['thread_ids'],
                    'is_multi_note': True,
                    'reasoning': result.get('reasoning', ''),
                })
            else:
                # Keep as separate chords
                for idx in group_indices:
                    t = classified_threads[idx]
                    chord_groups.append({
                        'chord_name': t.chord_name,
                        'thread_ids': [t.id],
                        'is_multi_note': False,
                    })

    return chord_groups


def main():
    """CLI entry point for classifier."""
    parser = argparse.ArgumentParser(description="LEGATO Classification Engine")
    parser.add_argument(
        "--phase",
        choices=["parse", "pre-classify", "classify", "full"],
        default="full",
        help="Processing phase: parse → pre-classify → classify"
    )
    parser.add_argument(
        "--input",
        help="Input file (transcript or threads.json)"
    )
    parser.add_argument(
        "--output",
        required=True,
        help="Output JSON file"
    )
    args = parser.parse_args()

    # Get transcript from input or environment
    if args.phase in ("parse", "full"):
        if args.input:
            if Path(args.input).exists():
                transcript = Path(args.input).read_text()
            else:
                transcript = args.input
        elif os.environ.get("TRANSCRIPT"):
            transcript = os.environ["TRANSCRIPT"]
        else:
            print("Error: No transcript provided", file=sys.stderr)
            sys.exit(1)

        source_id = os.environ.get("SOURCE_ID")
        threads = parse_threads(transcript, source_id)

        if args.phase == "parse":
            with open(args.output, "w") as f:
                json.dump(threads, f, indent=2)
            print(f"Parsed {len(threads)} threads")
            return

    # Pre-classify phase: report categories and motif previews to Pit
    if args.phase == "pre-classify":
        with open(args.input) as f:
            threads = json.load(f)

        # Load category definitions from environment (sent by Pit)
        category_definitions = None
        if os.environ.get("CATEGORY_DEFINITIONS"):
            try:
                category_definitions = json.loads(os.environ["CATEGORY_DEFINITIONS"])
                if category_definitions:
                    print(f"Loaded {len(category_definitions)} category definitions from Pit", file=sys.stderr)
            except json.JSONDecodeError as e:
                print(f"Warning: Failed to parse CATEGORY_DEFINITIONS: {e}", file=sys.stderr)

        # Report pre-classify stage to Pit for debug visibility
        run_id = os.environ.get("GITHUB_RUN_ID")
        if run_id:
            success = report_pre_classify_stage(
                run_id=run_id,
                threads=threads,
                category_definitions=category_definitions
            )
            if not success:
                print("Warning: Failed to report pre-classify stage to Pit", file=sys.stderr)
        else:
            print("Warning: No GITHUB_RUN_ID, skipping pre-classify report", file=sys.stderr)

        # Output the pre-classify data for debugging/inspection
        categories = category_definitions if category_definitions else DEFAULT_CATEGORIES
        output = {
            "stage": "pre-classify",
            "run_id": run_id,
            "categories": [
                {
                    "name": cat.get("name", ""),
                    "display_name": cat.get("display_name", cat.get("name", "")),
                    "description": cat.get("description", "")
                }
                for cat in categories
            ],
            "motifs": [
                {
                    "id": thread.get("id", ""),
                    "title": thread.get("summary", ""),
                    "preview": thread.get("text", thread.get("raw_text", ""))[:200]
                }
                for thread in threads
            ]
        }

        with open(args.output, "w") as f:
            json.dump(output, f, indent=2)

        print(f"Pre-classify stage: {len(output['categories'])} categories, {len(output['motifs'])} motifs")
        return

    if args.phase in ("classify", "full"):
        if args.phase == "classify":
            with open(args.input) as f:
                threads = json.load(f)

        # Load category definitions from environment (sent by Pit)
        category_definitions = None
        if os.environ.get("CATEGORY_DEFINITIONS"):
            try:
                category_definitions = json.loads(os.environ["CATEGORY_DEFINITIONS"])
                if category_definitions:
                    print(f"Loaded {len(category_definitions)} category definitions from Pit", file=sys.stderr)
            except json.JSONDecodeError as e:
                print(f"Warning: Failed to parse CATEGORY_DEFINITIONS: {e}", file=sys.stderr)

        classified = classify_threads(threads, category_definitions=category_definitions)

        # Group related chord threads into multi-note chords
        chord_groups = []
        needs_chord_count = sum(1 for t in classified if t.needs_chord)

        if needs_chord_count > 1:
            print(f"Analyzing {needs_chord_count} chord candidates for grouping...", file=sys.stderr)
            chord_groups = group_chords(classified)
            multi_note_count = sum(1 for g in chord_groups if g['is_multi_note'])
            if multi_note_count > 0:
                print(f"Created {multi_note_count} multi-note chord groups", file=sys.stderr)

        # Build output with chord grouping info
        output = {
            'threads': [t.to_dict() for t in classified],
            'chord_groups': chord_groups,
        }

        with open(args.output, "w") as f:
            json.dump(output, f, indent=2)

        print(f"Classified {len(classified)} threads: {needs_chord_count} need chord escalation")


if __name__ == "__main__":
    main()
