"""
LEGATO Knowledge Module.

Handles extraction of knowledge artifacts from classified threads
and committing them to the Legato.Library repository.
"""

import os
import sys
import json
import argparse
import subprocess
import base64
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Optional

from .classifier import ClassifiedThread, ThreadType, KnowledgeCategory, ChordScope


@dataclass
class KnowledgeArtifact:
    """A knowledge artifact ready for the Library."""

    id: str
    title: str
    category: KnowledgeCategory
    content: str
    domain_tags: list = field(default_factory=list)
    key_phrases: list = field(default_factory=list)
    source_transcript: Optional[str] = None
    created: Optional[str] = None
    correlation_score: float = 0.0
    related: list = field(default_factory=list)

    # Chord escalation fields
    needs_chord: bool = False
    chord_name: Optional[str] = None
    chord_scope: Optional[str] = None  # "note" or "chord"

    def to_markdown(self) -> str:
        """Convert to markdown format for Library."""
        created = self.created or datetime.utcnow().isoformat() + "Z"

        frontmatter_lines = [
            "---",
            f"id: {self.id}",
            f'title: "{self.title}"',
            f"category: {self.category.value}",
            f"created: {created}",
            f"source_transcript: {self.source_transcript or 'unknown'}",
            f"domain_tags: {json.dumps(self.domain_tags)}",
            f"key_phrases: {json.dumps(self.key_phrases)}",
            f"correlation_score: {self.correlation_score}",
            f"related: {json.dumps(self.related)}",
            f"needs_chord: {str(self.needs_chord).lower()}",
        ]

        # Add chord fields if needs_chord is true
        if self.needs_chord:
            frontmatter_lines.append(f"chord_name: {self.chord_name or 'null'}")
            frontmatter_lines.append(f"chord_scope: {self.chord_scope or 'null'}")
            frontmatter_lines.append("chord_id: null")
            frontmatter_lines.append("chord_status: null")
            frontmatter_lines.append("chord_repo: null")

        frontmatter_lines.append("---")
        frontmatter = "\n".join(frontmatter_lines)

        return f"{frontmatter}\n\n{self.content}"

    def get_path(self) -> str:
        """Get the file path for this artifact."""
        date = datetime.utcnow().strftime("%Y-%m-%d")
        slug = self.id.split(".")[-1]
        return f"{self.category.value}s/{date}-{slug}.md"


def load_prompt(prompt_name: str) -> str:
    """Load a prompt from the prompts directory."""
    prompts_dir = Path(__file__).parent.parent.parent / "prompts"
    prompt_file = prompts_dir / f"{prompt_name}.md"

    if prompt_file.exists():
        return prompt_file.read_text()

    raise FileNotFoundError(f"Prompt not found: {prompt_file}")


def call_claude(system_prompt: str, user_input: str) -> str:
    """Call Claude API with the given prompts."""
    try:
        import anthropic
    except ImportError:
        raise RuntimeError("anthropic package not installed")

    client = anthropic.Anthropic()

    message = client.messages.create(
        model="claude-opus-4-5-20251101",
        max_tokens=4096,
        system=system_prompt,
        messages=[{"role": "user", "content": user_input}]
    )

    return message.content[0].text


def extract_knowledge(thread: ClassifiedThread) -> KnowledgeArtifact:
    """
    Extract a knowledge artifact from a classified thread.

    All threads are KNOWLEDGE in the new ontology. If needs_chord is True,
    the artifact will include chord escalation fields in frontmatter.

    Args:
        thread: A classified thread

    Returns:
        KnowledgeArtifact ready for commit
    """
    extractor_prompt = load_prompt("knowledge-extractor")

    # Prepare input for Claude
    input_data = {
        "thread_id": thread.id,
        "category": thread.knowledge_category.value if thread.knowledge_category else "glimmer",
        "title": thread.knowledge_title or "Untitled",
        "text": thread.raw_text,
        "domain_tags": thread.domain_tags,
        "key_phrases": thread.key_phrases,
        "source_id": thread.source_id,
    }

    # Add chord context if needs_chord
    if thread.needs_chord:
        input_data["needs_chord"] = True
        input_data["chord_name"] = thread.chord_name
        input_data["chord_scope"] = thread.chord_scope.value if thread.chord_scope else None

    response = call_claude(extractor_prompt, json.dumps(input_data, indent=2))

    # Extract markdown content from response
    content = response.strip()

    # Remove any code block markers if present
    if content.startswith("```markdown"):
        content = content[11:]
    if content.startswith("```"):
        content = content[3:]
    if content.endswith("```"):
        content = content[:-3]

    # Generate artifact ID
    category = thread.knowledge_category or KnowledgeCategory.GLIMMER
    slug = (thread.knowledge_title or thread.chord_name or thread.id).lower()
    slug = "".join(c if c.isalnum() or c == "-" else "-" for c in slug)
    slug = "-".join(filter(None, slug.split("-")))[:50]

    artifact_id = f"library.{category.value}s.{slug}"

    return KnowledgeArtifact(
        id=artifact_id,
        title=thread.knowledge_title or "Untitled",
        category=category,
        content=content,
        domain_tags=thread.domain_tags,
        key_phrases=thread.key_phrases,
        source_transcript=thread.source_id,
        created=datetime.utcnow().isoformat() + "Z",
        correlation_score=thread.correlation_score,
        related=[m.get("signal_id") for m in thread.correlation_matches if isinstance(m, dict) and m.get("score", 0) > 0.5],
        needs_chord=thread.needs_chord,
        chord_name=thread.chord_name,
        chord_scope=thread.chord_scope.value if thread.chord_scope else None,
    )


def group_chord_candidates(threads: list[ClassifiedThread], similarity_threshold: float = 0.5) -> dict:
    """
    Group threads that need chords by domain_tag similarity.

    When multiple threads in the same transcript discuss related topics,
    they should spawn as a single multi-note chord rather than separate chords.

    Args:
        threads: List of classified threads
        similarity_threshold: Minimum Jaccard similarity to group (0.0-1.0)

    Returns:
        Dict mapping group_id to list of threads:
        {
            "group-1": [thread1, thread2],  # These share enough domain_tags
            "group-2": [thread3],           # This is distinct
        }
    """
    # Filter to only chord candidates
    chord_threads = [t for t in threads if t.needs_chord]

    if not chord_threads:
        return {}

    if len(chord_threads) == 1:
        return {"group-1": chord_threads}

    # Calculate pairwise similarity based on domain_tags
    def jaccard_similarity(tags1: list, tags2: list) -> float:
        if not tags1 or not tags2:
            return 0.0
        set1, set2 = set(tags1), set(tags2)
        intersection = len(set1 & set2)
        union = len(set1 | set2)
        return intersection / union if union > 0 else 0.0

    # Simple greedy clustering
    groups = []
    used = set()

    for i, thread in enumerate(chord_threads):
        if i in used:
            continue

        # Start new group with this thread
        group = [thread]
        used.add(i)

        # Find similar threads
        for j, other in enumerate(chord_threads):
            if j in used:
                continue

            similarity = jaccard_similarity(thread.domain_tags, other.domain_tags)
            if similarity >= similarity_threshold:
                group.append(other)
                used.add(j)

        groups.append(group)

    # Convert to dict with group IDs
    result = {}
    for idx, group in enumerate(groups, 1):
        group_id = f"group-{idx}"

        # Generate a combined chord name from the group
        if len(group) > 1:
            # Use the most common domain tags
            all_tags = []
            for t in group:
                all_tags.extend(t.domain_tags or [])

            # Get most frequent tag as basis for name
            if all_tags:
                from collections import Counter
                common_tag = Counter(all_tags).most_common(1)[0][0]
                chord_name = f"{common_tag}-multi"
            else:
                chord_name = f"multi-chord-{idx}"

            # Update all threads in group with shared chord name and related IDs
            thread_ids = [t.id for t in group]
            for t in group:
                t.chord_name = chord_name
                # Add other thread IDs as related
                t.correlation_matches.append({
                    "grouped_threads": [tid for tid in thread_ids if tid != t.id],
                    "group_reason": "domain_tag_similarity"
                })

        result[group_id] = group

    return result


def commit_knowledge(artifact: KnowledgeArtifact, library_repo: Optional[str] = None) -> dict:
    """
    Commit a knowledge artifact to the Library repository.

    Args:
        artifact: The artifact to commit
        library_repo: Repository in owner/repo format (default from env)

    Returns:
        Commit result dictionary
    """
    library_repo = library_repo or os.environ.get("LIBRARY_REPO", "Legato/Legato.Library")
    token = os.environ.get("GH_TOKEN")

    if not token:
        raise RuntimeError("GH_TOKEN environment variable not set")

    content = artifact.to_markdown()
    file_path = artifact.get_path()
    commit_msg = f"Add {artifact.category.value}: {artifact.title}"

    # Encode content
    content_b64 = base64.b64encode(content.encode()).decode()

    # Check if file exists
    check_cmd = [
        "gh", "api",
        f"/repos/{library_repo}/contents/{file_path}",
        "--silent"
    ]

    result = subprocess.run(check_cmd, capture_output=True, text=True)
    file_exists = result.returncode == 0

    if file_exists:
        # Get SHA for update
        file_info = json.loads(result.stdout)
        sha = file_info.get("sha")

        update_cmd = [
            "gh", "api",
            "--method", "PUT",
            f"/repos/{library_repo}/contents/{file_path}",
            "-f", f"message={commit_msg}",
            "-f", f"content={content_b64}",
            "-f", f"sha={sha}"
        ]
        subprocess.run(update_cmd, check=True)
    else:
        create_cmd = [
            "gh", "api",
            "--method", "PUT",
            f"/repos/{library_repo}/contents/{file_path}",
            "-f", f"message={commit_msg}",
            "-f", f"content={content_b64}"
        ]
        subprocess.run(create_cmd, check=True)

    return {
        "repo": library_repo,
        "path": file_path,
        "action": "updated" if file_exists else "created",
        "artifact_id": artifact.id
    }


def handle_append(thread: ClassifiedThread, target_entry_id: str, commit: bool = False) -> dict:
    """Append content to an existing entry via Pit API.

    Args:
        thread: The classified thread with new content
        target_entry_id: Entry ID to append to
        commit: Whether to actually commit

    Returns:
        Result dict
    """
    import requests

    if not commit:
        return {
            "action": "would_append",
            "target_entry_id": target_entry_id,
            "thread_id": thread.id,
        }

    pit_url = os.environ.get('PIT_URL', 'https://legato-pit.fly.dev')
    token = os.environ.get('SYSTEM_PAT')

    if not token:
        raise RuntimeError("SYSTEM_PAT required for append")

    response = requests.post(
        f'{pit_url}/memory/api/append',
        headers={
            'Authorization': f'Bearer {token}',
            'Content-Type': 'application/json',
        },
        json={
            'entry_id': target_entry_id,
            'content': thread.raw_text,
            'source_transcript': thread.source_id,
        },
        timeout=30,
    )

    if response.ok:
        return {
            "action": "appended",
            "target_entry_id": target_entry_id,
            "thread_id": thread.id,
        }
    else:
        raise RuntimeError(f"Append failed: {response.status_code} - {response.text}")


def handle_queue_task(thread: ClassifiedThread, chord_repo: str, commit: bool = False) -> dict:
    """Queue a task on an existing chord instead of creating a new one.

    Args:
        thread: The classified thread
        chord_repo: The chord repo to queue the task on
        commit: Whether to actually commit

    Returns:
        Result dict
    """
    import requests

    if not commit:
        return {
            "action": "would_queue",
            "chord_repo": chord_repo,
            "thread_id": thread.id,
        }

    pit_url = os.environ.get('PIT_URL', 'https://legato-pit.fly.dev')
    token = os.environ.get('SYSTEM_PAT')

    if not token:
        raise RuntimeError("SYSTEM_PAT required for queue task")

    response = requests.post(
        f'{pit_url}/memory/api/queue-task',
        headers={
            'Authorization': f'Bearer {token}',
            'Content-Type': 'application/json',
        },
        json={
            'chord_repo': chord_repo,
            'title': thread.knowledge_title or thread.chord_name or 'New task',
            'description': thread.raw_text[:500],
            'source_entry_id': None,  # This is a new thread, not an existing entry
            'source_transcript': thread.source_id,
        },
        timeout=30,
    )

    if response.ok:
        data = response.json()
        return {
            "action": "queued_task",
            "chord_repo": chord_repo,
            "issue_url": data.get('issue_url'),
            "thread_id": thread.id,
        }
    else:
        raise RuntimeError(f"Queue task failed: {response.status_code} - {response.text}")


def get_recommendation(thread: ClassifiedThread) -> Optional[dict]:
    """Safely extract recommendation from correlation_matches.

    correlation_matches can contain various types of items:
    - Signal dicts with signal_id, score, title, path
    - Recommendation dicts with 'recommendation' key
    - Grouped thread dicts with 'grouped_threads' key
    - Potentially malformed items (strings, None, etc.)

    Returns the first valid recommendation dict, or None.
    """
    for match in thread.correlation_matches:
        # Skip non-dict items
        if not isinstance(match, dict):
            continue
        # Look for recommendation key
        rec = match.get('recommendation')
        if isinstance(rec, dict):
            return rec
    return None


def process_routing(routing_file: str, commit: bool = False) -> list[dict]:
    """
    Process classified threads from a routing file.

    Respects correlation_action from classification:
    - CREATE: Create new note in Library
    - APPEND: Append to existing note
    - QUEUE: Queue task on existing chord (instead of new chord)
    - SKIP: Skip processing (near-duplicate)

    Also groups related chord candidates so multiple related notes
    spawn as a single multi-note chord rather than separate chords.

    Args:
        routing_file: Path to routing.json
        commit: Whether to commit to Library

    Returns:
        List of results
    """
    with open(routing_file) as f:
        routing = json.load(f)

    # Validate routing structure
    if not isinstance(routing, list):
        raise ValueError(f"routing.json must be a list, got {type(routing).__name__}")

    # Parse all threads, filtering out invalid items
    threads = []
    for i, item in enumerate(routing):
        if not isinstance(item, dict):
            print(f"Warning: Skipping invalid item at index {i}: expected dict, got {type(item).__name__}: {repr(item)[:80]}", file=sys.stderr)
            continue
        try:
            threads.append(ClassifiedThread.from_dict(item))
        except (TypeError, ValueError) as e:
            print(f"Warning: Skipping invalid item at index {i}: {e}", file=sys.stderr)
            continue

    if not threads:
        print("Warning: No valid threads found in routing.json", file=sys.stderr)
        return []

    # Group chord candidates by domain_tag similarity
    # This modifies threads in-place to share chord_name
    chord_groups = group_chord_candidates(threads)
    if chord_groups:
        multi_groups = [g for g in chord_groups.values() if len(g) > 1]
        if multi_groups:
            print(f"Grouped {sum(len(g) for g in multi_groups)} chord candidates into {len(multi_groups)} multi-note chords")

    results = []

    for thread in threads:

        try:
            action = thread.correlation_action

            # Handle SKIP - near-duplicate, don't process
            if action == 'SKIP':
                recommendation = get_recommendation(thread)
                reason = recommendation.get('reason', 'duplicate') if recommendation else 'duplicate'
                print(f"Skipping {thread.id}: {reason}")
                results.append({
                    "action": "skipped",
                    "thread_id": thread.id,
                    "reason": reason,
                })
                continue

            # Handle APPEND - add to existing entry
            if action == 'APPEND':
                recommendation = get_recommendation(thread)
                if recommendation and recommendation.get('entry_id'):
                    result = handle_append(thread, recommendation['entry_id'], commit)
                    results.append(result)
                    print(f"Appending {thread.id} to {recommendation['entry_id']}")
                    continue

            # Handle QUEUE - task on existing chord
            if action == 'QUEUE':
                recommendation = get_recommendation(thread)
                if recommendation and recommendation.get('chord_repo'):
                    result = handle_queue_task(thread, recommendation['chord_repo'], commit)
                    results.append(result)
                    print(f"Queuing task on {recommendation['chord_repo']} for {thread.id}")
                    continue

            # Default: CREATE new entry
            artifact = extract_knowledge(thread)

            if commit:
                result = commit_knowledge(artifact)
                result["artifact"] = artifact.to_markdown()[:200] + "..."
            else:
                result = {
                    "artifact_id": artifact.id,
                    "path": artifact.get_path(),
                    "action": "would_create",
                    "needs_chord": artifact.needs_chord
                }

            results.append(result)

            if artifact.needs_chord:
                print(f"Processed knowledge (needs chord): {artifact.id}")
            else:
                print(f"Processed knowledge: {artifact.id}")

        except Exception as e:
            print(f"Error processing {thread.id}: {e}", file=sys.stderr)
            results.append({"error": str(e), "thread_id": thread.id})

    return results


def main():
    """CLI entry point for knowledge module."""
    parser = argparse.ArgumentParser(description="LEGATO Knowledge Module")
    parser.add_argument(
        "--input",
        required=True,
        help="Input routing.json file"
    )
    parser.add_argument(
        "--commit",
        action="store_true",
        help="Actually commit to Library"
    )
    parser.add_argument(
        "--output",
        help="Output results file"
    )
    args = parser.parse_args()

    results = process_routing(args.input, args.commit)

    if args.output:
        with open(args.output, "w") as f:
            json.dump(results, f, indent=2)

    committed = len([r for r in results if r.get("action") in ("created", "updated")])
    errors = len([r for r in results if "error" in r])

    print(f"Processed {len(results)} items: {committed} committed, {errors} errors")


if __name__ == "__main__":
    main()
