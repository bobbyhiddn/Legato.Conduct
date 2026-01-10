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
        related=[m.get("signal_id") for m in thread.correlation_matches if m.get("score", 0) > 0.5],
        needs_chord=thread.needs_chord,
        chord_name=thread.chord_name,
        chord_scope=thread.chord_scope.value if thread.chord_scope else None,
    )


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


def process_routing(routing_file: str, commit: bool = False) -> list[dict]:
    """
    Process classified threads from a routing file.

    Respects correlation_action from classification:
    - CREATE: Create new note in Library
    - APPEND: Append to existing note
    - QUEUE: Queue task on existing chord (instead of new chord)
    - SKIP: Skip processing (near-duplicate)

    Args:
        routing_file: Path to routing.json
        commit: Whether to commit to Library

    Returns:
        List of results
    """
    with open(routing_file) as f:
        routing = json.load(f)

    results = []

    for item in routing:
        thread = ClassifiedThread.from_dict(item)

        try:
            action = thread.correlation_action

            # Handle SKIP - near-duplicate, don't process
            if action == 'SKIP':
                recommendation = None
                for match in thread.correlation_matches:
                    if 'recommendation' in match:
                        recommendation = match['recommendation']
                        break

                print(f"Skipping {thread.id}: {recommendation.get('reason', 'duplicate') if recommendation else 'duplicate'}")
                results.append({
                    "action": "skipped",
                    "thread_id": thread.id,
                    "reason": recommendation.get('reason') if recommendation else "duplicate",
                })
                continue

            # Handle APPEND - add to existing entry
            if action == 'APPEND':
                recommendation = None
                for match in thread.correlation_matches:
                    if 'recommendation' in match:
                        recommendation = match['recommendation']
                        break

                if recommendation and recommendation.get('entry_id'):
                    result = handle_append(thread, recommendation['entry_id'], commit)
                    results.append(result)
                    print(f"Appending {thread.id} to {recommendation['entry_id']}")
                    continue

            # Handle QUEUE - task on existing chord
            if action == 'QUEUE':
                recommendation = None
                for match in thread.correlation_matches:
                    if 'recommendation' in match:
                        recommendation = match['recommendation']
                        break

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
