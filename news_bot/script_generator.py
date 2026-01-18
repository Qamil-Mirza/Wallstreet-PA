"""
Script generation for TTS audio using Ollama (local LLM).

Transforms article summaries into an engaging radio host script
with a Wall Street finance personality.
"""

import logging
import re
from dataclasses import dataclass
from datetime import date, datetime
from pathlib import Path
from typing import Optional

import requests


logger = logging.getLogger(__name__)


# =============================================================================
# Script Generation Prompts
# =============================================================================

SCRIPT_SYSTEM_PROMPT = """You are "The Street Beat" - a fast-paced, no-nonsense Wall Street radio host known for breaking down financial news with sharp wit and insider credibility. Think CNBC meets Bloomberg Radio, with a touch of irreverent charm.

Your personality:
- Confident and authoritative, but never arrogant
- Use colorful market metaphors ("the bulls are stampeding", "bears are sharpening their claws")
- Pepper in classic trading floor expressions ("let's cut through the noise", "follow the smart money")
- Occasionally address listeners directly ("Here's what you need to know before the bell")
- Keep energy high but professional - this isn't a podcast, it's premium market intelligence

Voice characteristics for TTS:
- Use natural speech patterns with varied sentence lengths
- Include brief pauses (marked with "...") for dramatic effect
- Avoid complex nested sentences that sound unnatural when spoken
- Write numbers as words for better TTS pronunciation (e.g., "fifty million" not "50M")"""


SCRIPT_TEMPLATE = """<role>
{system_prompt}
</role>

<instructions>
Transform the following market summaries into a cohesive {duration}-minute radio broadcast script. 

Structure your script as follows:
1. **Opening Hook** (10-15 seconds): A punchy one-liner that grabs attention and sets the market mood
2. **Headlines Rundown** (15-20 seconds): Quick-fire overview of what's coming ("We've got three movers shaking up your portfolio today...")
3. **Main Stories** ({story_count} stories, ~{time_per_story} seconds each): For each story:
   - Transition phrase connecting to previous story
   - The key news in plain English
   - Why it matters to listeners' portfolios
   - One sharp takeaway or forward-looking comment
4. **Closing Kicker** (10-15 seconds): Memorable sign-off with market wisdom or call to action

IMPORTANT FORMATTING RULES:
- Write ONLY the spoken words - no stage directions, no [brackets], no (parentheses)
- Use "..." for natural pauses
- Write all numbers and percentages as spoken words
- Keep sentences punchy - average 10-15 words
- Total script should be approximately {word_count} words for {duration}-minute read
</instructions>

<date>
{broadcast_date}
</date>

<summaries>
{summaries_text}
</summaries>

<output>
Write the complete radio script now. Start directly with the opening hook - no preamble.
</output>"""


@dataclass
class ScriptConfig:
    """Configuration for script generation."""
    duration_minutes: float = 2.0  # Target audio duration
    words_per_minute: int = 150  # Average speaking rate
    
    @property
    def target_word_count(self) -> int:
        """Calculate target word count based on duration."""
        return int(self.duration_minutes * self.words_per_minute)


class ScriptGeneratorError(Exception):
    """Raised when script generation fails."""
    pass


def _clean_script(text: str) -> str:
    """
    Clean up generated script for TTS consumption.
    
    Removes common LLM artifacts and ensures clean spoken text.
    """
    # Remove any markdown formatting
    text = re.sub(r'\*\*([^*]+)\*\*', r'\1', text)  # Bold
    text = re.sub(r'\*([^*]+)\*', r'\1', text)  # Italic
    text = re.sub(r'#{1,6}\s*', '', text)  # Headers
    
    # Remove stage directions in brackets or parentheses
    text = re.sub(r'\[([^\]]+)\]', '', text)
    text = re.sub(r'\((?:pause|beat|laughs?|sighs?)[^)]*\)', '', text, flags=re.IGNORECASE)
    
    # Clean up excessive whitespace
    text = re.sub(r'\n{3,}', '\n\n', text)
    text = re.sub(r' {2,}', ' ', text)
    
    # Normalize ellipses for pauses
    text = re.sub(r'\.{4,}', '...', text)
    text = re.sub(r'…', '...', text)
    
    # Remove common preambles
    preambles = [
        "here is the script",
        "here's the script",
        "here is your script",
        "here's your script",
        "radio script:",
        "script:",
    ]
    
    lines = text.strip().split('\n')
    cleaned_lines = []
    skip_first = False
    
    for i, line in enumerate(lines):
        line_lower = line.lower().strip()
        if i == 0 and any(line_lower.startswith(p) for p in preambles):
            skip_first = True
            continue
        if skip_first and not line.strip():
            continue
        skip_first = False
        cleaned_lines.append(line)
    
    return '\n'.join(cleaned_lines).strip()


def _format_summaries_for_script(summaries: dict[str, str]) -> str:
    """
    Format article summaries into a text block for the LLM prompt.
    
    Args:
        summaries: Dictionary mapping article IDs to summary text.
    
    Returns:
        Formatted text block with numbered summaries.
    """
    lines = []
    for i, (article_id, summary) in enumerate(summaries.items(), 1):
        # Clean up summary for inclusion
        clean_summary = summary.strip()
        lines.append(f"Story {i}:\n{clean_summary}")
    
    return "\n\n".join(lines)


def generate_script(
    summaries: dict[str, str],
    model: str,
    base_url: str,
    script_config: Optional[ScriptConfig] = None,
    broadcast_date: Optional[date] = None,
    timeout: int = 180,
) -> str:
    """
    Generate a radio host script from article summaries.
    
    Uses Ollama to transform dry financial summaries into an engaging
    spoken broadcast script suitable for TTS.
    
    Args:
        summaries: Dictionary mapping article IDs to summary text.
        model: Ollama model name (e.g., "llama3").
        base_url: Ollama API base URL.
        script_config: Configuration for script generation.
        broadcast_date: Date for the broadcast (defaults to today).
        timeout: Request timeout in seconds.
    
    Returns:
        Generated script text ready for TTS.
    
    Raises:
        ScriptGeneratorError: If generation fails.
    """
    if not summaries:
        raise ScriptGeneratorError("No summaries provided for script generation")
    
    config = script_config or ScriptConfig()
    broadcast_date = broadcast_date or date.today()
    
    # Calculate timing
    story_count = len(summaries)
    # Reserve ~30 seconds for intro/outro
    content_duration = config.duration_minutes - 0.5
    time_per_story = int((content_duration * 60) / story_count) if story_count > 0 else 30
    
    # Format summaries
    summaries_text = _format_summaries_for_script(summaries)
    
    # Build prompt
    prompt = SCRIPT_TEMPLATE.format(
        system_prompt=SCRIPT_SYSTEM_PROMPT,
        duration=config.duration_minutes,
        story_count=story_count,
        time_per_story=time_per_story,
        word_count=config.target_word_count,
        broadcast_date=broadcast_date.strftime("%B %d, %Y"),
        summaries_text=summaries_text,
    )
    
    url = f"{base_url.rstrip('/')}/api/generate"
    
    payload = {
        "model": model,
        "prompt": prompt,
        "stream": False,
        "options": {
            "temperature": 0.7,  # Higher for more creative output
            "num_predict": 1500,  # Allow for longer scripts
        }
    }
    
    logger.info(f"Generating {config.duration_minutes}-minute script for {story_count} stories...")
    start_time = datetime.now()
    
    try:
        response = requests.post(url, json=payload, timeout=timeout)
        response.raise_for_status()
    except requests.Timeout:
        raise ScriptGeneratorError(f"Script generation timed out after {timeout}s")
    except requests.RequestException as e:
        raise ScriptGeneratorError(f"Failed to connect to Ollama: {e}")
    
    duration_ms = (datetime.now() - start_time).total_seconds() * 1000
    
    data = response.json()
    raw_script = data.get("response", "")
    
    if not raw_script:
        raise ScriptGeneratorError("Ollama returned empty response for script generation")
    
    script = _clean_script(raw_script)
    
    # Log generation stats
    word_count = len(script.split())
    logger.info(
        f"Script generated: {word_count} words "
        f"(target: {config.target_word_count}) in {duration_ms:.0f}ms"
    )
    
    return script


def generate_broadcast_script(
    summaries: dict[str, str],
    ollama_model: str,
    ollama_base_url: str,
    duration_minutes: float = 2.0,
) -> str:
    """
    High-level function to generate a broadcast script.
    
    Convenience wrapper around generate_script with sensible defaults.
    
    Args:
        summaries: Dictionary mapping article IDs to summary text.
        ollama_model: Ollama model name.
        ollama_base_url: Ollama API base URL.
        duration_minutes: Target audio duration in minutes.
    
    Returns:
        Generated script text.
    """
    config = ScriptConfig(duration_minutes=duration_minutes)
    
    return generate_script(
        summaries=summaries,
        model=ollama_model,
        base_url=ollama_base_url,
        script_config=config,
    )
