"""
Email rendering and sending via SMTP.

Builds HTML emails and sends them using configured SMTP server.
"""

import logging
import smtplib
from datetime import date
from email.encoders import encode_base64
from email.mime.audio import MIMEAudio
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from pathlib import Path
from typing import Optional

from .classifier import ArticleCategory, bucket_articles
from .config import Config
from .news_client import ArticleMeta
from .selection import get_article_category_label


logger = logging.getLogger(__name__)


# Section display configuration with emoji icons
SECTION_CONFIG = {
    "World News": {"emoji": "🌍", "color": "#6366f1"},
    "US Tech": {"emoji": "🇺🇸💻", "color": "#0ea5e9"},
    "US Industry": {"emoji": "🇺🇸🏭", "color": "#14b8a6"},
    "Malaysia Tech": {"emoji": "🇲🇾💻", "color": "#f59e0b"},
    "Malaysia Industry": {"emoji": "🇲🇾🏭", "color": "#ef4444"},
    "RSS Feeds": {"emoji": "📡", "color": "#8b5cf6"},
}


EMAIL_TEMPLATE = """<!DOCTYPE html>
<html>
<head>
    <meta charset="utf-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>The Daily Briefing</title>
    <style>
        body {{
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, 'Helvetica Neue', Arial, sans-serif;
            line-height: 1.6;
            color: #1a1a1a;
            max-width: 600px;
            margin: 0 auto;
            padding: 20px;
            background-color: #f5f5f5;
        }}
        .container {{
            background-color: #ffffff;
            border-radius: 8px;
            padding: 30px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }}
        .header {{
            border-bottom: 3px solid #0066cc;
            padding-bottom: 15px;
            margin-bottom: 25px;
        }}
        .header h1 {{
            margin: 0;
            color: #0066cc;
            font-size: 24px;
            font-weight: 700;
        }}
        .header .date {{
            color: #666;
            font-size: 14px;
            margin-top: 5px;
        }}
        .section {{
            margin-bottom: 30px;
        }}
        .section-header {{
            display: flex;
            align-items: center;
            padding: 12px 15px;
            margin-bottom: 20px;
            border-radius: 6px;
            background-color: #f8f9fa;
            border-left: 4px solid;
        }}
        .section-header h2 {{
            margin: 0;
            font-size: 16px;
            font-weight: 600;
        }}
        .section-header .emoji {{
            margin-right: 10px;
            font-size: 18px;
        }}
        .section-empty {{
            color: #888;
            font-style: italic;
            font-size: 14px;
            padding: 10px 0;
        }}
        .article {{
            margin-bottom: 25px;
            padding-bottom: 20px;
            border-bottom: 1px solid #eee;
        }}
        .article:last-child {{
            border-bottom: none;
            margin-bottom: 0;
            padding-bottom: 0;
        }}
        .category {{
            display: inline-block;
            padding: 4px 10px;
            border-radius: 4px;
            font-size: 11px;
            font-weight: 600;
            text-transform: uppercase;
            letter-spacing: 0.5px;
            margin-bottom: 10px;
        }}
        .category-macro {{
            background-color: #e8f4fd;
            color: #0066cc;
        }}
        .category-deal {{
            background-color: #f0fdf4;
            color: #15803d;
        }}
        .category-feature {{
            background-color: #fef3c7;
            color: #b45309;
        }}
        .headline {{
            font-size: 18px;
            font-weight: 600;
            margin: 0 0 8px 0;
        }}
        .headline a {{
            color: #1a1a1a;
            text-decoration: none;
        }}
        .headline a:hover {{
            color: #0066cc;
        }}
        .meta {{
            font-size: 12px;
            color: #888;
            margin-bottom: 12px;
        }}
        .summary {{
            font-size: 14px;
            color: #444;
        }}
        .summary ul {{
            margin: 0;
            padding-left: 20px;
        }}
        .summary li {{
            margin-bottom: 6px;
        }}
        .footer {{
            margin-top: 30px;
            padding-top: 20px;
            border-top: 1px solid #eee;
            font-size: 12px;
            color: #888;
            text-align: center;
        }}
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>📊 The Daily Briefing</h1>
            <div class="date">The Journey — {formatted_date}</div>
        </div>
        
        {sections_html}
        
        <div class="footer">
            Curated by The Journey • Powered by Ollama
        </div>
    </div>
</body>
</html>
"""

SECTION_TEMPLATE = """
<div class="section">
    <div class="section-header" style="border-color: {color};">
        <span class="emoji">{emoji}</span>
        <h2>{section_name}</h2>
    </div>
    {articles_html}
</div>
"""

ARTICLE_TEMPLATE = """
<div class="article">
    <span class="category category-{category_class}">{category_label}</span>
    <h2 class="headline"><a href="{url}">{title}</a></h2>
    <div class="meta">{source} • {published_time}</div>
    <div class="summary">{summary_html}</div>
</div>
"""


def _format_summary_html(summary: str) -> str:
    """Format analyst-style summary as HTML with proper structure."""
    # Split on double newlines to separate main content from "So what?" section
    parts = summary.strip().split("\n\n")
    
    html_parts = []
    for part in parts:
        # Clean up each part
        text = " ".join(line.strip() for line in part.strip().split("\n") if line.strip())
        
        # Remove any stray bullet characters
        for prefix in ["•", "-", "*", "·"]:
            if text.startswith(prefix):
                text = text[1:].strip()
        
        if text:
            # Style "So what?" differently for emphasis
            if text.lower().startswith("so what?"):
                html_parts.append(f'<p style="margin-top: 8px; color: #374151;"><strong>{text[:8]}</strong>{text[8:]}</p>')
            else:
                html_parts.append(f"<p>{text}</p>")
    
    return "".join(html_parts) if html_parts else "<p></p>"


def _get_category_class(
    article: ArticleMeta,
    buckets: dict[ArticleCategory, list[ArticleMeta]]
) -> str:
    """Get CSS class for article category."""
    for category, articles in buckets.items():
        if any(a.id == article.id for a in articles):
            return category
    return "feature"


def build_email_html(
    email_date: date,
    articles: list[ArticleMeta],
    summaries: dict[str, str],
    buckets: Optional[dict[ArticleCategory, list[ArticleMeta]]] = None
) -> str:
    """
    Build HTML email content (legacy flat list format).
    
    Args:
        email_date: Date for the email header.
        articles: List of selected articles.
        summaries: Dictionary mapping article IDs to summaries.
        buckets: Optional category buckets for labeling.
    
    Returns:
        Complete HTML email string.
    """
    if buckets is None:
        # Create buckets from articles for categorization
        buckets = bucket_articles(articles)
    
    formatted_date = email_date.strftime("%A, %B %d, %Y")
    
    articles_html_parts = []
    
    for article in articles:
        summary = summaries.get(article.id, "")
        summary_html = _format_summary_html(summary)
        
        category_class = _get_category_class(article, buckets)
        category_label = get_article_category_label(article, buckets)
        
        source = article.source or "Unknown"
        published_time = article.published_at.strftime("%I:%M %p")
        
        article_html = ARTICLE_TEMPLATE.format(
            category_class=category_class,
            category_label=category_label,
            url=article.url,
            title=article.title,
            source=source,
            published_time=published_time,
            summary_html=summary_html,
        )
        articles_html_parts.append(article_html)
    
    articles_html = "\n".join(articles_html_parts)
    
    return EMAIL_TEMPLATE.format(
        formatted_date=formatted_date,
        sections_html=articles_html,
    )


def build_sectioned_email_html(
    email_date: date,
    sections: dict[str, list[ArticleMeta]],
    summaries: dict[str, str],
) -> str:
    """
    Build HTML email content organized by sections.
    
    Only includes sections that are present in the sections dict.
    Sections with no articles are skipped entirely.
    
    Args:
        email_date: Date for the email header.
        sections: Dictionary mapping section names to lists of articles.
                  Only sections in this dict will be rendered.
        summaries: Dictionary mapping article IDs to summaries.
    
    Returns:
        Complete HTML email string with sectioned layout.
    """
    formatted_date = email_date.strftime("%A, %B %d, %Y")
    
    # Define section order (for consistent ordering when multiple sections enabled)
    # Additional sections (like custom RSS feeds) will be rendered at the end
    section_order = [
        "World News",
        "US Tech",
        "US Industry",
        "Malaysia Tech",
        "Malaysia Industry",
        "RSS Feeds",
    ]
    
    # Add any sections not in the predefined order (custom RSS sections)
    for section_name in sections.keys():
        if section_name not in section_order:
            section_order.append(section_name)
    
    sections_html_parts = []
    
    for section_name in section_order:
        # Only render sections that exist in the dict AND have articles
        if section_name not in sections:
            continue
        
        articles = sections[section_name]
        if not articles:
            continue  # Skip empty sections entirely
        
        config = SECTION_CONFIG.get(section_name, {"emoji": "📰", "color": "#666"})
        
        # Build articles HTML for this section
        articles_html_parts = []
        for article in articles:
            summary = summaries.get(article.id, "")
            summary_html = _format_summary_html(summary) if summary else "<p><em>Summary not available</em></p>"
            
            source = article.source or "Unknown"
            published_time = article.published_at.strftime("%I:%M %p")
            
            # Use a simplified article format without category tags for sectioned view
            article_html = f"""
<div class="article">
    <h2 class="headline"><a href="{article.url}">{article.title}</a></h2>
    <div class="meta">{source} • {published_time}</div>
    <div class="summary">{summary_html}</div>
</div>
"""
            articles_html_parts.append(article_html)
        
        articles_html = "\n".join(articles_html_parts)
        
        section_html = SECTION_TEMPLATE.format(
            color=config["color"],
            emoji=config["emoji"],
            section_name=section_name,
            articles_html=articles_html,
        )
        sections_html_parts.append(section_html)
    
    sections_html = "\n".join(sections_html_parts)
    
    return EMAIL_TEMPLATE.format(
        formatted_date=formatted_date,
        sections_html=sections_html,
    )


class EmailError(Exception):
    """Raised when email sending fails."""
    pass


def send_email(
    config: Config,
    subject: str,
    html_body: str,
    attachment_path: Optional[Path] = None,
) -> None:
    """
    Send an HTML email via SMTP with optional MP3 attachment.
    
    Args:
        config: Application configuration with SMTP settings.
        subject: Email subject line.
        html_body: HTML content for the email body.
        attachment_path: Optional path to an MP3 file to attach.
    
    Raises:
        EmailError: If sending fails.
    """
    # Use mixed for attachments, alternative for HTML-only
    msg = MIMEMultipart("mixed" if attachment_path else "alternative")
    msg["Subject"] = subject
    msg["From"] = config.smtp_user
    msg["To"] = config.recipient_email
    
    # Attach HTML content
    html_part = MIMEText(html_body, "html")
    msg.attach(html_part)
    
    # Attach audio file if provided (supports MP3 and WAV)
    if attachment_path and attachment_path.exists():
        try:
            # Determine audio subtype based on file extension
            suffix = attachment_path.suffix.lower()
            if suffix == ".mp3":
                subtype = "mpeg"
            elif suffix == ".wav":
                subtype = "wav"
            else:
                subtype = "mpeg"  # Default to MP3
            
            with open(attachment_path, "rb") as audio_file:
                audio_part = MIMEAudio(audio_file.read(), _subtype=subtype)
            audio_part.add_header(
                "Content-Disposition",
                "attachment",
                filename=attachment_path.name,
            )
            msg.attach(audio_part)
            logger.info(f"Attached audio file: {attachment_path.name}")
        except Exception as e:
            logger.warning(f"Failed to attach audio file: {e}")
    
    try:
        with smtplib.SMTP(config.smtp_host, config.smtp_port) as server:
            server.starttls()
            server.login(config.smtp_user, config.smtp_password)
            server.sendmail(
                config.smtp_user,
                config.recipient_email,
                msg.as_string()
            )
        logger.info(f"Email sent to {config.recipient_email}")
        
    except smtplib.SMTPAuthenticationError as e:
        raise EmailError(f"SMTP authentication failed: {e}")
    except smtplib.SMTPException as e:
        raise EmailError(f"Failed to send email: {e}")
    except Exception as e:
        raise EmailError(f"Unexpected error sending email: {e}")
