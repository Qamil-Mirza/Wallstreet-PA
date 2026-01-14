"""
Email rendering and sending via SMTP.

Builds HTML emails and sends them using configured SMTP server.
"""

import logging
import smtplib
from datetime import date
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from typing import Optional

from .classifier import ArticleCategory, bucket_articles
from .config import Config
from .news_client import ArticleMeta
from .selection import get_article_category_label


logger = logging.getLogger(__name__)


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
        .article {{
            margin-bottom: 30px;
            padding-bottom: 25px;
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
        
        {articles_html}
        
        <div class="footer">
            Curated by The Journey • Powered by Ollama
        </div>
    </div>
</body>
</html>
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
    """Format narrative summary as HTML paragraph."""
    # Clean up the text - join multiple lines into flowing text
    text = " ".join(line.strip() for line in summary.strip().split("\n") if line.strip())
    
    # Remove any stray bullet characters that might have slipped through
    for prefix in ["•", "-", "*", "·"]:
        if text.startswith(prefix):
            text = text[1:].strip()
    
    return f"<p>{text}</p>"


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
    Build HTML email content.
    
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
        articles_html=articles_html,
    )


class EmailError(Exception):
    """Raised when email sending fails."""
    pass


def send_email(config: Config, subject: str, html_body: str) -> None:
    """
    Send an HTML email via SMTP.
    
    Args:
        config: Application configuration with SMTP settings.
        subject: Email subject line.
        html_body: HTML content for the email body.
    
    Raises:
        EmailError: If sending fails.
    """
    msg = MIMEMultipart("alternative")
    msg["Subject"] = subject
    msg["From"] = config.smtp_user
    msg["To"] = config.recipient_email
    
    # Attach HTML content
    html_part = MIMEText(html_body, "html")
    msg.attach(html_part)
    
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
