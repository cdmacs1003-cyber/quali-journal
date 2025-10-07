from qj_paths import rel as qj_rel
"""
Config schema definition for QualiJournal.

This module defines a unified configuration schema using Pydantic models.  It
provides strongly?몋yped configuration objects with sensible default values
across the various subsystems (engine_core, orchestrator and web admin).

By parsing ``config.json`` through these models, the application obtains a
single, consistent configuration dictionary with all expected keys.  This
helps eliminate discrepancies between default values defined in different
modules and enables input validation: type errors or unrecognised fields
raise exceptions early, preventing misconfiguration from propagating deep
into the system.

The root model is :class:`ConfigSchema`, which aggregates the nested
sub?멵onfigurations for paths, features, community and external RSS.  See
individual class docstrings for details.
"""

from __future__ import annotations

import json
from typing import Dict, List, Optional

from pydantic import BaseModel, Field, ValidationError


class PathsConfig(BaseModel):
    """Filesystem and resource locations."""

    archive: str = Field(
        'archive', description="Base directory for generated artefacts such as reports and selections."
    )
    reports: str = Field(
        'archive/reports', description="Directory where generated reports (HTML, MD, JSON) are saved."
    )
    community_selection_file: str = Field(
        'archive/selected_community.json',
        description="Default path for the community selection JSON file."
    )
    community_sources_candidates: List[str] = Field(
        default_factory=lambda: ['feeds/community_sources.json', 'community_sources.json'],
        description="Candidate file paths for community source definitions."
    )
    keywords_txt: str = Field(
        '而ㅻ??덊떚_?ㅼ썙??txt', description="Path to the plain text file containing community keywords."
    )
    selection_file: str = Field(
        'selected_articles.json', description="Default path for the official selection JSON file."
    )
    smt_sources_file: str = Field(
        'smt_sources.json', description="Path to the SMT (Surface Mount Technology) sources configuration file."
    )
    supply_sources_file: str = Field(
        'supply_sources.json', description="Path to the supply chain sources configuration file."
    )
    backup: str = Field(
        'archive/backup', description="Directory for backup copies of published materials."
    )
    log: str = Field(
        'logs/news.log', description="Path to the rotating log file for the news collector."
    )


class QualityConfig(BaseModel):
    """Quality criteria for article selection."""

    min_chars: int = Field(600, description="Minimum number of characters required for an article to qualify.")
    min_sentences: int = Field(5, description="Minimum number of sentences required for an article to qualify.")
    min_links: int = Field(2, description="Minimum number of outbound links required for an article to qualify.")
    strict: bool = Field(True, description="Whether to enforce all quality criteria strictly.")
    max_total: int = Field(6, description="Maximum number of articles to select in total.")
    max_per_section: int = Field(
        3, description="Maximum number of articles to select per section when grouping by category."
    )


class FactcheckConfig(BaseModel):
    """Configuration for fact?멵hecking (FC) scoring."""

    enabled: bool = Field(True, description="Whether fact?멵hecking is enabled during article scoring.")
    min_score: int = Field(52, description="Minimum score required to pass fact?멵hecking.")
    min_evidence: int = Field(2, description="Minimum number of evidences required for fact?멵hecking.")
    strict: bool = Field(True, description="Whether to require both minimum score and evidence to pass.")


class DynamicThresholdConfig(BaseModel):
    """Adaptive scoring thresholds for community items."""

    enabled: bool = Field(False, description="Whether dynamic thresholds are used for approval scoring.")
    floor: float = Field(0.0, description="Lower bound on the dynamic threshold.")
    ceil: float = Field(5.0, description="Upper bound on the dynamic threshold.")
    percentile: int = Field(
        50,
        description=(
            "Percentile of the score distribution to consider when computing the dynamic threshold."
        ),
    )
    sigma_k: float = Field(0.5, description="Multiplier applied to the standard deviation when computing the threshold.")
    window_days: int = Field(
        7,
        description="Number of days worth of data to use when computing the dynamic threshold window."
    )


class CommunityFilters(BaseModel):
    """Optional filters applied to community article selection."""

    kw_min_tokens: Optional[int] = Field(
        None,
        description="Minimum number of tokens required in a keyword for a community article to qualify."
    )
    kw_regex: Optional[List[str]] = Field(
        None,
        description="List of regular expressions; an article's title must match at least one to qualify."
    )
    allow_domains: Optional[List[str]] = Field(
        None, description="Whitelist of domains allowed for community posts."
    )
    block_title_patterns: Optional[List[str]] = Field(
        None, description="List of regex patterns to block titles containing promotional or irrelevant content."
    )
    min_title_len: Optional[int] = Field(
        None, description="Minimum length of a community article title to qualify."
    )
    require_keyword: Optional[bool] = Field(
        None, description="Whether a keyword must be present in the title or body for the article to qualify."
    )
    synonyms_source: Optional[str] = Field(
        None,
        description="Path to a JSON file containing keyword synonyms used for matching community posts."
    )
    use_keyword_synonyms: Optional[bool] = Field(
        None, description="Whether to expand keywords to include synonyms during matching."
    )


class CommunityConfig(BaseModel):
    """Parameters controlling community content ingestion and scoring."""

    enabled: bool = Field(True, description="Whether community ingestion is enabled.")
    fresh_hours: int = Field(336, description="Number of hours during which a community post is considered fresh.")
    min_upvotes: int = Field(0, description="Minimum number of upvotes required for a community post to qualify.")
    min_comments: int = Field(0, description="Minimum number of comments required for a community post to qualify.")
    score_threshold: float = Field(0.0, description="Minimum total score required for a community post to qualify.")
    max_total: int = Field(300, description="Maximum number of community posts to select.")
    max_per_section: Optional[int] = Field(
        6, description="Maximum number of community posts per section when grouping by category."
    )
    reddit_pages: int = Field(5, description="Number of pages to fetch when crawling Reddit for community posts.")
    score_weights: Dict[str, int] = Field(
        default_factory=lambda: {'keyword': 3, 'upvotes': 5, 'views': 2},
        description="Weights applied to keyword hits, upvotes and views when computing the total score."
    )
    norms: Dict[str, int] = Field(
        default_factory=lambda: {'kw_base': 2, 'upvotes_max': 200, 'views_max': 100000},
        description="Normalization parameters for score components."
    )
    filters: CommunityFilters = Field(
        default_factory=CommunityFilters,
        description="Filter definitions for community content."
    )
    dynamic_threshold: Optional[DynamicThresholdConfig] = Field(
        None, description="Dynamic threshold configuration for adaptive scoring."
    )
    html_title: Optional[str] = Field(
        '?꾨━ 而ㅻ??덊떚', description="Title used when rendering community selection pages."
    )
    exclude_ai: Optional[bool] = Field(
        True, description="Whether to exclude AI?몉elated topics from community ingestion."
    )
    selection_file: Optional[str] = Field(
        'archive/selected_community.json',
        description="Override path for the community selection file if different from the default."
    )


class ExternalRSSConfig(BaseModel):
    """Controls ingestion of external RSS feeds in keyword workflows."""

    enabled: bool = Field(False, description="Whether to include external RSS sources when gathering keywords.")
    max_total: int = Field(50, description="Maximum number of articles to import from external RSS.")
    gate_required: Optional[int] = Field(
        None,
        description=(
            "Minimum number of editor approvals required before external RSS sources are used."
        ),
    )


class FeaturesConfig(BaseModel):
    """Miscellaneous feature toggles and metadata."""

    require_editor_approval: bool = Field(
        True, description="Whether editor approval is required before articles can be published."
    )
    trusted_domains: List[str] = Field(
        default_factory=list, description="List of domains considered trusted when whitelisting sources."
    )
    output_formats: List[str] = Field(
        default_factory=lambda: ['html', 'md', 'json'],
        description="List of output formats to generate during publication."
    )
    section_order: List[str] = Field(
        default_factory=lambda: ['?쒖? ?댁뒪', '湲濡쒕쾶 ?꾩옄?앹궛', 'AI ?댁뒪', '而ㅻ??덊떚 ?뚯튂', '?꾨━?댁뒪'],
        description="Order in which sections appear in the published news output."
    )
    quality: Optional[QualityConfig] = Field(
        default_factory=QualityConfig, description="Quality criteria for article selection."
    )
    factcheck: Optional[FactcheckConfig] = Field(
        default_factory=FactcheckConfig, description="Fact?멵hecking parameters."
    )
    translate_model: Optional[str] = Field(
        'gpt-4o', description="Name of the OpenAI model to use for translations and summaries."
    )
    landing_rules: Optional[Dict[str, Dict[str, List[str]]]] = Field(
        None,
        description=(
            "Domain?몊pecific rules controlling which article paths are preferred or denied "
            "when scraping official sources."
        ),
    )


class ConfigSchema(BaseModel):
    """Root configuration model encompassing all sub?멵onfigs."""

    paths: PathsConfig = Field(default_factory=PathsConfig)
    features: FeaturesConfig = Field(default_factory=FeaturesConfig)
    community: CommunityConfig = Field(default_factory=CommunityConfig)
    external_rss: ExternalRSSConfig = Field(default_factory=ExternalRSSConfig)


def load_config(path: str = 'config.json') -> ConfigSchema:
    """
    Load configuration from a JSON file, applying defaults and validating types.

    Args:
        path: Location of the JSON configuration file.  If the file does not
            exist or contains invalid JSON, an empty configuration is assumed.

    Returns:
        ConfigSchema: Parsed configuration model with defaults filled in.

    Raises:
        ValidationError: If the provided configuration contains values of
            incorrect type or unknown fields.
    """
    try:
        with open(path, 'r', encoding='utf-8') as f:
            data = json.load(f)
            if not isinstance(data, dict):
                data = {}
    except FileNotFoundError:
        data = {}
    except Exception:
        # malformed JSON or other error
        data = {}
    # Pydantic will raise a ValidationError if keys have wrong types
    cfg_model = ConfigSchema(**data)
    return cfg_model


# Export a plain dict version of the default configuration for legacy modules
DEFAULT_CFG: Dict = ConfigSchema().dict()


def generate_json_schema() -> Dict:
    """
    Generate a JSON Schema (draft v7) describing the structure of the configuration.

    Returns:
        dict: JSON schema that can be used for validation in tools such as VSCode or AJV.
    """
    # Pydantic's .schema() produces a JSON Schema representation
    return ConfigSchema.schema()
