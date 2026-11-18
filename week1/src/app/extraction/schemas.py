from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

RoleLevel = Literal["Junior", "Mid", "Senior", "Staff"]


class JobPosting(BaseModel):
    model_config = ConfigDict(extra="forbid")

    title: str = Field(description="Job title as written, without company or location")
    company: str = Field(description="Hiring company name; not the recruiting agency")
    tech_stack: list[str] = Field(
        description="Technologies explicitly mentioned (languages, frameworks, databases, cloud). Canonical names, e.g. 'PostgreSQL' not 'pg'"
    )
    min_salary: int | None = Field(
        description="Minimum base salary, annualized, in USD. Convert '150k' to 150000, hourly x 2080. null if not stated or not in USD"
    )
    experience_years: int | None = Field(
        description="Minimum years of experience required. null if not stated; never guess"
    )
    is_remote: bool = Field(description="True only if fully remote is allowed; hybrid or onsite is False")
    role_level: RoleLevel = Field(description="Infer from title and requirements if not explicit")


class Extraction(BaseModel):
    """Top-level output: a text can contain zero, one, or several postings."""

    model_config = ConfigDict(extra="forbid")

    postings: list[JobPosting] = Field(description="Empty if the text is not a job posting")
