"""Domain models for the service catalog (structured knowledge).

The service catalog is the structured half of the platform's knowledge. Where a
:class:`~meridian.domain.models.KnowledgeChunk` is unstructured prose, a
:class:`ServiceRecord` is a row with typed fields - owner, tier, domain,
dependency count - that answers precise questions completely rather than
approximately.
"""

from pydantic import BaseModel, Field


class ServiceRecord(BaseModel):
    """A single service in the catalog.

    ``visibility`` mirrors the access-control model used for knowledge chunks:
    the caller's ACL groups are intersected with this list at query time, so a
    catalog query never returns a service the caller may not see.
    """

    service_id: str = Field(..., description="Stable unique identifier.")
    name: str = Field(..., description="Human-readable service name.")
    team: str = Field(..., description="Owning team.")
    domain: str = Field(..., description="Business domain, e.g. 'payments'.")
    tier: str = Field(..., description="Criticality tier, e.g. 'tier1'.")
    lifecycle: str = Field(default="active", description="Lifecycle state.")
    has_owner: bool = Field(default=True, description="Whether an owner is assigned.")
    dependencies: int = Field(default=0, description="Number of upstream dependencies.")
    description: str = Field(default="", description="Short description.")
    visibility: list[str] = Field(default_factory=list, description="ACL groups that may see this service.")


class StructuredQueryResult(BaseModel):
    """The outcome of a structured catalog query.

    Carries both the matched records and the compiled RediSearch expression that
    produced them. Surfacing the compiled query is deliberate: it makes the
    structured path auditable and is exactly what you would show when explaining
    that the system queries rather than stuffs context.
    """

    services: list[ServiceRecord] = Field(default_factory=list, description="Matched services.")
    compiled_query: str = Field(..., description="The RediSearch expression that was executed.")
    truncated: bool = Field(default=False, description="Whether more matching rows exist beyond the limit.")
