"""The typed filter model for a service-catalog structured query.

The structured-query route answers questions about the *service catalog* - who
owns a service, which team is on call, how many services sit in a domain. Unlike
free-text knowledge, this data is structured, so the right pattern is a precise
query, not retrieval-then-stuff. This model is the contract the router's
extraction step fills in, and it's what the :class:`QueryBuilder` compiles into
a RediSearch expression.

Keeping the filter as a typed model (rather than a loose dict) means the builder
never has to guess a field's type, and an invalid filter fails at construction
rather than deep inside query compilation.
"""

from pydantic import BaseModel, Field


class ServiceFilterModel(BaseModel):
    """Structured filter over the service catalog index.

    Every field is optional; a query with no fields set matches all services the
    caller is allowed to see. Fields map onto the three RediSearch field kinds
    the builder understands: tags (exact match), text (fuzzy match), and numeric
    (ranges).
    """

    # Tag fields - exact match.
    team: str | None = Field(default=None, description="Owning team, exact match.")
    domain: str | None = Field(default=None, description="Business domain, exact match.")
    tier: str | None = Field(default=None, description="Criticality tier, e.g. 'tier1'.")
    lifecycle: str | None = Field(default=None, description="Lifecycle state, e.g. 'active'.")
    has_owner: bool | None = Field(default=None, description="Whether an owner is assigned.")

    # Text fields - fuzzy match.
    name: str | None = Field(default=None, description="Service name, fuzzy match.")
    description: str | None = Field(default=None, description="Description text, fuzzy match.")

    # Numeric fields - ranges.
    min_dependencies: int | None = Field(default=None, description="Minimum dependency count.")
    max_dependencies: int | None = Field(default=None, description="Maximum dependency count.")
