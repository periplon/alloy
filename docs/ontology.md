# Ontology & Knowledge Graph

This guide covers Alloy's semantic ontology layer for entity extraction and knowledge graph queries.

## Overview

Alloy builds a knowledge graph from your indexed documents:

- **Entity Extraction** - Automatically identify people, organizations, topics, dates
- **Relationship Mapping** - Track connections between entities
- **Knowledge Queries** - Ask "What do I know about X?"
- **Expert Finding** - Discover who knows what based on document analysis

## Entity Types

Alloy recognizes 18 entity types organized into categories:

### GTD Types

| Type | Description |
|------|-------------|
| `Project` | Outcome requiring multiple actions |
| `Task` | Single next action |
| `WaitingFor` | Delegated item |
| `SomedayMaybe` | Deferred item |
| `Reference` | Knowledge/reference material |
| `CalendarEvent` | Time-specific commitment |
| `Context` | Location/situation (@home, @work) |
| `Area` | Area of focus/responsibility |
| `Goal` | 1-2 year outcome |
| `Vision` | 3-5 year vision |
| `Purpose` | Life purpose/principles |

### Knowledge Types

| Type | Description |
|------|-------------|
| `Person` | People mentioned in documents |
| `Organization` | Companies, teams, groups |
| `Location` | Places and addresses |
| `Topic` | Subject areas |
| `Concept` | Abstract ideas |
| `Date` | Temporal references |
| `Commitment` | Promises made/received |

## Relationship Types

Entities are connected by 21 relationship types:

### GTD Relationships

| Type | Description |
|------|-------------|
| `belongs_to_project` | Task belongs to Project |
| `has_context` | Task has Context |
| `in_area` | Project is in Area |
| `supports_goal` | Project supports Goal |
| `waiting_on` | Task waiting on Person |
| `delegated_to` | Task delegated to Person |
| `blocked_by` | Task blocked by Task |
| `depends_on` | Task depends on Task |
| `scheduled_for` | Task scheduled for CalendarEvent |
| `due_on` | Task due on Date |

### Knowledge Relationships

| Type | Description |
|------|-------------|
| `mentions` | Document mentions Entity |
| `related_to` | Entity related to Entity |
| `authored_by` | Document authored by Person |
| `about_topic` | Entity about Topic |
| `located_at` | Entity located at Location |
| `works_for` | Person works for Organization |
| `committed_to` | Person committed to Commitment |
| `references` | Task references Reference |
| `part_of` | Entity part of Entity |
| `contains` | Entity contains Entity |
| `same_as` | Entity same as Entity |

## Entity Extraction

### Automatic Extraction

When documents are indexed, Alloy can automatically extract entities:

```toml
[ontology]
enabled = true

[ontology.extraction]
extract_on_index = true
confidence_threshold = 0.7

[ontology.extraction.local]
enable_temporal = true    # Parse dates and times
enable_patterns = true    # Pattern-based NER
enable_embedding_ner = true
```

### Extraction Methods

| Method | Description |
|--------|-------------|
| Pattern | Regex-based entity detection |
| Temporal | Date, time, and recurrence parsing |
| ActionDetection | Task and commitment extraction |
| LocalNer | Heuristic-based named entity recognition |
| LlmNer | LLM-based extraction (optional) |
| CoOccurrence | Relationship from co-mention |
| Semantic | Embedding-based similarity |
| Manual | User-created entities |

### Temporal Parsing

The temporal parser handles natural language dates:

| Input | Parsed As |
|-------|-----------|
| "January 15, 2024" | Specific date |
| "tomorrow" | Relative date |
| "next Tuesday" | Relative date |
| "in 2 weeks" | Relative date |
| "by end of month" | Deadline |
| "every Monday" | Recurring weekly |
| "quarterly" | Recurring quarterly |
| "3pm" | Time |
| "from Monday to Friday" | Date range |

### Action Detection

The system detects action items from prose:

| Pattern | Detected As |
|---------|-------------|
| "I'll send you the report" | Commitment (made) |
| "Can you review this?" | Commitment (received) |
| "Follow up with Sarah" | Task |
| "Waiting for approval" | WaitingFor |
| "Blocked by design review" | BlockedItem |

### LLM-Enhanced Extraction (Optional)

For higher accuracy, enable LLM-based extraction:

```toml
[ontology.extraction.llm]
enabled = true
provider = "openai"
model = "gpt-4o-mini"
extract_tasks = true
extract_relationships = true
rate_limit_rpm = 60
```

## Knowledge Queries

### Semantic Search

Find information across your knowledge base:

```
# What do I know about machine learning?
knowledge_query(query: "machine learning", query_type: "semantic_search")

# Search for specific entity types
knowledge_query(
  query: "database",
  query_type: "semantic_search",
  entity_types: ["Person", "Topic"]
)
```

### Entity Lookup

Find a specific entity by name:

```
knowledge_query(query: "Alice Chen", query_type: "entity_lookup")
```

### Relationship Queries

Explore connections between entities:

```
# Who works with Alice?
knowledge_query(
  query: "Alice Chen",
  query_type: "relationship_query"
)

# Get connected entities
knowledge_query(
  query: "Kubernetes",
  query_type: "connected_entities"
)
```

### Topic Summary

Get a consolidated view of knowledge on a topic:

```
knowledge_query(
  query: "authentication",
  query_type: "topic_summary"
)
```

Returns:
- Related entities
- Key documents
- Summary of information
- Confidence score

### Expert Finding

Discover who knows about specific topics:

```
knowledge_query(
  query: "Who knows about Kubernetes?",
  query_type: "expert_finding"
)
```

The system identifies experts based on:
- Document authorship
- Frequency of mentions alongside topic
- Recency of mentions
- Connection strength

### Response Format

```json
{
  "entities": [
    {
      "id": "entity_abc",
      "type": "Person",
      "name": "Alice Chen",
      "aliases": ["A. Chen"],
      "metadata": {
        "organization": "Engineering",
        "topics": ["Kubernetes", "DevOps"]
      }
    }
  ],
  "relationships": [
    {
      "source": "Alice Chen",
      "type": "works_for",
      "target": "Acme Corp",
      "confidence": 0.92
    }
  ],
  "source_documents": [
    {
      "id": "doc_xyz",
      "path": "/meetings/2024-01-10.md",
      "relevance": 0.92
    }
  ],
  "summary": "Alice Chen is a DevOps engineer...",
  "confidence": 0.85
}
```

## Natural Language Interface

Use the unified query tool for natural language:

```
# These automatically route to knowledge queries
query(query: "What do I know about machine learning?")
query(query: "Who can help with the AWS migration?")
query(query: "What topics does Sarah know about?")
```

## Calendar Intelligence

### Date Extraction

Dates are automatically extracted from documents:

- Specific dates: "January 15, 2024"
- Relative dates: "next Tuesday", "in 2 weeks"
- Deadlines: "by end of month", "due Friday"
- Recurring: "every Monday", "weekly standup"

### Calendar Queries

```
# Today's events
calendar_query(query_type: "today")

# This week
calendar_query(query_type: "this_week")

# Find free time
calendar_query(query_type: "free_time", date_range: {
  start: "2024-01-15",
  end: "2024-01-19"
})

# Check for conflicts
calendar_query(query_type: "conflicts")
```

### Event Types

| Type | Description |
|------|-------------|
| `meeting` | Meetings with participants |
| `deadline` | Due dates |
| `reminder` | Time-based reminders |
| `blocked_time` | Focus/blocked time |
| `custom` | User-defined events |

## Entity Storage

Entities are stored with:

- Unique ID
- Type classification
- Name and aliases
- Embedding vector (for semantic similarity)
- Metadata (JSON)
- Source document references
- Confidence score
- Creation/update timestamps

Relationships store:
- Source and target entity references
- Relationship type
- Confidence score
- Source document references

## Graph Operations

### Traversal

Explore the knowledge graph:

```
# Get all entities connected to a node
knowledge_query(query: "Project Alpha", query_type: "connected_entities")
```

### Deduplication

The system automatically:
- Identifies potential duplicates via embeddings
- Merges entities with same_as relationships
- Maintains alias lists

### Confidence Scoring

All extracted entities have confidence scores (0.0-1.0):

- `0.9+` - High confidence (explicit mention)
- `0.7-0.9` - Good confidence (clear pattern match)
- `0.5-0.7` - Medium confidence (inference)
- `<0.5` - Low confidence (weak signal)

Configure threshold in `config.toml`:

```toml
[ontology.extraction]
confidence_threshold = 0.7
```

## Configuration

Full ontology configuration:

```toml
[ontology]
enabled = true
storage_backend = "embedded"

[ontology.extraction]
extract_on_index = true
confidence_threshold = 0.7

[ontology.extraction.local]
enable_temporal = true
enable_patterns = true
enable_embedding_ner = true

[ontology.extraction.llm]
enabled = false
provider = "openai"
model = "gpt-4o-mini"
extract_tasks = true
extract_relationships = true
max_tokens_per_doc = 4000
rate_limit_rpm = 60
```

## Best Practices

### Indexing for Knowledge

1. Index diverse document types (emails, notes, docs)
2. Include meeting notes for relationship extraction
3. Keep documents well-structured for better extraction

### Query Tips

1. Use semantic search for conceptual queries
2. Use entity lookup for specific names
3. Use expert finding for "who knows" questions
4. Filter by entity type to narrow results

### Maintenance

1. Review low-confidence entities periodically
2. Merge duplicate entities when discovered
3. Add aliases for commonly referred entities
