# CLI Reference

Alloy provides a full-featured command-line interface for indexing, searching, and managing documents with GTD workflow support and knowledge graph capabilities.

## Overview

```
alloy [OPTIONS] [COMMAND]

Commands:
  index      Index a local path or S3 URI
  search     Search indexed documents
  get        Get a document by ID
  sources    List indexed sources
  remove     Remove an indexed source
  stats      Show index statistics
  cluster    Cluster documents by semantic similarity
  backup     Create a backup of the index
  restore    Restore from a backup
  export     Export documents to a file
  import     Import documents from a file
  backups    List available backups
  serve      Run as MCP server (default)
  gtd        GTD workflow management
  calendar   Calendar queries and event management
  knowledge  Knowledge graph queries
  query      Natural language query
  ontology   Ontology management

Options:
  -c, --config <PATH>   Path to configuration file
      --json            Output as JSON
  -r, --remote <URL>    Connect to remote Alloy server via MCP
  -h, --help            Print help
  -V, --version         Print version
```

## Global Options

### `--config <PATH>`

Specify a custom configuration file path. If not provided, Alloy looks for configuration in the standard locations (see [Configuration](configuration.md)).

```bash
alloy --config ~/myconfig.toml index ~/docs
```

### `--json`

Output results as JSON instead of human-readable format. Useful for scripting and automation.

```bash
alloy --json search "query" | jq '.results[0].path'
alloy --json stats > stats.json
```

### `--remote <URL>`

Connect to a remote Alloy server instead of using local storage. The remote server must be running with HTTP transport.

```bash
# Local server
alloy serve --transport http --port 8080

# Remote client
alloy --remote http://localhost:8080 search "query"
alloy -r http://server:8080 stats
```

---

## Core Commands

### index

Index a local directory or S3 URI for search.

```
alloy index [OPTIONS] <PATH>

Arguments:
  <PATH>  Path to index (local path or S3 URI like s3://bucket/prefix)

Options:
  -p, --pattern <PATTERN>  Glob pattern to filter files (e.g., "*.md")
  -w, --watch              Watch for changes and auto re-index
      --recursive          Recursive indexing (default: true)
```

**Examples:**

```bash
# Index all files in a directory
alloy index ~/documents

# Index only markdown files
alloy index ~/projects --pattern "*.md"

# Index with file watching
alloy index ~/notes --watch

# Index code files
alloy index ~/code/myapp --pattern "**/*.{rs,py,js,ts}"

# Index S3 bucket
alloy index s3://my-bucket/docs

# Index S3 with pattern
alloy index s3://my-bucket/documents --pattern "*.pdf"
```

**Output:**

```
Indexed 42 documents
Source ID: src_abc123def456
Chunks created: 156
Watching: false
Successfully indexed from ~/documents
```

### search

Search indexed documents using hybrid vector + full-text search.

```
alloy search [OPTIONS] <QUERY>

Arguments:
  <QUERY>  Search query text

Options:
  -l, --limit <LIMIT>            Maximum results (default: 10)
  -w, --vector-weight <WEIGHT>   Vector search weight 0.0-1.0 (default: 0.5)
  -s, --source <SOURCE_ID>       Filter by source ID
```

**Examples:**

```bash
# Basic search
alloy search "async error handling"

# Get more results
alloy search "configuration" --limit 20

# Semantic-focused search (conceptual queries)
alloy search "how to handle failures gracefully" --vector-weight 0.8

# Keyword-focused search (exact matches)
alloy search "RUST_LOG environment" --vector-weight 0.2

# Filter by source
alloy search "TODO" --source src_abc123

# JSON output for scripting
alloy --json search "query" | jq '.results[] | {path, score}'
```

**Output:**

```
Found 5 results (42ms)

1. [0.872] /home/user/docs/error-handling.md
   "Error handling in async Rust requires careful consideration of..."

2. [0.756] /home/user/docs/patterns.md
   "Common patterns for handling errors include Result types and..."

3. [0.698] /home/user/code/main.rs
   "fn handle_error(e: Error) -> Result<()> { ... }"
```

#### Vector Weight Guide

| Weight | Behavior |
|--------|----------|
| `0.0` | Pure full-text search (keyword matching) |
| `0.3` | Primarily keywords with some semantic |
| `0.5` | Balanced hybrid (default) |
| `0.7` | Primarily semantic with keyword boost |
| `1.0` | Pure vector search (semantic similarity) |

### get

Retrieve a document by its ID.

```
alloy get [OPTIONS] <DOCUMENT_ID>

Arguments:
  <DOCUMENT_ID>  Document ID

Options:
      --no-content  Don't include document content (metadata only)
```

**Examples:**

```bash
# Get document with content
alloy get doc_xyz789

# Get metadata only (faster)
alloy get doc_xyz789 --no-content

# JSON output
alloy --json get doc_xyz789 | jq '.path, .size_bytes'
```

### sources

List all indexed sources.

```
alloy sources
```

**Output:**

```
SOURCE ID                                TYPE       DOCS     WATCH    PATH
----------------------------------------------------------------------------------------------------
src_abc123def456789012345678901234       local      42       yes      /home/user/docs
src_def456abc789012345678901234567       s3         100      no       s3://my-bucket/documents

Total: 2 sources
```

### remove

Remove an indexed source and all its documents.

```
alloy remove <SOURCE_ID>

Arguments:
  <SOURCE_ID>  Source ID to remove
```

### stats

Show index statistics.

```
alloy stats
```

**Output:**

```
Index Statistics
========================================
Sources:            3
Documents:          150
Chunks:             1200
Storage:            52428800 bytes
Embedding Dim:      384
Storage Backend:    Embedded
Embedding Provider: Local
Uptime:             3600s
```

### cluster

Cluster indexed documents by semantic similarity.

```
alloy cluster [OPTIONS]

Options:
  -s, --source <SOURCE_ID>      Filter by source ID
  -a, --algorithm <ALGORITHM>   Clustering algorithm: kmeans or dbscan (default: kmeans)
  -n, --num-clusters <N>        Number of clusters (for k-means, default: auto-detect)
```

### backup / restore / backups

Manage index backups.

```bash
# Create a backup
alloy backup --output ~/backup.tar --description "Before migration"

# List available backups
alloy backups

# Restore from backup
alloy restore ~/backup.tar
```

### export / import

Export and import documents.

```bash
# Export all documents as JSONL
alloy export documents.jsonl --format jsonl

# Export with embeddings (large file)
alloy export full-export.json --format json --include-embeddings

# Import documents
alloy import documents.jsonl
```

### serve

Run Alloy as an MCP server.

```
alloy serve [OPTIONS]

Options:
  -t, --transport <TYPE>  Transport type: stdio or http (default: stdio)
  -p, --port <PORT>       HTTP port (default: 8080)
      --https             Enable HTTPS with auto-generated certificates
      --json-logs         Enable JSON logging format
```

**Examples:**

```bash
# Stdio transport (for MCP clients)
alloy serve

# HTTP transport (for network access)
alloy serve --transport http --port 8080

# HTTPS with auto-generated certificates
alloy serve --https --port 8443

# With JSON logging
alloy serve --transport http --json-logs
```

#### HTTPS Mode

When using `--https`, Alloy automatically:

1. Creates a local Certificate Authority (CA) on first run
2. Generates a localhost certificate signed by the CA
3. Attempts to install the CA to your system trust store
4. Starts the HTTPS server

The CA certificate is stored at `~/.local/share/alloy/certs/alloy-ca.pem` and persists across restarts.

**First-time setup output:**

```
Generating new local CA certificate
CA certificate installed to macOS keychain
CA certificate: /Users/you/.local/share/alloy/certs/alloy-ca.pem
Alloy MCP server listening on https://0.0.0.0:8443
```

If automatic CA installation fails, you'll see manual instructions:

```bash
# macOS
security add-trusted-cert -r trustRoot -k ~/Library/Keychains/login.keychain-db ~/.local/share/alloy/certs/alloy-ca.pem

# Linux (Debian/Ubuntu)
sudo cp ~/.local/share/alloy/certs/alloy-ca.pem /usr/local/share/ca-certificates/alloy-ca.crt
sudo update-ca-certificates

# Windows
certutil -user -addstore Root ~/.local/share/alloy/certs/alloy-ca.pem
```

---

## GTD Commands

GTD (Getting Things Done) commands provide comprehensive task and project management following David Allen's methodology.

### alloy gtd projects

List and manage GTD projects.

```
alloy gtd projects [OPTIONS]

Options:
  -a, --action <ACTION>   Action: list, get, create, update, archive, complete (default: list)
  -i, --id <ID>           Project ID (for get/update/archive/complete)
      --status <STATUS>   Filter: active, on_hold, completed, archived
      --area <AREA>       Filter by area of focus
      --stalled <DAYS>    Show projects stalled for N days
      --no-next-action    Show only projects without next actions

Create/Update Options:
  -n, --name <NAME>       Project name
      --outcome <TEXT>    Desired outcome
      --set-area <AREA>   Area of focus
      --goal <GOAL_ID>    Supporting goal ID
```

**Examples:**

```bash
# List all active projects
alloy gtd projects

# List stalled projects (7+ days without activity)
alloy gtd projects --stalled 7

# Show projects missing next actions
alloy gtd projects --no-next-action

# Get a specific project
alloy gtd projects -a get --id proj_abc123

# Create a new project
alloy gtd projects -a create \
  --name "Website Redesign" \
  --outcome "Launch new company website with improved UX" \
  --set-area "Marketing"

# Update a project
alloy gtd projects -a update --id proj_abc123 --name "New Name"

# Archive a project
alloy gtd projects -a archive --id proj_abc123

# Complete a project
alloy gtd projects -a complete --id proj_abc123
```

### alloy gtd tasks

List and manage tasks with context-aware recommendations.

```
alloy gtd tasks [OPTIONS]

Options:
  -a, --action <ACTION>       Action: list, get, create, update, complete, recommend (default: list)
  -i, --id <ID>               Task ID (for get/update/complete)
      --context <CTX>         Filter by context (@home, @work, @computer, etc.)
  -e, --energy <LEVEL>        Filter by energy: low, medium, high
  -t, --time <MINUTES>        Filter by max duration
  -p, --project <PROJECT_ID>  Filter by project ID
      --status <STATUS>       Filter: next, scheduled, waiting, someday, done
      --due-before <DATE>     Due before date (YYYY-MM-DD)
      --description-contains <TEXT>  Filter by description (case-insensitive substring)
  -l, --limit <N>             Result limit (default: 20)

Create/Update Options:
  -d, --description <TEXT>    Task description
      --set-contexts <CTX>    Contexts (comma-separated)
      --set-energy <LEVEL>    Energy level: low, medium, high
      --priority <PRIORITY>   Priority: low, medium, high, critical
      --duration <MINUTES>    Estimated duration
      --due <DATE>            Due date (YYYY-MM-DD)
      --scheduled <DATE>      Scheduled date (YYYY-MM-DD)
      --assign-project <ID>   Assign to project
      --blocked-by <TASK_ID>  Blocked by task
```

**Examples:**

```bash
# List all tasks
alloy gtd tasks

# Filter tasks by description (case-insensitive)
alloy gtd tasks --description-contains "report"
alloy gtd tasks --description-contains "review" --status next

# Get task recommendations based on context
alloy gtd tasks -a recommend --context @computer --energy low --time 30

# List tasks for a specific project
alloy gtd tasks --project proj_website

# Create a new task
alloy gtd tasks -a create \
  --description "Review homepage mockups" \
  --set-contexts "@computer,@work" \
  --set-energy medium \
  --priority high \
  --duration 30 \
  --due 2024-02-01 \
  --assign-project proj_abc123

# Update a task
alloy gtd tasks -a update --id task_xyz --priority critical --due 2024-01-22

# Complete a task
alloy gtd tasks -a complete --id task_xyz
```

### alloy gtd waiting

Track delegated items waiting on others.

```
alloy gtd waiting [OPTIONS]

Options:
  -a, --action <ACTION>      Action: list, add, update, resolve, remind (default: list)
  -i, --id <ID>              Waiting item ID
      --status <STATUS>      Filter: pending, overdue, resolved
      --person <PERSON>      Filter by person
      --project <PROJECT>    Filter by project

Create Options:
  -d, --description <TEXT>   Description
      --delegated-to <NAME>  Person delegated to
      --expected-by <DATE>   Expected completion date (YYYY-MM-DD)
      --for-project <ID>     Related project ID
      --resolution <TEXT>    Resolution note (for resolve action)
```

**Examples:**

```bash
# List all waiting items
alloy gtd waiting

# List overdue items
alloy gtd waiting --status overdue

# Add a waiting-for item
alloy gtd waiting -a add \
  --description "Design mockups from Sarah" \
  --delegated-to "Sarah" \
  --expected-by 2024-01-25 \
  --for-project proj_abc123

# Resolve a waiting item
alloy gtd waiting -a resolve --id wait_abc --resolution "Received mockups"
```

### alloy gtd someday

Manage someday/maybe items.

```
alloy gtd someday [OPTIONS]

Options:
  -a, --action <ACTION>      Action: list, add, update, activate, archive (default: list)
  -i, --id <ID>              Item ID
      --category <CATEGORY>  Filter by category

Create/Update Options:
  -d, --description <TEXT>   Description
      --set-category <CAT>   Category
      --trigger <TEXT>       Trigger condition
      --review-date <DATE>   Review date (YYYY-MM-DD)
```

**Examples:**

```bash
# List all someday items
alloy gtd someday

# Filter by category
alloy gtd someday --category "Learning"

# Add a someday/maybe item
alloy gtd someday -a add \
  --description "Learn Kubernetes" \
  --set-category "Learning" \
  --trigger "When current project wraps up"

# Activate item (convert to project)
alloy gtd someday -a activate --id someday_abc
```

### alloy gtd review

Run the weekly review to get a comprehensive overview of your GTD system.

```
alloy gtd review [OPTIONS]

Options:
      --week-ending <DATE>   Week ending date (default: today)
  -s, --sections <SECTIONS>  Sections to include (comma-separated)
```

**Examples:**

```bash
# Run full weekly review
alloy gtd review

# Review with specific sections
alloy gtd review --sections "projects,waiting,inbox"

# JSON output for processing
alloy --json gtd review | jq '.stalled_projects'
```

**Output:**

```
=== Weekly Review ===
Period: 2024-01-08 to 2024-01-15

SUMMARY
  Tasks Completed:  12
  Active Projects:  5
  Inbox Items:      3

STALLED PROJECTS (need attention)
  - Website Redesign (7 days)
  - API Documentation (14 days)

OVERDUE WAITING ITEMS
  - Design mockups (from Sarah)

RECOMMENDATIONS
  • Project "Website Redesign" needs a next action defined
  • Consider moving "Learn French" from Someday to Projects
```

### alloy gtd horizons

View and manage GTD horizons (levels of focus).

```
alloy gtd horizons [OPTIONS]

Options:
  -a, --action <ACTION>  Action: list, add (default: list)
  -l, --level <LEVEL>    Level: runway, h10k, h20k, h30k, h40k, h50k

Create Options:
  -n, --name <NAME>          Name
  -d, --description <TEXT>   Description
```

**Examples:**

```bash
# View all horizons
alloy gtd horizons

# View specific horizon level
alloy gtd horizons --level h30k

# Add at specific level (e.g., a vision)
alloy gtd horizons -a add --level h40k \
  --name "Industry thought leader" \
  --description "Recognized expert in distributed systems"
```

### alloy gtd commitments

Track commitments made and received.

```
alloy gtd commitments [OPTIONS]

Options:
  -a, --action <ACTION>      Action: list, add, resolve (default: list)
  -i, --id <ID>              Commitment ID
  -f, --filter <FILTER>      Filter: made, received, all (default: all)
      --pending              Show only pending commitments

Create Options:
  -d, --description <TEXT>     Description
      --commitment-type <TYPE> Type: made, received
      --person <PERSON>        To/From person
      --due <DATE>             Due date (YYYY-MM-DD)
      --resolution <TEXT>      Resolution note (for resolve)
```

**Examples:**

```bash
# List all commitments
alloy gtd commitments

# List commitments made to others
alloy gtd commitments --filter made

# Add a commitment
alloy gtd commitments -a add \
  --description "Send quarterly report" \
  --commitment-type made \
  --person "Management" \
  --due 2024-01-31

# Resolve a commitment
alloy gtd commitments -a resolve --id commit_abc --resolution "Report sent"
```

### alloy gtd dependencies

Analyze task dependencies and identify blockers.

```
alloy gtd dependencies [OPTIONS]

Options:
  -a, --action <ACTION>  Action: list, add (default: list)
  -p, --project <ID>     Project ID to analyze
      --critical-path    Show critical path

Create Dependency Options:
      --blocked-task <ID>   Task that is blocked
      --blocking-task <ID>  Task that blocks
```

**Examples:**

```bash
# List all dependencies
alloy gtd dependencies

# Show dependencies for a project with critical path
alloy gtd dependencies --project proj_website --critical-path

# Add a dependency
alloy gtd dependencies -a add \
  --blocked-task task_deploy \
  --blocking-task task_testing
```

### alloy gtd attention

Analyze attention economics - where your time goes.

```
alloy gtd attention [OPTIONS]

Options:
  -p, --period <PERIOD>   Period: day, week, month (default: week)
  -g, --group-by <GROUP>  Group by: area, project (default: area)
```

**Examples:**

```bash
# Weekly attention breakdown by area
alloy gtd attention

# Monthly breakdown by project
alloy gtd attention --period month --group-by project
```

### alloy gtd areas

Manage areas of focus (ongoing responsibilities).

```
alloy gtd areas [OPTIONS]

Options:
  -a, --action <ACTION>  Action: list, add, update, archive (default: list)
  -i, --id <ID>          Area ID
  -n, --name <NAME>      Area name
  -d, --description <TEXT> Description
```

**Examples:**

```bash
# List areas
alloy gtd areas

# Add an area of focus
alloy gtd areas -a add \
  --name "Health" \
  --description "Physical and mental wellbeing"

# Archive an area
alloy gtd areas -a archive --id area_old
```

### alloy gtd goals

Manage goals (1-2 year objectives).

```
alloy gtd goals [OPTIONS]

Options:
  -a, --action <ACTION>  Action: list, add, update, complete, archive (default: list)
  -i, --id <ID>          Goal ID
  -n, --name <NAME>      Goal name
  -d, --description <TEXT> Description
      --target-date <DATE> Target date (YYYY-MM-DD)
      --area <AREA_ID>     Related area ID
```

**Examples:**

```bash
# List goals
alloy gtd goals

# Add a goal
alloy gtd goals -a add \
  --name "Complete AWS certification" \
  --description "Get AWS Solutions Architect certification" \
  --target-date 2024-06-30 \
  --area area_career

# Complete a goal
alloy gtd goals -a complete --id goal_cert
```

---

## Calendar Commands

Calendar commands provide event management and scheduling intelligence.

### alloy calendar today

Show today's events.

```bash
alloy calendar today
```

### alloy calendar week

Show this week's events.

```bash
alloy calendar week
```

### alloy calendar range

Show events in a date range.

```
alloy calendar range --start <DATE> --end <DATE>

Options:
  -s, --start <DATE>  Start date (YYYY-MM-DD)
  -e, --end <DATE>    End date (YYYY-MM-DD)
```

**Example:**

```bash
alloy calendar range --start 2024-01-15 --end 2024-01-31
```

### alloy calendar free

Find free time slots.

```
alloy calendar free [OPTIONS]

Options:
  -s, --start <DATE>      Start date (YYYY-MM-DD)
  -e, --end <DATE>        End date (YYYY-MM-DD)
  -m, --min-duration <N>  Minimum slot duration in minutes (default: 30)
```

**Example:**

```bash
alloy calendar free --start 2024-01-15 --end 2024-01-19 --min-duration 60
```

### alloy calendar conflicts

Check for scheduling conflicts.

```bash
alloy calendar conflicts
```

### alloy calendar upcoming

Show upcoming events.

```
alloy calendar upcoming [OPTIONS]

Options:
  -l, --limit <N>  Number of events (default: 10)
```

### alloy calendar events

Manage calendar events (CRUD operations).

```
alloy calendar events [OPTIONS]

Options:
  -a, --action <ACTION>  Action: list, get, add, update, delete (default: list)
  -i, --id <ID>          Event ID (for get/update/delete)

Create/Update Options:
  -t, --title <TEXT>       Event title
      --event-type <TYPE>  Type: meeting, deadline, reminder, blocked_time
      --start <DATETIME>   Start datetime (YYYY-MM-DDTHH:MM)
      --end <DATETIME>     End datetime (YYYY-MM-DDTHH:MM)
      --location <TEXT>    Location
      --participants <CSV> Participants (comma-separated)
      --project <ID>       Related project ID
      --recurrence <TYPE>  Recurrence: daily, weekly, biweekly, monthly, yearly
      --notes <TEXT>       Notes

Filter Options (for list):
      --from <DATE>        Date range start
      --to <DATE>          Date range end
      --filter-type <TYPE> Filter by event type
```

**Examples:**

```bash
# List events
alloy calendar events

# Add a meeting
alloy calendar events -a add \
  --title "Team Standup" \
  --event-type meeting \
  --start "2024-01-20T09:00" \
  --end "2024-01-20T09:30" \
  --location "Zoom" \
  --participants "Alice,Bob,Charlie" \
  --recurrence weekly

# Add a deadline
alloy calendar events -a add \
  --title "Project deadline" \
  --event-type deadline \
  --start "2024-02-15T17:00" \
  --project proj_website

# Add blocked focus time
alloy calendar events -a add \
  --title "Deep work - coding" \
  --event-type blocked_time \
  --start "2024-01-20T14:00" \
  --end "2024-01-20T17:00" \
  --notes "No meetings during this time"

# Update an event
alloy calendar events -a update --id event_xyz --title "Updated title"

# Delete an event
alloy calendar events -a delete --id event_xyz
```

---

## Knowledge Commands

Knowledge commands query the semantic knowledge graph built from indexed documents.

### alloy knowledge search

Semantic search across the knowledge graph.

```
alloy knowledge search [OPTIONS] <QUERY>

Arguments:
  <QUERY>  Search query

Options:
  -t, --types <TYPES>  Filter by entity types (comma-separated)
  -l, --limit <N>      Result limit (default: 10)
```

**Examples:**

```bash
# Search for any knowledge
alloy knowledge search "machine learning best practices"

# Search for specific entity types
alloy knowledge search "AWS deployment" --types "Person,Topic"
```

### alloy knowledge entity

Look up a specific entity.

```
alloy knowledge entity [OPTIONS] <NAME>

Arguments:
  <NAME>  Entity name

Options:
  -r, --relationships  Show relationships
```

**Examples:**

```bash
# Get entity details
alloy knowledge entity "Alice Chen"

# Include relationships
alloy knowledge entity "Kubernetes" --relationships
```

### alloy knowledge expert

Find experts on a topic.

```
alloy knowledge expert [OPTIONS] <TOPIC>

Arguments:
  <TOPIC>  Topic to find experts for

Options:
  -l, --limit <N>  Result limit (default: 5)
```

**Examples:**

```bash
alloy knowledge expert "PostgreSQL"
alloy knowledge expert "distributed systems" --limit 10
```

### alloy knowledge topic

Get a summary of a topic.

```
alloy knowledge topic <TOPIC>

Arguments:
  <TOPIC>  Topic to summarize
```

**Example:**

```bash
alloy knowledge topic "authentication"
```

### alloy knowledge connected

Explore connected entities via graph traversal.

```
alloy knowledge connected [OPTIONS] <ENTITY>

Arguments:
  <ENTITY>  Starting entity name

Options:
  -d, --depth <N>  Max traversal depth (default: 2)
```

**Example:**

```bash
alloy knowledge connected "Project Alpha" --depth 3
```

---

## Natural Language Query

The query command routes natural language questions to the appropriate subsystem.

```
alloy query [OPTIONS] <QUERY>

Arguments:
  <QUERY>  Natural language query

Options:
  -m, --mode <MODE>  Force query mode: auto, gtd, calendar, knowledge
```

**Examples:**

```bash
# Auto-detect query type
alloy query "What should I work on now?"
alloy query "Who knows about Kubernetes?"
alloy query "What meetings do I have this week?"
alloy query "What's blocking the website redesign project?"

# Force specific mode
alloy query "stalled projects" --mode gtd
alloy query "AWS expertise" --mode knowledge
```

---

## Ontology Commands

Ontology commands manage the semantic entity and relationship graph.

### alloy ontology stats

Show ontology statistics.

```bash
alloy ontology stats
```

**Output:**

```
Ontology Statistics
========================================
Total Entities:     256
  - Person:         45
  - Organization:   12
  - Topic:          89
  - Project:        23
  - Task:           67
  - Other:          20

Total Relationships: 412
  - works_for:      38
  - knows_about:    156
  - belongs_to:     89
  - mentions:       129
```

### alloy ontology entities

Manage entities in the ontology.

```
alloy ontology entities [OPTIONS]

Options:
  -a, --action <ACTION>    Action: list, get, add, update, delete, merge (default: list)
  -i, --id <ID>            Entity ID
  -t, --entity-type <TYPE> Filter by type
      --name-contains <TEXT> Search by name
  -l, --limit <N>          Result limit (default: 50)

Create/Update Options:
      --set-type <TYPE>    Entity type (Person, Organization, Topic, etc.)
  -n, --name <NAME>        Entity name
      --aliases <CSV>      Aliases (comma-separated)
      --metadata <JSON>    Metadata JSON

Merge Options:
      --merge-into <ID>    Target entity ID for merge
```

**Examples:**

```bash
# List all entities
alloy ontology entities

# Filter by type
alloy ontology entities --entity-type Person

# Search by name
alloy ontology entities --name-contains "Alice"

# Add a custom entity
alloy ontology entities -a add \
  --set-type Concept \
  --name "Microservices Architecture" \
  --aliases "microservices,MSA"

# Merge duplicate entities
alloy ontology entities -a merge --id entity_dup --merge-into entity_main
```

### alloy ontology relationships

Manage relationships between entities.

```
alloy ontology relationships [OPTIONS]

Options:
  -a, --action <ACTION>  Action: list, add, delete (default: list)
  -i, --id <ID>          Relationship ID (for delete)
      --source <ID>      Filter by source entity
      --target <ID>      Filter by target entity
      --rel-type <TYPE>  Filter by relationship type
  -l, --limit <N>        Result limit (default: 50)

Create Options:
      --from-entity <ID>  Source entity ID
      --to-entity <ID>    Target entity ID
      --set-type <TYPE>   Relationship type
```

**Examples:**

```bash
# List all relationships
alloy ontology relationships

# Filter by entity
alloy ontology relationships --source entity_alice

# Add a relationship
alloy ontology relationships -a add \
  --from-entity entity_alice \
  --to-entity entity_acme \
  --set-type works_for

# Delete a relationship
alloy ontology relationships -a delete --id rel_xyz
```

### alloy ontology extract

Extract entities from a document.

```
alloy ontology extract [OPTIONS] <DOCUMENT>

Arguments:
  <DOCUMENT>  Document ID or path

Options:
      --show-confidence  Show confidence scores
      --auto-add         Auto-add extracted entities to ontology
```

**Examples:**

```bash
# Extract and review entities
alloy ontology extract doc_xyz

# Extract with confidence scores
alloy ontology extract doc_xyz --show-confidence

# Extract and auto-add to ontology
alloy ontology extract doc_xyz --auto-add
```

### alloy ontology person

Shortcut to add a person entity with common attributes.

```
alloy ontology person [OPTIONS] <NAME>

Arguments:
  <NAME>  Person name

Options:
      --organization <ORG>  Organization they work for
      --email <EMAIL>       Email address
      --topics <CSV>        Topics they know about (comma-separated)
      --aliases <CSV>       Aliases (comma-separated)
```

**Example:**

```bash
alloy ontology person "Alice Chen" \
  --organization "Acme Corp" \
  --email "alice@acme.com" \
  --topics "Kubernetes,DevOps,AWS" \
  --aliases "A. Chen"
```

### alloy ontology organization

Shortcut to add an organization entity.

```
alloy ontology organization [OPTIONS] <NAME>

Arguments:
  <NAME>  Organization name

Options:
      --org-type <TYPE>  Organization type (company, team, department)
      --parent <ORG>     Parent organization
      --aliases <CSV>    Aliases (comma-separated)
```

**Example:**

```bash
alloy ontology organization "Acme Corp" \
  --org-type company \
  --aliases "Acme,Acme Corporation"
```

### alloy ontology topic

Shortcut to add a topic entity.

```
alloy ontology topic [OPTIONS] <NAME>

Arguments:
  <NAME>  Topic name

Options:
  -d, --description <TEXT> Description
      --parent <TOPIC>     Parent topic
      --aliases <CSV>      Aliases (comma-separated)
```

**Example:**

```bash
alloy ontology topic "Machine Learning" \
  --description "AI and ML technologies" \
  --aliases "ML,AI"
```

---

## Remote Mode

Remote mode allows you to connect to a running Alloy server instead of using local storage. This is useful for:

- Querying a shared index from multiple machines
- Running searches without local index files
- Testing server deployments

### Setup

1. Start the server with HTTP or HTTPS transport:

```bash
# HTTP
alloy serve --transport http --port 8080

# HTTPS (recommended for production)
alloy serve --https --port 8443
```

2. Use CLI with `--remote`:

```bash
# HTTP
alloy --remote http://localhost:8080 search "query"
alloy -r http://server.example.com:8080 stats

# HTTPS
alloy --remote https://localhost:8443 search "query"
alloy -r https://server.example.com:8443 stats
```

All commands work with remote mode:

```bash
alloy -r http://localhost:8080 index ~/docs
alloy -r http://localhost:8080 search "query"
alloy -r http://localhost:8080 gtd projects
alloy -r http://localhost:8080 gtd tasks -a create --description "Remote task"
alloy -r http://localhost:8080 calendar today
alloy -r http://localhost:8080 knowledge expert "kubernetes"
alloy -r http://localhost:8080 ontology stats
alloy -r http://localhost:8080 query "What should I work on?"
```

---

## Exit Codes

| Code | Meaning |
|------|---------|
| 0 | Success |
| 1 | General error |
| 2 | Configuration error |

---

## Scripting Examples

### Daily GTD Workflow

```bash
#!/bin/bash
# Morning routine script

echo "=== Morning Review ==="

# What's due today?
alloy gtd tasks --due-before $(date +%Y-%m-%d) --status next

# Overdue waiting items
alloy gtd waiting --status overdue

# Today's calendar
alloy calendar today

# Recommended tasks based on context
echo -e "\n=== Recommended Tasks (low energy, at computer) ==="
alloy gtd tasks -a recommend --context @computer --energy low --time 30
```

### Weekly Review Script

```bash
#!/bin/bash
# Weekly review automation

echo "=== Weekly Review ==="
alloy gtd review

echo -e "\n=== Stalled Projects ==="
alloy gtd projects --stalled 7

echo -e "\n=== Projects Without Next Actions ==="
alloy gtd projects --no-next-action

echo -e "\n=== Attention Economics ==="
alloy gtd attention --period week --group-by area
```

### Search and Process Results

```bash
# Find files containing "TODO" and extract paths
alloy --json search "TODO" --limit 100 | \
    jq -r '.results[].path' | \
    sort -u

# Get total document count
alloy --json stats | jq '.document_count'

# Export projects to JSON
alloy --json gtd projects > projects.json
```

### Monitoring

```bash
# Watch stats every 5 seconds
watch -n 5 'alloy --json stats | jq "{docs: .document_count, chunks: .chunk_count}"'
```

### Batch Operations

```bash
#!/bin/bash
# Index multiple directories
for dir in ~/projects/*/; do
    alloy index "$dir" --pattern "*.md"
done

# Create tasks from a file
while IFS= read -r task; do
    alloy gtd tasks -a create --description "$task" --set-contexts "@computer"
done < tasks.txt
```

### Remote Health Check

```bash
# Verify remote server is healthy
if alloy -r http://server:8080 --json stats | jq -e '.source_count > 0' > /dev/null; then
    echo "Server healthy"
else
    echo "Server may have issues"
    exit 1
fi
```
