# GTD Guide

This guide covers Alloy's GTD (Getting Things Done) features for personal productivity and task management.

## Overview

Alloy implements David Allen's GTD methodology with intelligent automation:

- **Automatic Extraction** - Tasks, projects, and commitments are extracted from indexed documents
- **Project Health Scoring** - Projects are automatically scored for health (0-100)
- **Context-Aware Recommendations** - Get task suggestions based on your current context
- **Weekly Review Assistant** - Comprehensive reports for your weekly review
- **Horizon Mapping** - View commitments across all 6 GTD levels

## Core Concepts

### Projects

A **Project** is any outcome requiring more than one action step. Projects have:

- **Outcome** - The desired end result
- **Status** - `active`, `on_hold`, `completed`, `archived`
- **Area** - Area of focus/responsibility
- **Health Score** - Automatic health metric (0-100)
- **Next Action** - The single next physical action

```
# List all active projects
gtd_projects(action: "list", filters: { status: "active" })

# Create a new project
gtd_projects(action: "create", project: {
  name: "Website Redesign",
  outcome: "Launch new company website with improved conversion rates",
  area: "Marketing"
})
```

### Tasks

A **Task** (or Next Action) is a single physical action. Tasks have:

- **Contexts** - Where/how it can be done (`@home`, `@computer`, etc.)
- **Energy Level** - Required energy (`low`, `medium`, `high`)
- **Duration** - Estimated time
- **Priority** - Importance (`low`, `medium`, `high`, `critical`)
- **Due Date** - Deadline (if any)

```
# Get tasks I can do at the computer
gtd_tasks(action: "list", filters: { contexts: ["@computer"] })

# Low energy tasks under 15 minutes
gtd_tasks(action: "list", filters: { energy_level: "low", time_available: 15 })
```

### Contexts

Contexts describe where or how a task can be done:

| Context | Description |
|---------|-------------|
| `@home` | Tasks requiring home environment |
| `@work` | Tasks for the office |
| `@computer` | Tasks requiring a computer |
| `@phone` | Phone calls to make |
| `@errand` | Tasks while out |
| `@anywhere` | Location-independent tasks |

Custom contexts can be configured in `config.toml`.

### Waiting For

Track items delegated to others:

```
# Add a waiting item
gtd_waiting(action: "add", item: {
  description: "Budget approval from finance",
  delegated_to: "Finance Team",
  expected_by: "2024-01-25"
})

# List overdue items
gtd_waiting(action: "list", filters: { status: "overdue" })
```

### Someday/Maybe

Items for future consideration:

```
# Add to someday/maybe
gtd_someday(action: "add", item: {
  description: "Learn Kubernetes",
  category: "Learning",
  trigger: "When current certifications are complete"
})

# Activate when ready
gtd_someday(action: "activate", item_id: "someday_xyz")
```

## Project Health Scoring

Projects are automatically scored on a 0-100 scale based on:

| Factor | Points | Description |
|--------|--------|-------------|
| Has Next Action | +30 | A clear next physical action is defined |
| Recent Activity | +20 | Activity within the last 7 days |
| Clear Outcome | +15 | Outcome statement exists |
| Reasonable Scope | +15 | Not too many tasks |
| No Blockers | +10 | No blocked tasks |
| Linked to Goal | +10 | Connected to a higher-level goal |

Projects with scores below 50 appear in the "needs attention" list.

```
# Find unhealthy projects
gtd_projects(action: "list", filters: { has_next_action: false })

# Find stalled projects
gtd_projects(action: "list", filters: { stalled_days: 7 })
```

## Task Recommendations

Get intelligent task recommendations based on:

- **Context** - Your current location/situation
- **Energy Level** - Your current energy
- **Time Available** - How much time you have
- **Focus Area** - Optional area to prioritize

```
# I'm at my computer, medium energy, have an hour
gtd_tasks(action: "recommend", filters: {
  contexts: ["@computer"],
  energy_level: "medium",
  time_available: 60
})

# Low energy, short time - what quick wins exist?
gtd_tasks(action: "recommend", filters: {
  energy_level: "low",
  time_available: 15
})
```

The recommendation engine scores tasks based on:

- Context match (+30 points)
- Energy level match (+20 points)
- Time fit (+15 points)
- Due date urgency (+20 points)
- Priority (+15 points)

## Weekly Review

The weekly review is essential to GTD. Alloy generates comprehensive reports:

```
# Full weekly review
gtd_weekly_review()

# Focus on specific areas
gtd_weekly_review(sections: ["projects_review", "waiting_for", "stalled_projects"])
```

### Review Sections

| Section | Contents |
|---------|----------|
| `inbox_status` | Unprocessed items count |
| `completed_tasks` | Tasks completed this week |
| `projects_review` | All active projects with health |
| `stalled_projects` | Projects without recent activity |
| `waiting_for` | Delegated items, especially overdue |
| `upcoming_calendar` | Next week's calendar preview |
| `someday_maybe` | Deferred items for consideration |
| `areas_check` | Area of focus balance |

### Sample Review Output

```json
{
  "period": { "start": "2024-01-08", "end": "2024-01-14" },
  "inbox_count": 5,
  "tasks_completed": 23,
  "projects_active": 8,
  "stalled_projects": [
    { "name": "Mobile App", "days_stalled": 12 }
  ],
  "waiting_overdue": [
    { "description": "Contract review", "days_overdue": 3 }
  ],
  "suggestions": [
    "Process 5 inbox items to achieve inbox zero",
    "Follow up on overdue waiting items",
    "Define next action for Mobile App project"
  ]
}
```

## GTD Horizons

View your commitments across all 6 levels of GTD horizons:

| Level | Name | Description |
|-------|------|-------------|
| Runway | Current Actions | Tasks and next actions |
| 10,000 ft | Projects | Current projects |
| 20,000 ft | Areas | Areas of focus/responsibility |
| 30,000 ft | Goals | 1-2 year goals |
| 40,000 ft | Vision | 3-5 year vision |
| 50,000 ft | Purpose | Life purpose and principles |

```
# View all horizons
gtd_horizons()

# Focus on specific level
gtd_horizons(level: "h30k")
```

## Commitment Tracking

Automatically extract and track promises:

```
# List commitments I made
gtd_commitments(action: "list", commitment_type: "made")

# List commitments made to me
gtd_commitments(action: "list", commitment_type: "received")
```

The system detects patterns like:
- "I'll send you the report by Friday" (commitment made)
- "Can you review this by EOD?" (commitment received)

## Dependency Tracking

Visualize task dependencies and blockers:

```
# Analyze dependencies for a project
gtd_dependencies(project_id: "proj_xyz")
```

The response includes:
- Dependency graph (nodes and edges)
- Critical path (longest chain)
- Blockers with impact analysis

## Attention Economics

Track where your attention goes:

```
# Weekly attention analysis
gtd_attention(period: "week", group_by: "area")

# Monthly by project
gtd_attention(period: "month", group_by: "project")
```

Identifies imbalances between prioritized areas and actual attention.

## Natural Language Queries

Use natural language for common GTD queries:

```
# These automatically route to appropriate tools
query(query: "What should I work on now?")
query(query: "What's blocking the website project?")
query(query: "Show me stalled projects")
query(query: "What's waiting on John?")
query(query: "What did I accomplish this week?")
```

## Automatic Extraction

When indexing documents, Alloy can automatically extract:

- **Tasks** - Action items from meeting notes, emails
- **Commitments** - Promises detected in communications
- **Dates** - Deadlines, due dates, scheduled events
- **People** - Names mentioned in context

Enable in configuration:

```toml
[gtd]
mode = "inference_and_manual"
auto_create_projects = true
auto_link_references = true

[ontology.extraction]
extract_on_index = true
```

## Best Practices

### Daily Usage

1. Start with `gtd_tasks(action: "recommend")` for task suggestions
2. Check `gtd_waiting(filters: { status: "overdue" })` for follow-ups
3. Process inbox items with `gtd_process_inbox`

### Weekly Review

1. Run `gtd_weekly_review()` for full report
2. Address stalled projects first
3. Follow up on overdue waiting items
4. Review someday/maybe list
5. Ensure all projects have next actions

### Project Hygiene

- Every project should have a clear outcome
- Every project needs exactly one next action
- Projects without activity for 7+ days need attention
- Archive completed projects promptly

## Configuration

See [Configuration](configuration.md) for full options:

```toml
[gtd]
enabled = true
mode = "inference_and_manual"
default_contexts = ["@home", "@work", "@phone", "@computer", "@errand"]
stalled_project_days = 7
quick_task_minutes = 2
```
