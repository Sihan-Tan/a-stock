---
name: query-writing
description: "Write and execute SQL queries to answer user questions about the database"
---

# Query Writing Skill

## When to Use
- User asks a question that requires querying the database
- User wants to analyze, count, aggregate, or compare data

## Workflow

### Step 1: Understand the Question
- Identify what tables and columns are needed
- If unsure about schema, read the `schema-exploration` skill first

### Step 2: Write the Query
- Start simple, then add complexity
- Use JOINs when data spans multiple tables
- Use GROUP BY for aggregations
- Use ORDER BY + LIMIT for "top N" questions
- Always use table aliases for clarity

### Step 3: Execute and Verify
- Run with `query_db(sql="YOUR QUERY")`
- If error: analyze the message, fix the query, retry
- If results look wrong: check JOINs and WHERE clauses

### Step 4: Present Results
- Format results as a clear answer in natural language
- Include the actual numbers/data
- For complex results, use a table format

## Error Recovery
- "no such table" -> re-list tables with schema-exploration
- "no such column" -> re-check table_info
- "ambiguous column" -> add table alias prefix
