---
name: schema-exploration
description: "Explore database schema -- list tables, examine columns and relationships"
---

# Schema Exploration Skill

## When to Use
- User asks about available tables or data structure
- Before writing any query (always explore schema first)
- When you need to understand table relationships

## Workflow
1. Use `query_db(sql="SELECT name FROM sqlite_master WHERE type='table'")` to list all tables
2. For each relevant table, run `query_db(sql="PRAGMA table_info(TableName)")` to see columns
3. Identify primary keys, foreign keys, and relationships
4. Summarize the schema for the user
