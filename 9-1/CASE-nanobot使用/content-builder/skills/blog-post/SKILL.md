---
name: blog-post
description: "Write a blog post -- research, draft, and polish a complete blog article"
---

# Blog Post Writing Skill

## When to Use
- User asks to write a blog post, article, or content piece

## Workflow

### Step 1: Research
1. Use `web_search` to find 3-5 recent, relevant sources on the topic
2. Save research notes to `research/<slug>.md` using `write_file`
3. Identify key themes, data points, and expert quotes

### Step 2: Outline
Plan the structure:
- Title (compelling, specific)
- Hook (2 sentences max)
- 3-5 main sections with subheadings
- Conclusion with CTA

### Step 3: Write
Draft the full post following the brand voice from AGENTS.md:
- 600-1200 words for standard posts
- Under 200 words if user requests "short"
- Save to `blogs/<slug>/post.md`

### Step 4: Quality Check
Before finishing, verify:
- [ ] Hook grabs attention in first 2 sentences
- [ ] Each section has a clear point
- [ ] Sources are cited
- [ ] CTA tells reader what to do next

## Output Format
Save the final post as Markdown to `blogs/<slug>/post.md`.
