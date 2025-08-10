# Update Documentation Hook

## Description
Automatically update documentation when code changes are made to ensure documentation stays synchronized with implementation.

## Trigger
File save events for source code files

## Actions
1. Update API documentation for modified modules
2. Regenerate examples if public APIs change
3. Update README files if project structure changes
4. Validate documentation links and references

## Configuration
```yaml
name: "Update Documentation"
trigger:
  event: "file_save"
  pattern: 
    - "**/src/**/*.py"
    - "**/utils/**/*.py"
    - "**/__init__.py"
conditions:
  - has_docstrings: true
  - not_in_path: "tests/"
actions:
  - extract_api_docs:
      source: "${file_path}"
      output: "docs/api/${module_name}.md"
  - update_readme:
      if_structure_changed: true
      template: "docs/templates/README_template.md"
  - validate_examples:
      pattern: "docs/examples/**/*.py"
      run_validation: true
  - check_links:
      files: "docs/**/*.md"
      fix_broken_links: true
notifications:
  docs_updated: "📚 Documentation updated for ${module_name}"
  examples_validated: "✅ Examples validated successfully"
  broken_links_found: "🔗 Broken links found and fixed in documentation"
```

## Expected Behavior
- Automatic API documentation generation
- README synchronization with code changes
- Example validation to ensure they work
- Link checking and fixing