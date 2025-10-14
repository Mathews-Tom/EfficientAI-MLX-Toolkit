#!/usr/bin/env bash
set -e

echo "════════════════════════════════════════════════════════════════"
echo "        .kiro → .sage Migration Script"
echo "════════════════════════════════════════════════════════════════"
echo ""

# Ensure docs/features directory exists
echo "📁 Setting up directory structure..."
mkdir -p docs/features
echo "   ✓ Created docs/features/"
echo ""

# Counter for tracking
total=0
migrated=0
skipped=0

echo "🔄 Processing .kiro/specs components..."
echo ""

# Iterate through each component directory
for component_dir in .kiro/specs/*/; do
    # Skip if not a directory
    [ -d "$component_dir" ] || continue

    # Extract component name from directory path
    component=$(basename "$component_dir")

    # Skip .DS_Store and other hidden files
    [[ "$component" == .* ]] && continue

    total=$((total + 1))

    echo "Processing: $component"

    # Define paths
    requirements_file="$component_dir/requirements.md"
    design_file="$component_dir/design.md"
    tasks_file="$component_dir/tasks.md"
    output_file="docs/features/${component}.md"

    # Check if feature already exists
    if [ -f "$output_file" ]; then
        echo "   ⚠️  Already exists, skipping: $output_file"
        skipped=$((skipped + 1))
        echo ""
        continue
    fi

    # Read content from source files
    requirements_content=""
    design_content=""
    tasks_content=""

    if [ -f "$requirements_file" ]; then
        requirements_content=$(cat "$requirements_file")
    fi

    if [ -f "$design_file" ]; then
        design_content=$(cat "$design_file")
    fi

    if [ -f "$tasks_file" ]; then
        tasks_content=$(cat "$tasks_file")
    fi

    # Extract introduction/overview from requirements
    # (typically the first section before "## Requirements")
    feature_description=$(echo "$requirements_content" | sed -n '1,/^## Requirements/p' | sed '$d')

    # Create consolidated feature file
    cat > "$output_file" <<EOF
# ${component}

**Created:** $(date +%Y-%m-%d)
**Status:** Migrated from .kiro
**Type:** Feature Request
**Source:** .kiro/specs/${component}/

---

## Feature Description

${feature_description}

## Requirements & User Stories

${requirements_content}

## Architecture & Design

${design_content}

## Implementation Tasks & Acceptance Criteria

${tasks_content}

---

**Migration Notes:**
- Consolidated from .kiro/specs/${component}/
- Original files: requirements.md, design.md, tasks.md
- Ready for sage workflow processing
EOF

    echo "   ✓ Created: $output_file"
    migrated=$((migrated + 1))
    echo ""
done

echo "════════════════════════════════════════════════════════════════"
echo "                    Migration Summary"
echo "════════════════════════════════════════════════════════════════"
echo ""
echo "Total components found:    $total"
echo "Successfully migrated:     $migrated"
echo "Skipped (already exist):   $skipped"
echo ""
echo "📂 Output directory: docs/features/"
echo ""

if [ $migrated -gt 0 ]; then
    echo "✅ Migration complete!"
    echo ""
    echo "🔍 Next Steps:"
    echo ""
    echo "1. Review generated feature files:"
    echo "   ls -la docs/features/"
    echo ""
    echo "2. Run sage workflow to process features:"
    echo "   /sage.specify     # Generate specs from features"
    echo "   /sage.plan        # Generate implementation plans"
    echo "   /sage.tasks       # Generate task breakdowns"
    echo ""
    echo "3. Review generated artifacts:"
    echo "   ls -la docs/specs/"
    echo ""
    echo "4. Once verified, remove .kiro directory:"
    echo "   rm -rf .kiro/"
    echo ""
else
    echo "ℹ️  No new features to migrate."
    echo ""
fi

echo "════════════════════════════════════════════════════════════════"
