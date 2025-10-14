#!/usr/bin/env bash
set -e

echo "════════════════════════════════════════════════════════════════"
echo "     Generate Specs from Features (Batch Processing)"
echo "════════════════════════════════════════════════════════════════"
echo ""

# Ensure directories exist
mkdir -p docs/specs
mkdir -p .sage/tickets

# Initialize ticket index
if [ ! -f .sage/tickets/index.json ]; then
    echo '{"version":"1.0","tickets":[]}' > .sage/tickets/index.json
fi

total=0
generated=0
skipped=0

echo "🔄 Processing feature files..."
echo ""

for feature_file in docs/features/*.md; do
    [ -f "$feature_file" ] || continue

    component=$(basename "$feature_file" .md)
    total=$((total + 1))

    echo "Processing: $component"

    # Skip if spec already exists
    if [ -f "docs/specs/$component/spec.md" ]; then
        echo "   ⚠️  Spec already exists, skipping"
        skipped=$((skipped + 1))
        echo ""
        continue
    fi

    # Create component spec directory
    mkdir -p "docs/specs/$component"

    # ===== STEP 1: Extract Metadata =====
    created_date=$(grep "^\*\*Created:\*\*" "$feature_file" | sed 's/\*\*Created:\*\* //' | tr -d '\n')
    [ -z "$created_date" ] && created_date=$(date +%Y-%m-%d)

    # ===== STEP 2: Extract Introduction =====
    introduction=$(sed -n '/^## Feature Description/,/^## Requirements & User Stories/p' "$feature_file" | \
        sed '1d;$d' | \
        sed '/^# Requirements Document/d' | \
        sed '/^## Introduction/d' | \
        sed '/^$/d' | \
        head -20)

    [ -z "$introduction" ] && introduction="Component description to be added."

    # ===== STEP 3: Extract Requirements =====
    requirements_section=$(sed -n '/^## Requirements & User Stories/,/^## Architecture & Design/p' "$feature_file" | sed '1d;$d')

    # Count requirements
    num_requirements=$(echo "$requirements_section" | grep -c "^### Requirement [0-9]" || echo "0")

    # ===== STEP 4: Extract Functional Requirements =====
    functional_reqs=""
    for i in $(seq 1 $num_requirements); do
        # Extract each requirement block
        req_block=$(echo "$requirements_section" | sed -n "/^### Requirement $i/,/^### Requirement $((i+1))/p" | sed '$d')

        # Extract user story
        user_story=$(echo "$req_block" | grep "^\*\*User Story:\*\*" | sed 's/\*\*User Story:\*\* //')

        # Extract SHALL statements
        shall_statements=$(echo "$req_block" | grep "SHALL" | sed 's/^[0-9]*\. WHEN.*THEN the system /- System /' | sed 's/^[0-9]*\. WHEN.*THEN /- /')

        if [ -n "$user_story" ]; then
            functional_reqs="${functional_reqs}
### FR-${i}: $(echo "$user_story" | cut -d',' -f2- | sed 's/^ I want //' | cut -d',' -f1)

**User Story:** $user_story

**Requirements:**
$shall_statements

"
        fi
    done

    # ===== STEP 5: Identify Non-Functional Requirements =====
    nfr_performance=$(echo "$requirements_section" | grep -i "performance\|speed\|latency\|throughput\|efficiency" | head -5 || echo "- Performance requirements to be defined")
    nfr_security=$(echo "$requirements_section" | grep -i "privacy\|security\|protection\|confidential" | head -5 || echo "- Security requirements to be defined")
    nfr_scalability=$(echo "$requirements_section" | grep -i "scalable\|edge\|distributed\|concurrent" | head -5 || echo "- Scalability requirements to be defined")

    # ===== STEP 6: Extract Architecture & Design =====
    architecture_section=$(sed -n '/^## Architecture & Design/,/^## Implementation Tasks/p' "$feature_file" | sed '1d;$d')

    # If architecture section is too large, truncate
    architecture_summary=$(echo "$architecture_section" | head -100)

    # ===== STEP 7: Extract Implementation Tasks =====
    tasks_section=$(sed -n '/^## Implementation Tasks & Acceptance Criteria/,/^---$/p' "$feature_file" | sed '1d;$d')

    # Extract top-level tasks for acceptance criteria
    acceptance_criteria=$(echo "$tasks_section" | grep "^- \[ \]" | head -10 | sed 's/^- \[ \] /- /')

    # ===== STEP 8: Determine Priority =====
    priority="P1"
    impl_status="Planned"

    # P0 (Critical) - Already implemented
    if echo "$component" | grep -qE "lora-finetuning|model-compression|core-ml-diffusion|dspy-toolkit|shared-utilities|efficientai-mlx|development-knowledge"; then
        priority="P0"
        impl_status="Implemented"
    fi

    # P1 (High) - Planned next
    if echo "$component" | grep -qE "multimodal-clip|federated-learning|mlops-integration"; then
        priority="P1"
        impl_status="Planned"
    fi

    # P2 (Future) - Research/experimental
    if echo "$component" | grep -qE "adaptive-diffusion|evolutionary|meta-learning|quantized-model"; then
        priority="P2"
        impl_status="Future"
    fi

    # ===== STEP 9: Generate Spec File =====
    # Capitalize first letter of component name (bash 3.2 compatible)
    component_title=$(echo "$component" | sed 's/-/ /g' | awk '{for(i=1;i<=NF;i++)sub(/./,toupper(substr($i,1,1)),$i)}1' | sed 's/ /-/g')

    cat > "docs/specs/$component/spec.md" <<EOF
# ${component_title} Specification

**Created:** $(date +%Y-%m-%d)
**Source:** docs/features/${component}.md
**Original:** .kiro/specs/${component}/
**Status:** Migrated from .kiro
**Implementation Status:** $impl_status
**Priority:** $priority

---

## 1. Overview

### Purpose

$introduction

### Success Metrics

- Feature implementation complete
- All acceptance criteria met
- Tests passing with adequate coverage
- Performance targets achieved

### Target Users

$(echo "$requirements_section" | grep -o "As a [^,]*" | sort -u | sed 's/As a /- /')

## 2. Functional Requirements

$functional_reqs

## 3. Non-Functional Requirements

### 3.1 Performance

$nfr_performance

### 3.2 Security & Privacy

$nfr_security

### 3.3 Scalability & Reliability

$nfr_scalability

## 4. Architecture & Design

$architecture_summary

### Key Components

- Architecture details available in source feature document
- See: docs/features/${component}.md for complete architecture specification

## 5. Acceptance Criteria

$acceptance_criteria

### Definition of Done

- All functional requirements implemented
- Non-functional requirements validated
- Comprehensive test coverage
- Documentation complete
- Code review approved

## 6. Dependencies

### Technical Dependencies

- MLX framework (Apple Silicon optimization)
- PyTorch with MPS backend
- Python 3.11+
- uv package manager

### Component Dependencies

- shared-utilities (logging, config, benchmarking)
- efficientai-mlx-toolkit (CLI integration)

### External Integrations

- To be identified during implementation planning

---

## Traceability

- **Feature Request:** docs/features/${component}.md
- **Original Spec:** .kiro/specs/${component}/
- **Implementation Status:** $impl_status
- **Epic Ticket:** .sage/tickets/[COMPONENT]-001.md

## Notes

- Migrated from .kiro system on $(date +%Y-%m-%d)
- Ready for /sage.plan (implementation planning)
- Source contains detailed design, interfaces, and task breakdown
EOF

    # ===== STEP 10: Generate Epic Ticket =====
    # Create ticket ID (first 4 letters of component, uppercase)
    component_abbrev=$(echo "$component" | tr '[:lower:]' '[:upper:]' | sed 's/-//g' | cut -c1-4)
    ticket_id="${component_abbrev}-001"

    # Get first sentence of introduction for ticket description
    ticket_desc=$(echo "$introduction" | head -1 | cut -c1-200)

    cat > ".sage/tickets/${ticket_id}.md" <<EOF
# ${ticket_id}: ${component_title} Implementation

**State:** UNPROCESSED
**Priority:** $priority
**Type:** Epic
**Created:** $(date +%Y-%m-%d)
**Implementation Status:** $impl_status

## Description

$ticket_desc

## Acceptance Criteria

$acceptance_criteria

## Dependencies

### Component Dependencies
- shared-utilities
- efficientai-mlx-toolkit

### Technical Dependencies
- MLX framework
- PyTorch MPS
- Python 3.11+

## Context

**Specification:** docs/specs/${component}/spec.md
**Feature Request:** docs/features/${component}.md
**Original Spec:** .kiro/specs/${component}/

## Progress

**Current Phase:** Specification Complete
**Next Step:** Run /sage.plan for implementation planning
**Status:** Ready for planning phase

---

**Migration Notes:**
- Migrated from .kiro system
- Consolidated requirements, design, and tasks
- Ready for sage workflow processing
EOF

    # ===== STEP 11: Update Ticket Index =====
    # Use Python for JSON manipulation
    python3 << PYTHON
import json
from pathlib import Path

index_file = Path(".sage/tickets/index.json")
with open(index_file) as f:
    data = json.load(f)

# Check if ticket already exists
ticket_exists = any(t["id"] == "${ticket_id}" for t in data["tickets"])

if not ticket_exists:
    data["tickets"].append({
        "id": "${ticket_id}",
        "type": "Epic",
        "state": "UNPROCESSED",
        "priority": "${priority}",
        "component": "${component}",
        "spec": "docs/specs/${component}/spec.md",
        "feature": "docs/features/${component}.md",
        "status": "${impl_status}",
        "created": "$(date +%Y-%m-%d)"
    })

    with open(index_file, 'w') as f:
        json.dump(data, f, indent=2)
PYTHON

    echo "   ✓ Generated: docs/specs/$component/spec.md"
    echo "   ✓ Generated: .sage/tickets/${ticket_id}.md"
    echo "   ✓ Updated: .sage/tickets/index.json"
    generated=$((generated + 1))
    echo ""
done

echo "════════════════════════════════════════════════════════════════"
echo "                    Generation Summary"
echo "════════════════════════════════════════════════════════════════"
echo ""
echo "Total feature files:     $total"
echo "Specs generated:         $generated"
echo "Skipped (exist):         $skipped"
echo "Epic tickets created:    $generated"
echo ""
echo "📂 Output directories:"
echo "   Specs:   docs/specs/"
echo "   Tickets: .sage/tickets/"
echo ""

if [ $generated -gt 0 ]; then
    echo "✅ Spec generation complete!"
    echo ""
    echo "🔍 Next Steps:"
    echo ""
    echo "1. Review generated specs:"
    echo "   ls -la docs/specs/*/"
    echo ""
    echo "2. Review epic tickets:"
    echo "   cat .sage/tickets/index.json | python3 -m json.tool"
    echo ""
    echo "3. Continue sage workflow:"
    echo "   /sage.plan      # Generate implementation plans"
    echo "   /sage.tasks     # Generate task breakdowns"
    echo ""
else
    echo "ℹ️  No new specs to generate."
    echo ""
fi

echo "════════════════════════════════════════════════════════════════"
