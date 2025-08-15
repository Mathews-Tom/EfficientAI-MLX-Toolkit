#!/bin/bash
# Knowledge Base Setup Script
# Wrapper script for easy knowledge base management

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Script directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
KB_ROOT="$(dirname "$SCRIPT_DIR")"

# Helper functions
print_header() {
    echo -e "${BLUE}=== $1 ===${NC}"
}

print_success() {
    echo -e "${GREEN}✅ $1${NC}"
}

print_warning() {
    echo -e "${YELLOW}⚠️  $1${NC}"
}

print_error() {
    echo -e "${RED}❌ $1${NC}"
}

check_requirements() {
    print_header "Checking Requirements"
    
    # Check Python
    if ! command -v python3 &> /dev/null; then
        print_error "Python 3 is required but not installed"
        exit 1
    fi
    
    python_version=$(python3 --version | cut -d' ' -f2)
    print_success "Python $python_version found"
    
    # Check uv
    if ! command -v uv &> /dev/null; then
        print_warning "uv package manager not found"
        echo "Installing uv..."
        curl -LsSf https://astral.sh/uv/install.sh | sh
        source ~/.bashrc 2>/dev/null || source ~/.zshrc 2>/dev/null || true
        
        if ! command -v uv &> /dev/null; then
            print_error "Failed to install uv. Please install manually: https://docs.astral.sh/uv/"
            exit 1
        fi
    fi
    
    print_success "uv package manager found"
}

init_knowledge_base() {
    print_header "Initializing Knowledge Base"
    
    cd "$KB_ROOT"
    
    if [ -d ".meta" ] && [ "$1" != "--force" ]; then
        print_warning "Knowledge base already exists. Use --force to reinitialize"
        return 0
    fi
    
    uv run python "$SCRIPT_DIR/init_knowledge_base.py" . "$@"
    
    if [ $? -eq 0 ]; then
        print_success "Knowledge base initialized successfully"
    else
        print_error "Failed to initialize knowledge base"
        exit 1
    fi
}

run_maintenance() {
    print_header "Running Maintenance"
    
    cd "$KB_ROOT"
    
    if [ ! -d ".meta" ]; then
        print_error "No knowledge base found. Run 'init' first"
        exit 1
    fi
    
    uv run python "$SCRIPT_DIR/maintenance.py" --kb-path . "$@"
}

configure_kb() {
    print_header "Configuring Knowledge Base"
    
    cd "$KB_ROOT"
    
    if [ ! -d ".meta" ]; then
        print_error "No knowledge base found. Run 'init' first"
        exit 1
    fi
    
    uv run python "$SCRIPT_DIR/configure_knowledge_base.py" --kb-path . "$@"
}

migrate_docs() {
    print_header "Migrating Documentation"
    
    cd "$KB_ROOT"
    
    if [ ! -d ".meta" ]; then
        print_error "No knowledge base found. Run 'init' first"
        exit 1
    fi
    
    uv run python "$SCRIPT_DIR/migrate_documentation.py" --kb-path . "$@"
}

run_tests() {
    print_header "Running Tests"
    
    cd "$KB_ROOT"
    
    if [ ! -d "tests" ]; then
        print_warning "No tests directory found"
        return 0
    fi
    
    uv run python -m pytest tests/ -v
    
    if [ $? -eq 0 ]; then
        print_success "All tests passed"
    else
        print_error "Some tests failed"
        exit 1
    fi
}

show_status() {
    print_header "Knowledge Base Status"
    
    cd "$KB_ROOT"
    
    if [ ! -d ".meta" ]; then
        print_error "No knowledge base found. Run 'init' first"
        exit 1
    fi
    
    # Show basic info
    echo "📍 Location: $KB_ROOT"
    echo "📊 Statistics:"
    uv run python -m kb stats 2>/dev/null || echo "   Unable to get statistics"
    
    echo ""
    echo "🔧 Maintenance Status:"
    uv run python "$SCRIPT_DIR/maintenance.py" --kb-path . status
}

show_help() {
    cat << EOF
Knowledge Base Setup Script

Usage: $0 <command> [options]

Commands:
    init [--force] [--minimal]     Initialize new knowledge base
    configure <subcommand>         Configure knowledge base settings
    migrate <subcommand>           Migrate existing documentation
    maintenance [subcommand]       Run maintenance tasks
    test                          Run test suite
    status                        Show knowledge base status
    help                          Show this help message

Examples:
    $0 init                       # Initialize new knowledge base
    $0 init --force               # Reinitialize existing knowledge base
    $0 configure show             # Show current configuration
    $0 configure add-category ml  # Add new category
    $0 migrate file doc.md perf   # Migrate single file to performance category
    $0 maintenance full           # Run full maintenance
    $0 test                       # Run all tests
    $0 status                     # Show status

For detailed help on subcommands:
    $0 configure --help
    $0 migrate --help
    $0 maintenance --help

EOF
}

# Main script logic
case "$1" in
    "init")
        check_requirements
        shift
        init_knowledge_base "$@"
        ;;
    "configure")
        shift
        configure_kb "$@"
        ;;
    "migrate")
        shift
        migrate_docs "$@"
        ;;
    "maintenance")
        shift
        run_maintenance "$@"
        ;;
    "test")
        run_tests
        ;;
    "status")
        show_status
        ;;
    "help"|"--help"|"-h")
        show_help
        ;;
    "")
        print_error "No command specified"
        show_help
        exit 1
        ;;
    *)
        print_error "Unknown command: $1"
        show_help
        exit 1
        ;;
esac