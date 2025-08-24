# Makefile for FEAST.jl documentation

# Variables
PYTHON := python3
PIP := pip3
MKDOCS := mkdocs
SERVE_PORT := 8000

# Default target
.PHONY: help
help:
	@echo "FEAST.jl Documentation Build System"
	@echo "====================================="
	@echo ""
	@echo "Available targets:"
	@echo "  install     - Install documentation dependencies"
	@echo "  serve       - Serve documentation locally"
	@echo "  build       - Build static documentation"
	@echo "  deploy      - Deploy to GitHub Pages"
	@echo "  clean       - Clean build artifacts"
	@echo "  test        - Test documentation links and formatting"
	@echo "  spell       - Run spell check (requires aspell)"
	@echo "  pdf         - Generate PDF documentation"
	@echo ""

# Install dependencies
.PHONY: install
install:
	@echo "Installing documentation dependencies..."
	$(PIP) install --user mkdocs
	$(PIP) install --user mkdocs-material
	$(PIP) install --user mkdocs-git-revision-date-localized-plugin
	$(PIP) install --user mkdocs-minify-plugin
	$(PIP) install --user pymdown-extensions
	$(PIP) install --user mkdocs-awesome-pages-plugin
	@echo "Dependencies installed successfully!"

# Serve documentation locally
.PHONY: serve
serve:
	@echo "Starting local documentation server..."
	@echo "Documentation will be available at http://localhost:$(SERVE_PORT)"
	$(MKDOCS) serve --dev-addr localhost:$(SERVE_PORT)

# Build static documentation
.PHONY: build
build: clean
	@echo "Building static documentation..."
	$(MKDOCS) build --clean --verbose
	@echo "Documentation built successfully in site/"

# Deploy to GitHub Pages
.PHONY: deploy
deploy:
	@echo "Deploying documentation to GitHub Pages..."
	$(MKDOCS) gh-deploy --clean --verbose
	@echo "Documentation deployed successfully!"

# Clean build artifacts
.PHONY: clean
clean:
	@echo "Cleaning build artifacts..."
	rm -rf site/
	rm -rf .mkdocs_cache/
	find . -name "*.pyc" -delete
	find . -name "__pycache__" -type d -exec rm -rf {} +
	@echo "Clean completed!"

# Test documentation
.PHONY: test
test:
	@echo "Testing documentation..."
	@echo "Checking for broken internal links..."
	@find . -name "*.md" -exec grep -l "\[.*\](.*\.md)" {} \; | while read file; do \
		echo "Checking links in $$file"; \
		grep -o "\[.*\](.*\.md)" "$$file" | sed 's/.*](//' | sed 's/).*//' | while read link; do \
			if [ ! -f "$$link" ] && [ ! -f "$$(dirname $$file)/$$link" ]; then \
				echo "  Broken link: $$link in $$file"; \
			fi; \
		done; \
	done
	@echo "Checking for TODO items..."
	@grep -r "TODO\|FIXME\|XXX" *.md || echo "  No TODO items found"
	@echo "Testing completed!"

# Spell check (requires aspell)
.PHONY: spell
spell:
	@echo "Running spell check..."
	@which aspell > /dev/null || (echo "aspell not found. Install with: apt-get install aspell aspell-en" && exit 1)
	@find . -name "*.md" -exec aspell check --mode=markdown --lang=en {} \;
	@echo "Spell check completed!"

# Generate PDF documentation
.PHONY: pdf
pdf:
	@echo "Generating PDF documentation..."
	@which pandoc > /dev/null || (echo "pandoc not found. Install from https://pandoc.org/" && exit 1)
	@echo "Combining all markdown files..."
	@cat index.md getting_started.md api_reference.md examples.md \
	     matrix_free_interface.md performance.md custom_contours.md \
	     > feast_documentation_combined.md
	pandoc feast_documentation_combined.md \
	       --pdf-engine=xelatex \
	       --toc \
	       --toc-depth=3 \
	       --number-sections \
	       --highlight-style=github \
	       --geometry margin=1in \
	       --variable mainfont="Arial" \
	       --variable monofont="Courier New" \
	       --metadata title="FEAST.jl Documentation" \
	       --metadata author="FEAST.jl Contributors" \
	       --metadata date="$$(date +%Y-%m-%d)" \
	       -o FEAST.jl_Documentation.pdf
	@rm feast_documentation_combined.md
	@echo "PDF generated: FEAST.jl_Documentation.pdf"

# Watch for changes and rebuild
.PHONY: watch
watch:
	@echo "Watching for changes..."
	@which inotifywait > /dev/null || (echo "inotify-tools not found. Install with: apt-get install inotify-tools" && exit 1)
	@while inotifywait -r -e modify,create,delete *.md mkdocs.yml; do \
		echo "Changes detected, rebuilding..."; \
		$(MKDOCS) build --quiet; \
		echo "Rebuild completed at $$(date)"; \
	done

# Validate YAML configuration
.PHONY: validate-config
validate-config:
	@echo "Validating mkdocs.yml configuration..."
	@$(PYTHON) -c "import yaml; yaml.safe_load(open('mkdocs.yml'))" && echo "✅ mkdocs.yml is valid YAML"
	@$(MKDOCS) build --dry-run > /dev/null && echo "✅ MkDocs configuration is valid"

# Check dependencies
.PHONY: check-deps
check-deps:
	@echo "Checking documentation dependencies..."
	@$(PYTHON) --version
	@$(MKDOCS) --version || echo "❌ MkDocs not found - run 'make install'"
	@$(PYTHON) -c "import material" && echo "✅ mkdocs-material installed" || echo "❌ mkdocs-material not found"

# Development setup
.PHONY: dev-setup
dev-setup: install validate-config
	@echo "Setting up development environment..."
	@echo "✅ Development setup complete!"
	@echo "Run 'make serve' to start local server"

# Production build with optimization
.PHONY: prod-build
prod-build: clean validate-config
	@echo "Building production documentation..."
	@echo "Optimizing images..."
	@find . -name "*.png" -exec optipng -quiet {} \; 2>/dev/null || echo "optipng not found, skipping image optimization"
	$(MKDOCS) build --clean --verbose --strict
	@echo "Minifying HTML/CSS/JS..."
	@find site/ -name "*.html" -exec htmlmin {} {} \; 2>/dev/null || echo "htmlmin not found, skipping minification"
	@echo "✅ Production build complete!"

# Check for updates
.PHONY: update-deps
update-deps:
	@echo "Updating documentation dependencies..."
	$(PIP) install --user --upgrade mkdocs mkdocs-material pymdown-extensions
	@echo "✅ Dependencies updated!"

# Generate table of contents
.PHONY: toc
toc:
	@echo "Generating table of contents..."
	@$(PYTHON) -c "
import re
import glob

def extract_headings(filename):
    with open(filename, 'r') as f:
        content = f.read()
    
    headings = re.findall(r'^(#{1,6})\s+(.+)$', content, re.MULTILINE)
    return [(len(h[0]), h[1]) for h in headings]

files = ['index.md', 'getting_started.md', 'api_reference.md', 'examples.md', 
         'matrix_free_interface.md', 'performance.md', 'custom_contours.md']

print('# FEAST.jl Documentation - Table of Contents')
print()

for filename in files:
    try:
        headings = extract_headings(filename)
        if headings:
            print(f'## {filename}')
            for level, title in headings:
                indent = '  ' * (level - 1)
                link = title.lower().replace(' ', '-').replace('(', '').replace(')', '')
                print(f'{indent}- [{title}]({filename}#{link})')
            print()
    except FileNotFoundError:
        print(f'Warning: {filename} not found')
"
	@echo "✅ Table of contents generated!"

# Show documentation statistics
.PHONY: stats
stats:
	@echo "Documentation Statistics"
	@echo "======================="
	@echo "Files: $$(find . -name '*.md' | wc -l)"
	@echo "Total lines: $$(cat *.md | wc -l)"
	@echo "Total words: $$(cat *.md | wc -w)"
	@echo "Total characters: $$(cat *.md | wc -c)"
	@echo ""
	@echo "File breakdown:"
	@for file in *.md; do \
		lines=$$(wc -l < "$$file"); \
		words=$$(wc -w < "$$file"); \
		echo "  $$file: $$lines lines, $$words words"; \
	done

# All-in-one development command
.PHONY: dev
dev: dev-setup serve

# All-in-one production command  
.PHONY: production
production: prod-build deploy

.DEFAULT_GOAL := help