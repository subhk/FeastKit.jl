# FEAST.jl Documentation

This directory contains the complete documentation for FEAST.jl, designed to be deployed as a modern, interactive web documentation site.

## 📚 Documentation Structure

```
docs/
├── index.md                    # Main landing page
├── getting_started.md          # Tutorial for new users
├── api_reference.md           # Complete API documentation
├── examples.md               # Comprehensive examples
├── matrix_free_interface.md  # Matrix-free methods guide
├── performance.md            # Performance optimization
├── custom_contours.md        # Advanced contour integration
├── mkdocs.yml               # MkDocs configuration
├── _config.yml              # GitHub Pages configuration
├── Makefile                 # Build automation
├── stylesheets/             # Custom CSS styling
│   ├── extra.css           # Main custom styles
│   └── julia-highlighting.css # Julia syntax highlighting
└── javascripts/            # Custom JavaScript
    ├── mathjax.js          # Mathematical notation
    └── julia-highlighting.js # Enhanced code highlighting
```

## 🚀 Quick Start

### View Documentation Locally

1. **Install dependencies**:
   ```bash
   cd docs/
   make install
   ```

2. **Serve locally**:
   ```bash
   make serve
   ```

3. **Open your browser** to `http://localhost:8000`

### Build Static Site

```bash
make build
```

The static site will be generated in the `site/` directory.

## 🔧 Build Options

### Using MkDocs (Recommended)

```bash
# Install MkDocs and dependencies
pip install mkdocs mkdocs-material pymdown-extensions

# Serve documentation locally
mkdocs serve

# Build static documentation  
mkdocs build

# Deploy to GitHub Pages
mkdocs gh-deploy
```

### Using GitHub Pages (Alternative)

1. **Enable GitHub Pages** in repository settings
2. **Choose source**: `docs/` folder from `main` branch  
3. **Configure** `_config.yml` with your repository details
4. **Push changes** - documentation will auto-deploy

### Using Make (Automated)

```bash
# Full development setup
make dev-setup

# Start development server
make serve

# Production build with optimizations
make prod-build

# Deploy to GitHub Pages
make deploy
```

## 📖 Documentation Features

### 🎨 Modern Design
- **Material Design** theme with FEAST.jl branding
- **Dark/light mode** toggle
- **Responsive** design for all devices
- **Fast search** with instant results

### 💻 Code Features
- **Julia syntax highlighting** with FEAST-specific functions
- **Copy-to-clipboard** for all code blocks
- **Interactive examples** with expected outputs
- **Collapsible sections** for long code blocks

### 📊 Mathematical Notation
- **MathJax** rendering for equations
- **Custom macros** for FEAST-specific notation
- **Equation numbering** for references
- **Interactive math** with hover effects

### 🔍 Navigation
- **Multi-level navigation** with expand/collapse
- **Table of contents** for each page
- **Cross-references** between sections
- **Search integration** across all content

## 📝 Content Overview

| Page | Purpose | Target Audience |
|------|---------|----------------|
| **Home** | Overview, quick start | All users |
| **Getting Started** | Step-by-step tutorial | New users |
| **Examples** | Working code examples | All users |
| **API Reference** | Complete function docs | Developers |
| **Matrix-Free** | Large-scale problems | Advanced users |
| **Performance** | Optimization guide | Performance-focused users |
| **Custom Contours** | Advanced techniques | Experts |

## ✨ Key Features Documented

### Core FEAST Algorithm
- Standard and generalized eigenvalue problems
- Real symmetric and complex Hermitian matrices
- Non-Hermitian general eigenvalue problems
- Polynomial eigenvalue problems

### Matrix-Free Interface
- `LinearOperator` and `MatrixVecFunction` types
- Iterative solver integration (GMRES, CG, BiCGSTAB)
- Custom linear solver callbacks
- Memory-efficient large-scale computations

### Advanced Contour Integration  
- Gauss-Legendre, Trapezoidal, and Zolotarev methods
- Custom contour shapes (rectangular, star, lens)
- Adaptive contour placement
- Multi-level strategies for complex eigenvalue distributions

### Performance Optimization
- Memory usage analysis and optimization
- Parallel computing (shared and distributed memory)
- Problem-specific optimizations
- Benchmarking and profiling tools

### Real-World Applications
- Structural dynamics and vibration analysis
- Quantum mechanics and electronic structure
- Fluid dynamics stability analysis
- PDE eigenvalue problems

## 🛠️ Customization

### Styling
Edit `stylesheets/extra.css` to customize:
- Color scheme and branding
- Typography and spacing
- Component styling
- Responsive behavior

### JavaScript
Modify `javascripts/` files for:
- Enhanced code highlighting
- Interactive features
- Mathematical notation
- Custom functionality

### Configuration
Update `mkdocs.yml` to change:
- Site metadata and navigation
- Plugin configuration
- Theme settings
- Build options

## 📦 Deployment Options

### GitHub Pages (Automatic)
1. Push documentation to `docs/` folder
2. Enable Pages in repository settings
3. Choose `docs/` folder as source
4. Documentation automatically builds and deploys

### GitHub Actions (CI/CD)
```yaml
# .github/workflows/docs.yml
name: Deploy Documentation
on:
  push:
    branches: [ main ]
    paths: [ 'docs/**' ]

jobs:
  deploy:
    runs-on: ubuntu-latest
    steps:
    - uses: actions/checkout@v3
    - uses: actions/setup-python@v4
      with:
        python-version: 3.x
    - run: pip install mkdocs-material
    - run: mkdocs gh-deploy --force
```

### Custom Hosting
1. Run `make build` to generate static site
2. Deploy `site/` directory to your web server
3. Configure server for proper routing

## 🧪 Testing

```bash
# Test all documentation
make test

# Check for broken links
make test | grep "Broken link"

# Spell check (requires aspell)
make spell

# Validate configuration
make validate-config
```

## 📊 Analytics and Metrics

### Built-in Analytics
- **Google Analytics** integration (configure in `mkdocs.yml`)
- **Search analytics** via MkDocs
- **Page view tracking** 

### Documentation Metrics
```bash
# Generate documentation statistics
make stats

# Output:
# Files: 8
# Total lines: 2847
# Total words: 19264
# Total characters: 142851
```

## 🤝 Contributing to Documentation

### Content Guidelines
1. **Clear structure** with logical headings
2. **Working examples** with expected outputs  
3. **Performance tips** where relevant
4. **Cross-references** to related sections
5. **Mathematical notation** for algorithms

### Style Guidelines
1. **Concise explanations** with examples
2. **Code blocks** with proper syntax highlighting
3. **Admonitions** for tips, warnings, notes
4. **Tables** for parameter documentation
5. **Consistent formatting** across all pages

### Adding New Content
1. Create new `.md` file in `docs/`
2. Add to navigation in `mkdocs.yml`
3. Include cross-references from existing pages
4. Test locally with `make serve`
5. Submit pull request

## 🔧 Troubleshooting

### Common Issues

**MkDocs not found**:
```bash
pip install --user mkdocs mkdocs-material
# or
make install
```

**Port already in use**:
```bash
mkdocs serve --dev-addr localhost:8001
# or
make serve SERVE_PORT=8001
```

**Build failures**:
```bash
# Check configuration
make validate-config

# Clean and rebuild
make clean build
```

### Getting Help

- **Documentation Issues**: [GitHub Issues](https://github.com/your-repo/FEAST.jl/issues)
- **MkDocs Help**: [MkDocs Documentation](https://www.mkdocs.org/)
- **Material Theme**: [Material for MkDocs](https://squidfunk.github.io/mkdocs-material/)

---

<div align="center">
  <p><strong>Build beautiful, comprehensive documentation for FEAST.jl</strong></p>
  <p><a href="https://your-domain.github.io/FEAST.jl">View Live Documentation</a></p>
</div>