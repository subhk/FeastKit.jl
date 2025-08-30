# FeastKit.jl Documentation

This directory contains the complete documentation for FeastKit.jl. It is configured to be served directly from the repository via GitHub Pages using the `docs/` folder on the `main` branch (no separate gh-pages branch).

## Documentation Structure

```
docs/
├── index.md                    # Main landing page
├── getting_started.md          # Tutorial for new users
├── api_reference.md           # Complete API documentation
├── examples.md               # Comprehensive examples
├── matrix_free_interface.md  # Matrix-free methods guide
├── performance.md            # Performance optimization
├── custom_contours.md        # Advanced contour integration
├── _config.yml              # GitHub Pages (Jekyll) configuration
└── assets/                  # Optional images/static assets (add as needed)
```

## Quick Start

### Publish on GitHub Pages (from main/docs)

1) In your GitHub repository: Settings → Pages  
2) Source: Deploy from a branch → Branch: `main`, Folder: `/docs`  
3) Save. The site will be available at `https://<username>.github.io/<repo>/`

## Build Options

### Configure site metadata

Edit `docs/_config.yml` and set:
- `url`: `https://<username>.github.io`
- `baseurl`: `/<repo>` (e.g. `/FeastKit.jl`)
- `repository`: `<username>/<repo>`

## Documentation Features

### What you get
- Clear landing page, getting-started guide, “Zero to FeastKit” walkthrough
- API overview and examples
- Advanced topic stubs you can expand over time

## Content Overview

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

## Customization

### Configuration
Update `_config.yml` to change:
- Site metadata and repository links
- Theme and plugin settings supported by GitHub Pages

## Deployment Options

### GitHub Pages (Automatic)
1. Push documentation to `docs/` folder
2. Enable Pages in repository settings
3. Choose `docs/` folder as source
4. Documentation automatically builds and deploys

### Custom Hosting
If you prefer to host elsewhere, any static-site host that supports Jekyll can serve `docs/` directly.

## Testing

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

## Analytics and Metrics

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

## Contributing to Documentation

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

## Troubleshooting

### Common Issues

**Pages not updating**:
- Verify GitHub Pages is set to `main` + `/docs`
- Check `_config.yml` `url` and `baseurl` are correct
- Wait a few minutes; Pages builds are asynchronous

### Getting Help

- **Documentation Issues**: [GitHub Issues](https://github.com/your-repo/FeastKit.jl/issues)
- **MkDocs Help**: [MkDocs Documentation](https://www.mkdocs.org/)
- **Material Theme**: [Material for MkDocs](https://squidfunk.github.io/mkdocs-material/)

---

<div align="center">
  <p><strong>Build beautiful, comprehensive documentation for FeastKit.jl</strong></p>
  <p><a href="https://subhk.github.io/FeastKit.jl">View Live Documentation</a></p>
</div>
