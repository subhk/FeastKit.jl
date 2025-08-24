// Enhanced Julia syntax highlighting for FEAST.jl documentation

document.addEventListener('DOMContentLoaded', function() {
    // FEAST-specific function names to highlight
    const feastFunctions = [
        'feast', 'feast_general', 'feast_srci!', 'feast_hrci!', 'feast_grci!',
        'feast_matfree_srci!', 'feast_matfree_grci!', 'feast_polynomial',
        'feast_contour', 'feast_gcontour', 'feast_customcontour', 'feast_contour_expert',
        'feast_contour_custom_weights!', 'feast_rational_expert',
        'LinearOperator', 'MatrixVecFunction', 'create_iterative_solver',
        'feastinit!', 'feastdefault!', 'feast_set_defaults!',
        'eigvals_feast', 'eigen_feast', 'feast_summary', 'feast_validate_interval',
        'allocate_matfree_workspace'
    ];
    
    // FEAST-specific types to highlight
    const feastTypes = [
        'FeastResult', 'FeastParameters', 'FeastWorkspaceReal', 'FeastWorkspaceComplex',
        'FeastContour', 'MatrixFreeOperator', 'ParallelFeastState', 'MPIFeastState'
    ];
    
    // Process all code blocks
    const codeBlocks = document.querySelectorAll('code.language-julia, pre code');
    
    codeBlocks.forEach(block => {
        let html = block.innerHTML;
        
        // Highlight FEAST functions
        feastFunctions.forEach(func => {
            const regex = new RegExp(`\\b(${func.replace('!', '\\!')})\\b`, 'g');
            html = html.replace(regex, '<span class="nf feast">$1</span>');
        });
        
        // Highlight FEAST types
        feastTypes.forEach(type => {
            const regex = new RegExp(`\\b(${type})\\b`, 'g');
            html = html.replace(regex, '<span class="nc feast-type">$1</span>');
        });
        
        block.innerHTML = html;
    });
    
    // Add copy-to-clipboard functionality
    addCopyButtons();
    
    // Add line numbers to code blocks
    addLineNumbers();
    
    // Add collapsible sections for long code blocks
    addCollapsibleSections();
});

function addCopyButtons() {
    const codeBlocks = document.querySelectorAll('pre code');
    
    codeBlocks.forEach(block => {
        const pre = block.parentElement;
        if (pre.querySelector('.copy-button')) return; // Already has button
        
        const button = document.createElement('button');
        button.className = 'copy-button md-clipboard';
        button.innerHTML = '📋';
        button.title = 'Copy code';
        
        button.addEventListener('click', () => {
            const text = block.textContent;
            navigator.clipboard.writeText(text).then(() => {
                button.innerHTML = '✅';
                button.title = 'Copied!';
                setTimeout(() => {
                    button.innerHTML = '📋';
                    button.title = 'Copy code';
                }, 2000);
            });
        });
        
        pre.style.position = 'relative';
        pre.appendChild(button);
    });
}

function addLineNumbers() {
    const codeBlocks = document.querySelectorAll('pre code.language-julia');
    
    codeBlocks.forEach(block => {
        const lines = block.textContent.split('\n');
        if (lines.length > 5) { // Only add line numbers for longer blocks
            const pre = block.parentElement;
            if (pre.classList.contains('line-numbers')) return;
            
            pre.classList.add('line-numbers');
            
            let html = '';
            lines.forEach((line, index) => {
                if (index < lines.length - 1 || line.trim()) { // Skip empty last line
                    html += `<span class="line-number">${index + 1}</span>${line}\n`;
                }
            });
            
            block.innerHTML = highlightSyntax(html);
        }
    });
}

function addCollapsibleSections() {
    const codeBlocks = document.querySelectorAll('pre code');
    
    codeBlocks.forEach(block => {
        const lines = block.textContent.split('\n');
        if (lines.length > 20) { // Make very long code blocks collapsible
            const pre = block.parentElement;
            const container = document.createElement('details');
            const summary = document.createElement('summary');
            
            container.className = 'code-collapsible';
            summary.textContent = `Show code (${lines.length} lines)`;
            summary.className = 'code-summary';
            
            const parent = pre.parentElement;
            parent.insertBefore(container, pre);
            container.appendChild(summary);
            container.appendChild(pre);
            
            // Start collapsed for very long blocks
            if (lines.length > 50) {
                container.open = false;
            } else {
                container.open = true;
            }
        }
    });
}

function highlightSyntax(html) {
    // Basic Julia syntax highlighting
    html = html.replace(/\b(function|end|if|else|elseif|for|while|try|catch|finally|return|break|continue|struct|mutable|abstract|primitive|type|const|global|local|let|do|begin|quote|macro|module|baremodule|using|import|export|public)\b/g, '<span class="k">$1</span>');
    html = html.replace(/\b(true|false|nothing|missing|undef)\b/g, '<span class="kc">$1</span>');
    html = html.replace(/\b(\d+\.?\d*([eE][+-]?\d+)?)\b/g, '<span class="m">$1</span>');
    html = html.replace(/"([^"\\]|\\.)*"/g, '<span class="s">"$1"</span>');
    html = html.replace(/#.*/g, '<span class="c1">$&</span>');
    
    return html;
}

// Add keyboard shortcuts for code blocks
document.addEventListener('keydown', function(e) {
    // Ctrl/Cmd + Click to copy code block
    if ((e.ctrlKey || e.metaKey) && e.target.tagName === 'CODE') {
        const text = e.target.textContent;
        navigator.clipboard.writeText(text);
        
        // Visual feedback
        e.target.style.background = 'rgba(25, 118, 210, 0.2)';
        setTimeout(() => {
            e.target.style.background = '';
        }, 300);
    }
});

// Add interactive REPL simulation
function addREPLSimulation() {
    const replBlocks = document.querySelectorAll('.language-julia-repl');
    
    replBlocks.forEach(block => {
        const container = document.createElement('div');
        container.className = 'repl-container';
        
        const header = document.createElement('div');
        header.className = 'repl-header';
        header.innerHTML = '<span class="repl-title">Julia REPL</span><span class="repl-version">v1.9.0</span>';
        
        const pre = block.parentElement;
        pre.parentElement.insertBefore(container, pre);
        container.appendChild(header);
        container.appendChild(pre);
        
        // Add interactive features
        makeREPLInteractive(block);
    });
}

function makeREPLInteractive(block) {
    const lines = block.textContent.split('\n');
    let html = '';
    
    lines.forEach(line => {
        if (line.startsWith('julia>')) {
            html += `<div class="repl-input">${line}</div>`;
        } else if (line.startsWith('pkg>')) {
            html += `<div class="repl-pkg">${line}</div>`;
        } else if (line.startsWith('help?>')) {
            html += `<div class="repl-help">${line}</div>`;
        } else if (line.startsWith('shell>')) {
            html += `<div class="repl-shell">${line}</div>`;
        } else if (line.trim()) {
            html += `<div class="repl-output">${line}</div>`;
        }
    });
    
    block.innerHTML = html;
}

// Initialize enhanced features
document.addEventListener('DOMContentLoaded', function() {
    addREPLSimulation();
    
    // Add expand/collapse all button for long pages
    const longCodeBlocks = document.querySelectorAll('.code-collapsible');
    if (longCodeBlocks.length > 3) {
        addExpandCollapseAllButton();
    }
});

function addExpandCollapseAllButton() {
    const button = document.createElement('button');
    button.className = 'expand-collapse-all md-button';
    button.textContent = 'Expand All Code';
    button.style.position = 'fixed';
    button.style.bottom = '20px';
    button.style.right = '20px';
    button.style.zIndex = '1000';
    
    let expanded = false;
    
    button.addEventListener('click', () => {
        const collapsibles = document.querySelectorAll('.code-collapsible');
        expanded = !expanded;
        
        collapsibles.forEach(details => {
            details.open = expanded;
        });
        
        button.textContent = expanded ? 'Collapse All Code' : 'Expand All Code';
    });
    
    document.body.appendChild(button);
}