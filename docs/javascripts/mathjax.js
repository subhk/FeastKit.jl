// MathJax configuration for FEAST.jl documentation

window.MathJax = {
  tex: {
    inlineMath: [["\\(", "\\)"]],
    displayMath: [["\\[", "\\]"]],
    processEscapes: true,
    processEnvironments: true,
    packages: {
      '[+]': ['ams', 'physics', 'mathtools', 'cases']
    },
    macros: {
      // Common mathematical macros for eigenvalue problems
      "R": "\\mathbb{R}",
      "C": "\\mathbb{C}",
      "N": "\\mathbb{N}",
      "Z": "\\mathbb{Z}",
      "Q": "\\mathbb{Q}",
      
      // Linear algebra
      "norm": ["\\left\\|#1\\right\\|", 1],
      "abs": ["\\left|#1\\right|", 1],
      "inner": ["\\langle #1, #2 \\rangle", 2],
      "tr": "\\operatorname{tr}",
      "rank": "\\operatorname{rank}",
      "span": "\\operatorname{span}",
      "null": "\\operatorname{null}",
      "range": "\\operatorname{range}",
      
      // FEAST-specific notation
      "contour": "\\Gamma",
      "zne": "z_e",
      "wne": "w_e",
      "feast": "\\textsc{Feast}",
      "rci": "\\textsc{RCI}",
      
      // Eigenvalue problems
      "eigenval": "\\lambda",
      "eigenvec": "\\mathbf{x}",
      "eigenspace": "\\mathcal{E}",
      "spectrum": "\\sigma",
      
      // Matrices
      "mat": ["\\mathbf{#1}", 1],
      "vec": ["\\mathbf{#1}", 1],
      "bA": "\\mathbf{A}",
      "bB": "\\mathbf{B}",
      "bx": "\\mathbf{x}",
      "by": "\\mathbf{y}",
      "bz": "\\mathbf{z}",
      
      // Complex analysis
      "res": "\\operatorname{Res}",
      "conj": ["\\overline{#1}", 1],
      "real": "\\operatorname{Re}",
      "imag": "\\operatorname{Im}",
      
      // Differential operators
      "grad": "\\nabla",
      "div": "\\nabla \\cdot",
      "curl": "\\nabla \\times",
      "laplacian": "\\nabla^2",
      
      // Function spaces
      "Lp": ["L^{#1}", 1],
      "Sobolev": ["H^{#1}", 1],
      "Ck": ["C^{#1}", 1],
      
      // Algorithm notation
      "bigO": ["\\mathcal{O}\\left(#1\\right)", 1],
      "order": ["\\mathcal{O}\\left(#1\\right)", 1],
      
      // Common functions
      "sinc": "\\operatorname{sinc}",
      "sign": "\\operatorname{sign}",
      "diag": "\\operatorname{diag}",
      
      // Bold Greek letters
      "blambda": "\\boldsymbol{\\lambda}",
      "bmu": "\\boldsymbol{\\mu}",
      "bsigma": "\\boldsymbol{\\sigma}",
      "btau": "\\boldsymbol{\\tau}",
      "btheta": "\\boldsymbol{\\theta}",
      "bphi": "\\boldsymbol{\\phi}",
      "bpsi": "\\boldsymbol{\\psi}",
      "bomega": "\\boldsymbol{\\omega}",
      
      // Numerical methods
      "tol": "\\varepsilon",
      "eps": "\\varepsilon",
      "residual": "\\mathbf{r}",
      "iterate": ["#1^{(#2)}", 2],
      
      // Contour integration specific
      "contourintegral": "\\oint_{\\contour}",
      "momentmatrix": "\\mathbf{S}",
      "projector": "\\mathbf{P}",
      
      // Units and constants
      "Hz": "\\,\\text{Hz}",
      "rad": "\\,\\text{rad}",
      "deg": "\\,\\text{deg}",
      "Re": "\\,\\text{Re}",
      
      // Special cases for FEAST algorithm
      "feastmoment": ["S_k", 0],
      "feastweight": ["w_e", 0],
      "feastnode": ["z_e", 0]
    }
  },
  options: {
    ignoreHtmlClass: "tex2jax_ignore",
    processHtmlClass: "tex2jax_process"
  },
  svg: {
    fontCache: 'global',
    displayAlign: 'left',
    displayIndent: '2em'
  },
  loader: {
    load: ['[tex]/ams', '[tex]/physics', '[tex]/mathtools', '[tex]/cases']
  }
};

// Add custom styling for mathematical expressions in FEAST context
document.addEventListener('DOMContentLoaded', function() {
  // Style eigenvalue equations
  const eigenvalueEquations = document.querySelectorAll('.MJX-TEX');
  eigenvalueEquations.forEach(eq => {
    if (eq.textContent.includes('\\lambda') || eq.textContent.includes('eigenval')) {
      eq.style.color = '#1976D2';
    }
  });
  
  // Add hover effects for mathematical expressions
  document.addEventListener('click', function(e) {
    if (e.target.closest('.MathJax')) {
      const mathElement = e.target.closest('.MathJax');
      mathElement.style.background = 'rgba(25, 118, 210, 0.1)';
      setTimeout(() => {
        mathElement.style.background = '';
      }, 1000);
    }
  });
});

// Custom renderer for algorithm pseudocode
function renderAlgorithm(element) {
  const algorithmText = element.textContent;
  const lines = algorithmText.split('\n').filter(line => line.trim());
  
  let html = '<div class="algorithm-box">';
  html += '<div class="algorithm-title">Algorithm</div>';
  html += '<div class="algorithm-content">';
  
  lines.forEach((line, index) => {
    const indent = (line.match(/^\s*/) || [''])[0].length;
    const cleanLine = line.trim();
    
    html += `<div class="algorithm-line" style="margin-left: ${indent * 20}px;">`;
    
    if (cleanLine.startsWith('Input:') || cleanLine.startsWith('Output:')) {
      html += `<strong>${cleanLine}</strong>`;
    } else if (cleanLine.startsWith('for ') || cleanLine.startsWith('while ') || 
               cleanLine.startsWith('if ') || cleanLine.startsWith('else')) {
      html += `<span class="algorithm-keyword">${cleanLine}</span>`;
    } else {
      html += cleanLine;
    }
    
    html += '</div>';
  });
  
  html += '</div></div>';
  element.innerHTML = html;
}

// Initialize algorithm rendering
document.addEventListener('DOMContentLoaded', function() {
  document.querySelectorAll('.algorithm').forEach(renderAlgorithm);
});

// Add equation numbering
let equationNumber = 1;
document.addEventListener('DOMContentLoaded', function() {
  // Number display equations
  document.querySelectorAll('.MathJax_Display').forEach(display => {
    if (!display.querySelector('.equation-number')) {
      const numberSpan = document.createElement('span');
      numberSpan.className = 'equation-number';
      numberSpan.textContent = `(${equationNumber++})`;
      numberSpan.style.float = 'right';
      numberSpan.style.marginTop = '0.5em';
      numberSpan.style.color = '#666';
      display.appendChild(numberSpan);
    }
  });
});