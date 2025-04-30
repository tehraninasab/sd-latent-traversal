// Wait for the DOM to be fully loaded
document.addEventListener('DOMContentLoaded', function() {
    // Add smooth scrolling for navigation links
    addSmoothScrolling();
    
    // Initialize section highlighting in navigation
    initSectionHighlighting();
    
    // Add hover effects to figures
    enhanceFigures();
});

/**
 * Adds smooth scrolling behavior to navigation links
 */
function addSmoothScrolling() {
    const navLinks = document.querySelectorAll('.navigation a');
    
    navLinks.forEach(anchor => {
        anchor.addEventListener('click', function(e) {
            e.preventDefault();
            
            const targetId = this.getAttribute('href');
            const targetElement = document.querySelector(targetId);
            
            if (targetElement) {
                window.scrollTo({
                    top: targetElement.offsetTop - 80,
                    behavior: 'smooth'
                });
                
                // Update URL without page reload
                history.pushState(null, null, targetId);
                
                // Update active class
                navLinks.forEach(link => link.classList.remove('active'));
                this.classList.add('active');
            }
        });
    });
}

/**
 * Initializes section highlighting in navigation based on scroll position
 */
function initSectionHighlighting() {
    const sections = document.querySelectorAll('section');
    const navLinks = document.querySelectorAll('.navigation a');
    
    // Set initial active state if URL has hash
    if (window.location.hash) {
        const activeLink = document.querySelector(`.navigation a[href="${window.location.hash}"]`);
        if (activeLink) {
            navLinks.forEach(link => link.classList.remove('active'));
            activeLink.classList.add('active');
        }
    }
    
    // Listen for scroll events to update active section
    window.addEventListener('scroll', function() {
        let currentSection = '';
        
        sections.forEach(section => {
            const sectionTop = section.offsetTop;
            const sectionHeight = section.clientHeight;
            
            if (window.scrollY >= sectionTop - 200) {
                currentSection = '#' + section.getAttribute('id');
            }
        });
        
        navLinks.forEach(link => {
            link.classList.remove('active');
            if (link.getAttribute('href') === currentSection) {
                link.classList.add('active');
            }
        });
    });
}

/**
 * Enhances figures with hover effects and zoom capability
 */
function enhanceFigures() {
    const figures = document.querySelectorAll('.figure');
    
    figures.forEach(figure => {
        // Add subtle hover effect
        figure.addEventListener('mouseenter', function() {
            this.style.transform = 'scale(1.01)';
            this.style.transition = 'transform 0.3s ease';
        });
        
        figure.addEventListener('mouseleave', function() {
            this.style.transform = 'scale(1)';
        });
    });
}

/**
 * Adds table of contents highlighting based on visible section
 * This runs on scroll and identifies which section is most visible
 */
function highlightTableOfContents() {
    const sections = document.querySelectorAll('section');
    const navLinks = document.querySelectorAll('.navigation a');
    
    // Calculate which section takes up most of the viewport
    const viewportHeight = window.innerHeight;
    let maxVisibleSection = null;
    let maxVisibleAmount = 0;
    
    sections.forEach(section => {
        const rect = section.getBoundingClientRect();
        const visibleTop = Math.max(0, rect.top);
        const visibleBottom = Math.min(viewportHeight, rect.bottom);
        const visibleAmount = Math.max(0, visibleBottom - visibleTop);
        
        if (visibleAmount > maxVisibleAmount) {
            maxVisibleAmount = visibleAmount;
            maxVisibleSection = section;
        }
    });
    
    // Update active link in navigation
    if (maxVisibleSection) {
        const currentSectionId = '#' + maxVisibleSection.getAttribute('id');
        navLinks.forEach(link => {
            link.classList.remove('active');
            if (link.getAttribute('href') === currentSectionId) {
                link.classList.add('active');
            }
        });
    }
}

// Add scroll event listener for more precise table of contents highlighting
window.addEventListener('scroll', function() {
    // Add throttling to avoid performance issues
    if (!window.requestAnimationFrame) {
        highlightTableOfContents();
        return;
    }
    
    if (!this.ticking) {
        this.ticking = true;
        window.requestAnimationFrame(() => {
            highlightTableOfContents();
            this.ticking = false;
        });
    }
});

// Enable PDF download button (simulated functionality)
document.addEventListener('DOMContentLoaded', function() {
    const downloadBtn = document.querySelector('.btn');
    
    if (downloadBtn) {
        downloadBtn.addEventListener('click', function(e) {
             window.open('https://arxiv.org/pdf/2503.23618v1', '_blank');
        });
    }
});