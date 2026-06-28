// ═══════════════════════════════════════════════════
// NAVBAR — Standalone Component Script
// ═══════════════════════════════════════════════════

// ── Load navbar HTML into page ──
async function loadNavbar() {
    const placeholder = document.getElementById('navbar-placeholder');
    if (!placeholder) return;

    try {
        const response = await fetch('./navbar.html');
        if (!response.ok) throw new Error('Failed to load navbar');
        const html = await response.text();
        placeholder.innerHTML = html;

        // Initialize navbar features after HTML is injected
        initNavbarScroll();
        initMobileMenu();
        initActiveNavLink();
    } catch (error) {
        console.error('Navbar load error:', error);
    }
}

// ── Scroll effect: add shadow on scroll ──
function initNavbarScroll() {
    const navbar = document.getElementById('navbar');
    if (!navbar) return;

    window.addEventListener('scroll', () => {
        navbar.classList.toggle('scrolled', window.scrollY > 10);
    });
}

// ── Mobile hamburger menu drawer toggle ──
function initMobileMenu() {
    const menuIcon = document.getElementById('menuIcon');
    const closeMenu = document.getElementById('closeMenu');
    const navDrawer = document.getElementById('navDrawer');
    const navOverlay = document.getElementById('navOverlay');

    if (!menuIcon || !navDrawer || !navOverlay) return;

    // Open drawer
    menuIcon.addEventListener('click', () => {
        navDrawer.classList.add('open');
        navOverlay.classList.add('open');
        document.body.style.overflow = 'hidden'; // Prevent background scroll
    });

    // Close drawer function
    const closeDrawer = () => {
        navDrawer.classList.remove('open');
        navOverlay.classList.remove('open');
        document.body.style.overflow = '';
    };

    if (closeMenu) {
        closeMenu.addEventListener('click', closeDrawer);
    }
    
    navOverlay.addEventListener('click', closeDrawer);

    // Close menu when a link is clicked
    const drawerLinks = navDrawer.querySelectorAll('.drawer-link');
    drawerLinks.forEach(link => {
        link.addEventListener('click', closeDrawer);
    });
}

// ── Highlight active nav link based on current page ──
function initActiveNavLink() {
    const navLinksAll = document.querySelectorAll('.nav-link, .drawer-link');
    const path = window.location.pathname.toLowerCase();

    // Clear active class from all links first
    navLinksAll.forEach(link => link.classList.remove('active'));

    if (path.includes('how-it-works')) {
        navLinksAll.forEach(link => {
            const href = link.getAttribute('href');
            if (href && href.includes('how-it-works')) {
                link.classList.add('active');
            }
        });
    } else if (path.includes('about')) {
        navLinksAll.forEach(link => {
            const href = link.getAttribute('href');
            if (href && href.includes('about')) {
                link.classList.add('active');
            }
        });
    } else {
        // Default to Home page (index.html or /)
        navLinksAll.forEach(link => {
            const href = link.getAttribute('href');
            if (href === '/' || href.includes('index.html')) {
                link.classList.add('active');
            }
        });
    }
}

// ── Auto-initialize when DOM is ready ──
document.addEventListener('DOMContentLoaded', loadNavbar);
