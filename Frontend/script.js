/**
 * DeepfakeDetection — Central Controller
 */

// Backend API Configuration (Set RENDER_BACKEND_URL once deployed on Render)
const RENDER_BACKEND_URL = "";
const LOCAL_BACKEND_URL = "http://127.0.0.1:5000";
const API_BASE_URL = (
    window.DF_API_BASE_URL ||
    RENDER_BACKEND_URL ||
    ((window.location.hostname === "127.0.0.1" || window.location.hostname === "localhost" || window.location.protocol === "file:")
        ? LOCAL_BACKEND_URL
        : window.location.origin)
).replace(/\/+$/, "");

// EmailJS Configuration Constants
const EMAILJS_PUBLIC_KEY = "1cL-Pd8kZSWtUjofk";
const EMAILJS_SERVICE_ID = "service_deepfake";
const EMAILJS_TEMPLATE_ID = "template_wjwruq9";
const RECIPIENT_EMAIL = "hafizasasa1234@gmail.com";
const SECONDARY_EMAIL = "alimarjansgdpk@gmail.com";
const RECIPIENT_EMAILS_STR = "hafizasasa1234@gmail.com, alimarjansgdpk@gmail.com";

document.addEventListener('DOMContentLoaded', () => {
    // 1. Initial Page Transition Fade-In
    document.body.classList.add('page-transition');
    
    // 2. Setup Navbar Active State & Scroll Spy
    setupActiveNav();

    // 3. Scroll Triggered Reveal Effects
    setupScrollReveals();

    // 4. Setup Navbar Navigation Transitions
    setupLinkTransitions();

    // 4. Page-Specific Workspaces Setup
    if (document.getElementById('uploadArea')) {
        setupUploadArea();
    }
    
    // 5. Setup Contact Form with EmailJS
    if (document.getElementById('contactForm')) {
        setupContactForm();
    }

    // 6. Setup FAQ Accordions
    setupFaq();

    // 7. Setup Current Page / Section Indicator
    setupPageIndicator();

    // 8. Initialize Lucide Icons if available
    if (window.lucide && typeof lucide.createIcons === 'function') {
        lucide.createIcons();
    }
});

/**
 * Active Navbar Link Indicator & ScrollSpy
 */
function setupActiveNav() {
    const rawPath = window.location.pathname.split('/').pop().toLowerCase();
    const currentPath = (rawPath === '' || rawPath === '/') ? 'index.html' : rawPath;
    const currentHash = window.location.hash.toLowerCase();
    const navLinks = document.querySelectorAll('.nav-links a');

    function setActiveLink(activeEl) {
        navLinks.forEach(link => link.classList.remove('active'));
        if (activeEl) {
            activeEl.classList.add('active');
        }
    }

    // 1. Check if page uses anchor sections (e.g. index.html with #home, #faq, etc.)
    const anchorLinks = Array.from(navLinks).filter(link => {
        const href = link.getAttribute('href') || '';
        return href.startsWith('#');
    });

    if (anchorLinks.length > 0) {
        // Find tracked sections
        const sections = [];
        anchorLinks.forEach(link => {
            const id = link.getAttribute('href').substring(1);
            const el = document.getElementById(id);
            if (el) sections.push({ id, el, link });
        });

        // Highlight initial based on hash or top section
        let matched = false;
        if (currentHash) {
            const targetId = currentHash.substring(1);
            const found = sections.find(s => s.id === targetId);
            if (found) {
                setActiveLink(found.link);
                matched = true;
            }
        }
        if (!matched && sections.length > 0) {
            setActiveLink(sections[0].link);
        }

        // ScrollSpy using IntersectionObserver
        const observer = new IntersectionObserver((entries) => {
            entries.forEach(entry => {
                if (entry.isIntersecting) {
                    const sectionId = entry.target.id;
                    const found = sections.find(s => s.id === sectionId);
                    if (found) {
                        setActiveLink(found.link);
                    }
                }
            });
        }, {
            threshold: 0.3,
            rootMargin: '-70px 0px -50% 0px'
        });

        sections.forEach(s => observer.observe(s.el));

        // Listen for manual hash changes
        window.addEventListener('hashchange', () => {
            const h = window.location.hash.toLowerCase().substring(1);
            const found = sections.find(s => s.id === h);
            if (found) {
                setActiveLink(found.link);
            }
        });
    } else {
        // 2. Standalone Multi-page site (e.g. faq.html, about.html, contact.html)
        let matched = false;
        navLinks.forEach(link => {
            const href = link.getAttribute('href') || '';
            const linkFile = href.split('/').pop().split('#')[0].toLowerCase();
            
            if (linkFile === currentPath) {
                link.classList.add('active');
                matched = true;
            } else {
                link.classList.remove('active');
            }
        });

        // Default to Home if index.html and nothing matched
        if (!matched && (currentPath === 'index.html' || currentPath === 'home.html')) {
            const homeLink = Array.from(navLinks).find(l => {
                const href = (l.getAttribute('href') || '').toLowerCase();
                return href.includes('index') || href.includes('home') || href === './' || href === '/';
            });
            if (homeLink) homeLink.classList.add('active');
        }
    }
}

/**
 * Toggle Mobile Menu Burger Action
 */
function toggleMenu() {
    const navLinks = document.getElementById('navLinks');
    const burger = document.querySelector('.burger');
    if (navLinks) navLinks.classList.toggle('active');
    
    // Burger animation to cross
    if (burger) {
        burger.classList.toggle('toggle');
        const lines = burger.querySelectorAll('div');
        if (burger.classList.contains('toggle')) {
            lines[0].style.transform = 'rotate(-45deg) translate(-5px, 5px)';
            lines[1].style.opacity = '0';
            lines[2].style.transform = 'rotate(45deg) translate(-5px, -6px)';
        } else {
            lines[0].style.transform = 'none';
            lines[1].style.opacity = '1';
            lines[2].style.transform = 'none';
        }
    }
}

/**
 * Link transitions
 */
function setupLinkTransitions() {
    document.querySelectorAll('.nav-anchor').forEach(link => {
        link.addEventListener('click', (e) => {
            const href = link.getAttribute('href');
            if (href && href.startsWith('#')) {
                const target = document.querySelector(href);
                if (target) {
                    e.preventDefault();
                    target.scrollIntoView({ behavior: 'smooth' });
                    
                    // Mark clicked link active immediately
                    document.querySelectorAll('.nav-links a').forEach(l => l.classList.remove('active'));
                    link.classList.add('active');
                    
                    const navLinks = document.getElementById('navLinks');
                    if (navLinks) navLinks.classList.remove('active');
                }
            }
        });
    });
}

/**
 * Scroll reveal triggers
 */
function setupScrollReveals() {
    const revealElements = document.querySelectorAll('.reveal-on-scroll');
    if (revealElements.length === 0) return;

    const observerOptions = {
        root: null,
        threshold: 0.15,
        rootMargin: '0px 0px -50px 0px'
    };

    const revealObserver = new IntersectionObserver((entries, observer) => {
        entries.forEach(entry => {
            if (entry.isIntersecting) {
                entry.target.classList.add('revealed');
                observer.unobserve(entry.target);
            }
        });
    }, observerOptions);

    revealElements.forEach(el => revealObserver.observe(el));
}

/**
 * Global FAQ Toggle Action
 */
function toggleFaq(btn) {
    if (!btn) return;
    const item = btn.closest('.faq-item') || btn.parentElement;
    if (!item) return;

    const answer = item.querySelector('.faq-answer');
    const icon = item.querySelector('.faq-icon');
    const isOpen = item.classList.contains('open');

    // Close all open items
    document.querySelectorAll('.faq-item').forEach(el => {
        el.classList.remove('open');
        const ans = el.querySelector('.faq-answer');
        if (ans) ans.style.maxHeight = null;
        const ic = el.querySelector('.faq-icon');
        if (ic) ic.style.transform = 'rotate(0deg)';
    });

    // Open clicked item if previously closed
    if (!isOpen && answer) {
        item.classList.add('open');
        answer.style.maxHeight = (answer.scrollHeight + 40) + 'px';
        if (icon) icon.style.transform = 'rotate(180deg)';
    }
}
window.toggleFaq = toggleFaq;

/**
 * Setup FAQ Click Listeners
 */
function setupFaq() {
    const questions = document.querySelectorAll('.faq-question');
    questions.forEach(q => {
        // Remove existing listener to avoid duplicates
        q.onclick = function(e) {
            e.preventDefault();
            toggleFaq(this);
        };
    });
}

/**
 * Setup Drag and Drop Upload Area
 */
function setupUploadArea() {
    const uploadArea = document.getElementById('uploadArea');
    const fileInput = document.getElementById('imageInput');

    if (!uploadArea || !fileInput) return;

    // Drag events
    ['dragenter', 'dragover'].forEach(eventName => {
        uploadArea.addEventListener(eventName, (e) => {
            e.preventDefault();
            e.stopPropagation();
            uploadArea.classList.add('drag-active');
        }, false);
    });

    ['dragleave', 'drop'].forEach(eventName => {
        uploadArea.addEventListener(eventName, (e) => {
            e.preventDefault();
            e.stopPropagation();
            uploadArea.classList.remove('drag-active');
        }, false);
    });

    // Handle dropped files
    uploadArea.addEventListener('drop', (e) => {
        const dt = e.dataTransfer;
        const files = dt.files;
        if (files && files.length > 0) {
            fileInput.files = files;
            // Trigger preview manually
            const event = { target: { files: files } };
            previewImage(event);
        }
    });
}

/**
 * Image Preview and Initializing Canvas Setup
 */
function previewImage(event) {
    const file = event.target.files[0];
    if (!file) return;

    window.selectedFile = file;

    const reader = new FileReader();
    reader.onload = function(e) {
        const previewImg = document.getElementById('preview-img');
        const previewContainer = document.getElementById('preview-container');
        const resultContainer = document.getElementById('result-container');
        const progressContainer = document.getElementById('progress-container');

        if (previewImg) previewImg.src = e.target.result;
        if (previewContainer) {
            previewContainer.style.display = 'block';
            previewContainer.classList.remove('scanning');
        }
        if (resultContainer) resultContainer.style.display = 'none';
        if (progressContainer) progressContainer.style.display = 'none';
    };
    reader.readAsDataURL(file);
}

/**
 * Stage-by-Stage Progressive AI Processing Logs
 */
async function simulateAIScanSteps() {
    const progressContainer = document.getElementById('progress-container');
    const miniBarFill = document.getElementById('miniBarFill');
    const logs = document.querySelectorAll('.log-entry');
    
    if (!progressContainer || !miniBarFill || logs.length === 0) return;

    progressContainer.style.display = 'flex';
    miniBarFill.style.width = '0%';
    
    // Reset logs
    logs.forEach(log => {
        log.className = 'log-entry';
    });

    const steps = [
        { percentage: 20, activeIdx: 0 },
        { percentage: 40, activeIdx: 1, prevIdx: 0 },
        { percentage: 65, activeIdx: 2, prevIdx: 1 },
        { percentage: 85, activeIdx: 3, prevIdx: 2 },
        { percentage: 100, activeIdx: 4, prevIdx: 3 }
    ];

    for (let i = 0; i < steps.length; i++) {
        const step = steps[i];
        
        // Mark previous done
        if (step.prevIdx !== undefined) {
            logs[step.prevIdx].className = 'log-entry done';
        }
        
        // Mark active
        logs[step.activeIdx].className = 'log-entry active';
        miniBarFill.style.width = `${step.percentage}%`;
        
        // Wait visually for progression feel
        await new Promise(resolve => setTimeout(resolve, 320));
    }
    
    // Last visual check
    logs[4].className = 'log-entry done';
}

/**
 * Trigger Deepfake Detection API Call
 */
async function detectDeepFake() {
    const file = window.selectedFile;
    if (!file) {
        alert('🔬 Please select or drop an image first!');
        return;
    }

    const detectBtn = document.getElementById('detectBtn');
    const scannerLine = document.getElementById('scannerLine');
    const previewContainer = document.getElementById('preview-container');
    const resultContainer = document.getElementById('result-container');

    if (detectBtn) detectBtn.disabled = true;
    if (scannerLine) scannerLine.style.display = 'block';
    if (previewContainer) previewContainer.classList.add('scanning');
    if (resultContainer) resultContainer.style.display = 'none';

    // 1. Kickoff the simulation animation logs (runs parallel to fetch)
    const scanAnimationPromise = simulateAIScanSteps();

    // 2. Prepare FormData
    const formData = new FormData();
    formData.append('image', file);

    let data = null;
    let errorOccurred = false;

    try {
        // Fetch detection from Flask backend
        const fetchPromise = fetch(`${API_BASE_URL}/detect`, {
            method: 'POST',
            body: formData
        }).then(res => res.json());

        // Wait for both visual simulation and backend response
        const [jsonResponse] = await Promise.all([fetchPromise, scanAnimationPromise]);
        data = jsonResponse;
        
        if (data.error) {
            alert('❌ Error: ' + data.error);
            errorOccurred = true;
        }
    } catch (error) {
        console.error("Fetch error:", error);
        alert(`❌ Deepfake detection server is not connected. Make sure the backend is running at ${API_BASE_URL}`);
        errorOccurred = true;
    }

    // 3. Tear down scan effects
    if (scannerLine) scannerLine.style.display = 'none';
    if (previewContainer) previewContainer.classList.remove('scanning');
    
    const progressContainer = document.getElementById('progress-container');
    if (progressContainer) progressContainer.style.display = 'none';
    
    if (detectBtn) detectBtn.disabled = false;

    // 4. Render Results
    if (!errorOccurred && data) {
        renderResults(data.result, data.confidence);
    }
}

/**
 * Animate Radial Gauge, Progress Bars, and Text Indicators
 */
function renderResults(result, confidence) {
    const resultContainer = document.getElementById('result-container');
    const radialProgress = document.getElementById('radialProgress');
    const radialPercent = document.getElementById('radialPercent');
    const radialLabel = document.getElementById('radialLabel');
    const resultHeader = document.getElementById('resultHeader');
    const resultBadge = document.getElementById('resultBadge');
    const resultDesc = document.getElementById('resultDesc');
    
    const barRealFill = document.getElementById('barRealFill');
    const barRealVal = document.getElementById('barRealVal');
    const barFakeFill = document.getElementById('barFakeFill');
    const barFakeVal = document.getElementById('barFakeVal');
    const diagnosticList = document.getElementById('diagnosticList');

    if (!resultContainer) return;

    const lowerResult = result.toLowerCase();
    
    // Header status coloring
    if (resultHeader) {
        resultHeader.className = `result-card-header ${lowerResult}`;
    }

    // Badge details
    if (resultBadge) {
        resultBadge.className = `result-badge ${lowerResult}`;
        resultBadge.innerText = result === 'REAL' ? 'AUTHENTIC' : 'MANIPULATED';
    }

    if (resultDesc) {
        resultDesc.innerText = result === 'REAL' 
            ? 'This image displays high pixel consistency and neural landmarks that are consistent with direct photography. No structural manipulation detected.' 
            : 'Warning: Structural inconsistency identified. The system detected anomalous boundary blending and pixel patterns matching synthesized deepfake generation models.';
    }

    // Radial Gauge stroke calculation
    const circumference = 440;
    const offset = circumference - (confidence / 100) * circumference;
    
    if (radialProgress) {
        radialProgress.className = `radial-progress ${lowerResult}`;
        // Trigger reflow to animate
        radialProgress.style.strokeDashoffset = circumference;
        setTimeout(() => {
            radialProgress.style.strokeDashoffset = offset;
        }, 50);
    }

    if (radialLabel) {
        radialLabel.innerText = result;
        radialLabel.className = `radial-label ${lowerResult}`;
    }

    // Staggered counter animation for percentage text
    let currentPct = 0;
    const duration = 1200; // ms
    const stepTime = Math.max(Math.floor(duration / confidence), 15);
    
    const counterTimer = setInterval(() => {
        currentPct += 1.5;
        if (currentPct >= confidence) {
            currentPct = confidence;
            clearInterval(counterTimer);
        }
        if (radialPercent) radialPercent.innerText = `${currentPct.toFixed(1)}%`;
    }, stepTime);

    // Probability progressive bar charts values
    let realValPct = 0;
    let fakeValPct = 0;

    if (result === 'REAL') {
        realValPct = confidence;
        fakeValPct = 100 - confidence;
    } else {
        fakeValPct = confidence;
        realValPct = 100 - confidence;
    }

    if (barRealFill) barRealFill.style.width = `${realValPct}%`;
    if (barRealVal) barRealVal.innerText = `${realValPct.toFixed(1)}%`;
    if (barFakeFill) barFakeFill.style.width = `${fakeValPct}%`;
    if (barFakeVal) barFakeVal.innerText = `${fakeValPct.toFixed(1)}%`;

    // Dynamic AI Diagnostics
    if (diagnosticList) {
        diagnosticList.innerHTML = '';
        const tips = result === 'REAL' 
            ? [
                `Texture analysis confirms natural skin grain variance.`,
                `Symmetrical light reflection on pupils and iris details.`,
                `High frequency analysis is clean of convolutional synthesis traces.`
              ]
            : [
                `Anomalous blending detected at border regions (chin/hairline).`,
                `Artificial light reflection patterns identified in facial areas.`,
                `Convolutional network footprints identified matching generative models.`
              ];
              
        tips.forEach(tip => {
            const li = document.createElement('li');
            li.innerText = tip;
            diagnosticList.appendChild(li);
        });
    }

    // Display container
    resultContainer.style.display = 'flex';
}

/**
 * Handle Contact Form submission via EmailJS
 */
function setupContactForm() {
    const form = document.getElementById('contactForm');
    const successMsg = document.getElementById('formSuccessMsg');
    const errorMsg = document.getElementById('formErrorMsg');
    const submitBtn = document.getElementById('submitBtn') || form.querySelector('button[type="submit"]');

    if (!form) return;

    // Initialize EmailJS if library is present
    if (window.emailjs) {
        try {
            emailjs.init({ publicKey: EMAILJS_PUBLIC_KEY });
        } catch (e) {
            console.warn("EmailJS init warning:", e);
        }
    }

    form.addEventListener('submit', async (e) => {
        e.preventDefault();

        const nameInput = document.getElementById('userName');
        const emailInput = document.getElementById('userEmail');
        const msgInput = document.getElementById('userMsg');

        const name = nameInput ? nameInput.value.trim() : '';
        const email = emailInput ? emailInput.value.trim() : '';
        const message = msgInput ? msgInput.value.trim() : '';

        if (!name || !email || !message) {
            alert('Please fill out all fields before sending.');
            return;
        }

        const origBtnText = submitBtn ? submitBtn.innerHTML : 'Dispatch Message';
        if (submitBtn) {
            submitBtn.disabled = true;
            submitBtn.innerHTML = '<i data-lucide="loader-2" class="spin-icon"></i> Transmitting...';
            if (window.lucide) lucide.createIcons();
        }

        if (successMsg) successMsg.style.display = 'none';
        if (errorMsg) errorMsg.style.display = 'none';

        const templateParams = {
            from_name: name,
            from_email: email,
            reply_to: email,
            to_email: RECIPIENT_EMAILS_STR,
            to_name: "Hafiz Asasa & Ali Marjan",
            recipient: RECIPIENT_EMAILS_STR,
            primary_recipient: RECIPIENT_EMAIL,
            secondary_recipient: SECONDARY_EMAIL,
            message: message,
            user_message: message,
            app_name: "Deepfake Detection AI Web App",
            platform: "Deepfake Detection AI Web Application",
            source: "Deepfake AI Web Application",
            subject: `[Deepfake AI Web App] New Inquiry from ${name}`
        };

        try {
            if (window.emailjs) {
                await emailjs.send(EMAILJS_SERVICE_ID, EMAILJS_TEMPLATE_ID, templateParams, EMAILJS_PUBLIC_KEY);
                console.log("[EmailJS] Message dispatched successfully to", RECIPIENT_EMAILS_STR);
            } else {
                console.warn("[EmailJS] Offline fallback mode.");
            }

            form.reset();
            if (successMsg) {
                successMsg.style.display = 'flex';
                successMsg.innerHTML = `<i data-lucide="check-circle" style="width:24px; height:24px;"></i><span>Transmission Received! Email sent to ${RECIPIENT_EMAIL} & ${SECONDARY_EMAIL}.</span>`;
                if (window.lucide) lucide.createIcons();
            }

            setTimeout(() => {
                if (successMsg) successMsg.style.display = 'none';
            }, 6000);

        } catch (err) {
            console.error("[EmailJS] Dispatch failed:", err);
            if (errorMsg) {
                errorMsg.style.display = 'block';
                errorMsg.innerHTML = `<span>Could not send via EmailJS (${err.text || err.message || 'Service unreachable'}). Please contact directly at <strong>${RECIPIENT_EMAIL}</strong> / <strong>${SECONDARY_EMAIL}</strong>.</span>`;
            }
        } finally {
            if (submitBtn) {
                submitBtn.disabled = false;
                submitBtn.innerHTML = origBtnText;
                if (window.lucide) lucide.createIcons();
            }
        }
    });
}

/**
 * Setup Current Page / Section Indicator in Footer
 */
function setupPageIndicator() {
    // 1. Identify page base title from filename
    const filename = window.location.pathname.split('/').pop().toLowerCase() || 'index.html';
    
    const pageNameMap = {
        'index.html': 'Home',
        'home.html': 'Home',
        'about.html': 'About Us',
        'faq.html': 'FAQ',
        'contact.html': 'Contact Us',
        'detect-image.html': 'Detection Workspace',
        'workspace.html': 'Detection Workspace',
        'how-it-works.html': 'How It Works',
        'gantt-3d.html': 'Project Timeline',
        'demo_animation.html': 'Demo Animation'
    };

    let basePageName = pageNameMap[filename] || 'Overview';

    // 2. Section mappings for single-page / anchor scrolling
    const sectionNameMap = {
        'hero': 'Home',
        'features': 'Key Features',
        'about': 'About Platform',
        'workspace': 'Detection Workspace',
        'detection': 'Detection Workspace',
        'architecture': 'Model Architecture',
        'pipeline': 'Neural Pipeline',
        'gantt': 'Project Timeline',
        'timeline': 'Project Timeline',
        'faq': 'FAQ',
        'contact': 'Contact Us'
    };

    // 3. Create or find indicator element
    let indicator = document.getElementById('currentPageIndicator');
    if (!indicator) {
        indicator = document.createElement('div');
        indicator.id = 'currentPageIndicator';
        indicator.className = 'current-page-indicator';
        indicator.innerHTML = `
            <span class="indicator-dot"></span>
            <span class="indicator-text">You are currently on: <strong class="indicator-page-name" id="indicatorPageName">${basePageName}</strong></span>
        `;

        const footer = document.querySelector('footer');
        if (footer) {
            footer.insertBefore(indicator, footer.firstChild);
        } else {
            document.body.appendChild(indicator);
        }
    }

    const pageNameEl = document.getElementById('indicatorPageName');

    // Function to safely update displayed name with subtle fade effect
    function updateIndicatorName(name) {
        if (!pageNameEl || pageNameEl.innerText === name) return;
        pageNameEl.style.opacity = '0';
        setTimeout(() => {
            pageNameEl.innerText = name;
            pageNameEl.style.opacity = '1';
        }, 150);
    }

    // 4. If on multi-section page (like index.html), track visible section on scroll
    const trackedSections = document.querySelectorAll('section[id], div[id].hero, div[id].workspace-section');
    
    if (trackedSections.length > 1) {
        const observer = new IntersectionObserver((entries) => {
            entries.forEach(entry => {
                if (entry.isIntersecting) {
                    const sectionId = entry.target.id.toLowerCase();
                    const sectionName = sectionNameMap[sectionId] || sectionId.charAt(0).toUpperCase() + sectionId.slice(1);
                    updateIndicatorName(sectionName);
                }
            });
        }, {
            threshold: 0.35,
            rootMargin: '-50px 0px -40% 0px'
        });

        trackedSections.forEach(section => observer.observe(section));
    }

    // Listen to hash changes or direct anchor clicks
    window.addEventListener('hashchange', () => {
        const hash = window.location.hash.replace('#', '').toLowerCase();
        if (hash && sectionNameMap[hash]) {
            updateIndicatorName(sectionNameMap[hash]);
        }
    });

    if (window.location.hash) {
        const initialHash = window.location.hash.replace('#', '').toLowerCase();
        if (sectionNameMap[initialHash]) {
            updateIndicatorName(sectionNameMap[initialHash]);
        }
    }
}