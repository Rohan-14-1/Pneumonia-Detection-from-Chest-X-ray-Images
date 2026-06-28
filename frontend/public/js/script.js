// ═══════════════════════════════════════════════════
// PneumoScan AI — Main Application Logic
// (Navbar logic is in navbar.js)
// ═══════════════════════════════════════════════════

// Automatically use local Flask server when testing locally, otherwise use production Render backend
const API_URL = window.location.hostname === "localhost" || window.location.hostname === "127.0.0.1" || window.location.protocol === "file:"
    ? "http://localhost:5000"
    : "https://pneumonia-detection-from-chest-x-ray-images.onrender.com";

// ═══════════════ DOM SELECTORS ═══════════════
const imageInput = document.getElementById('imageInput');
const previewImage = document.getElementById('previewImage');
const previewPlaceholder = document.getElementById('previewPlaceholder');
const previewArea = document.getElementById('previewArea');
const dropZone = document.getElementById('dropZone');
const analyzeBtn = document.getElementById('analyzeBtn');
const loader = document.getElementById('loader');
const resultBox = document.getElementById('resultBox');

// Result elements
const predictionText = document.getElementById('predictionText');
const confidenceText = document.getElementById('confidenceText');
const confidenceBar = document.getElementById('confidenceBar');
const riskLevel = document.getElementById('riskLevel');
const lungSide = document.getElementById('lungSide');
const analysisTime = document.getElementById('analysisTime');
const recommendationText = document.getElementById('recommendationText');


// ═══════════════ FILE HANDLING ═══════════════
function handleFile(file) {
    if (!file) return;

    // Validate file type
    const validTypes = ['image/png', 'image/jpeg', 'image/jpg'];
    if (!validTypes.includes(file.type)) {
        alert('Please upload a PNG, JPG, or JPEG image.');
        return;
    }

    // Validate file size (10MB max)
    if (file.size > 10 * 1024 * 1024) {
        alert('File size exceeds 10MB. Please upload a smaller image.');
        return;
    }

    const reader = new FileReader();
    reader.onload = (e) => {
        previewImage.src = e.target.result;
        previewImage.classList.remove('hidden');
        previewPlaceholder.classList.add('hidden');
        resultBox.classList.add('hidden');
    };
    reader.readAsDataURL(file);
}

// File input change
imageInput.addEventListener('change', (e) => {
    handleFile(e.target.files[0]);
});


// ═══════════════ DRAG & DROP ═══════════════
dropZone.addEventListener('dragover', (e) => {
    e.preventDefault();
    e.stopPropagation();
    dropZone.classList.add('dragover');
});

dropZone.addEventListener('dragleave', (e) => {
    e.preventDefault();
    e.stopPropagation();
    dropZone.classList.remove('dragover');
});

dropZone.addEventListener('drop', (e) => {
    e.preventDefault();
    e.stopPropagation();
    dropZone.classList.remove('dragover');

    const files = e.dataTransfer.files;
    if (files.length > 0) {
        const dt = new DataTransfer();
        dt.items.add(files[0]);
        imageInput.files = dt.files;
        handleFile(files[0]);
    }
});

// Click on drop zone opens file browser
dropZone.addEventListener('click', (e) => {
    if (e.target.tagName !== 'INPUT' && !e.target.closest('.browse-btn')) {
        imageInput.click();
    }
});


// ═══════════════ PREDICTION ENGINE ═══════════════
async function predict() {
    const file = imageInput.files[0];
    if (!file) {
        alert('Please upload a chest X-ray image first.');
        return;
    }

    // UI → Loading state
    resultBox.classList.add('hidden');
    loader.classList.remove('hidden');
    analyzeBtn.disabled = true;
    analyzeBtn.innerHTML = '<i class="fas fa-spinner fa-spin"></i> Analyzing...';

    const startTime = performance.now();
    const formData = new FormData();
    formData.append('file', file);

    try {
        const response = await fetch(`${API_URL}/predict`, {
            method: 'POST',
            body: formData
        });

        if (!response.ok) {
            const errData = await response.json().catch(() => ({}));
            throw new Error(errData.error || `Server error (${response.status})`);
        }

        const data = await response.json();
        const endTime = performance.now();
        const processTime = ((endTime - startTime) / 1000).toFixed(2);

        // Small delay for smooth UX transition
        setTimeout(() => {
            displayResults(data, processTime);
        }, 600);

    } catch (error) {
        console.error('Prediction error:', error);
        loader.classList.add('hidden');
        analyzeBtn.disabled = false;
        analyzeBtn.innerHTML = '<i class="fas fa-wand-magic-sparkles"></i> Analyze X-ray with AI';
        alert(`Analysis failed: ${error.message}\n\nPlease ensure the API server is running.`);
    }
}


// ═══════════════ DISPLAY RESULTS ═══════════════
function displayResults(data, processTime) {
    loader.classList.add('hidden');
    resultBox.classList.remove('hidden');
    analyzeBtn.disabled = false;
    analyzeBtn.innerHTML = '<i class="fas fa-wand-magic-sparkles"></i> Analyze X-ray with AI';

    const isPneumonia = data.result === 'PNEUMONIA';
    const themeColor = isPneumonia ? '#ef4444' : '#10b981';

    // Prediction text
    predictionText.innerText = data.result;
    predictionText.style.color = themeColor;

    // Confidence
    confidenceText.innerText = data.confidence;
    confidenceBar.style.width = data.confidence + '%';
    confidenceBar.style.backgroundColor = themeColor;

    // Lung side
    lungSide.innerText = data.lung || '--';

    // Risk level
    riskLevel.innerText = isPneumonia ? 'CRITICAL' : 'LOW';
    riskLevel.style.color = themeColor;

    // Process time
    analysisTime.innerText = processTime + 's';

    // Recommendation
    if (isPneumonia) {
        recommendationText.innerText = `Anomalies detected in ${data.lung}. Correlation with clinical findings and immediate specialist consultation is strongly recommended.`;
        recommendationText.style.borderLeftColor = '#ef4444';
    } else {
        recommendationText.innerText = 'No significant pulmonary abnormalities detected. Continue routine clinical monitoring as per standard protocol.';
        recommendationText.style.borderLeftColor = '#10b981';
    }
}

