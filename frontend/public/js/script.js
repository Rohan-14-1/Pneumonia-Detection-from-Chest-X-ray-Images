// ============================================================
// API Configuration
// Replace this with your Cloud Run backend URL after deployment
// ============================================================
const API_URL = "https://pneumonia-detection-XXXXX-xx.a.run.app";

// Selectors
const imageInput = document.getElementById('imageInput');
const previewImage = document.getElementById('previewImage');
const placeholderText = document.getElementById('placeholderText');
const dropZone = document.getElementById('dropZone');
const scanOverlay = document.getElementById('scanOverlay');
const analyzeBtn = document.getElementById('analyzeBtn');
const resultBox = document.getElementById('resultBox');
const loader = document.getElementById('loader');

// Result Selectors
const predictionText = document.getElementById('predictionText');
const confidenceText = document.getElementById('confidenceText');
const confidenceBar = document.getElementById('confidenceBar');
const riskLevel = document.getElementById('riskLevel');
const analysisTime = document.getElementById('analysisTime');
const recommendationText = document.getElementById('recommendationText');

// Preview Image Handling
function handleFile(file) {
    if (file && file.type.startsWith('image/')) {
        const reader = new FileReader();
        reader.onload = (e) => {
            previewImage.src = e.target.result;
            previewImage.classList.remove('hidden');
            placeholderText.classList.add('hidden');
            resultBox.classList.add('hidden');
        };
        reader.readAsDataURL(file);
    }
}

imageInput.addEventListener('change', (e) => handleFile(e.target.files[0]));

// Drag and Drop
dropZone.addEventListener('dragover', (e) => {
    e.preventDefault();
    dropZone.style.boxShadow = "0 0 0 4px #2563eb44";
});

dropZone.addEventListener('dragleave', () => {
    dropZone.style.boxShadow = "0 25px 50px -12px rgba(0, 0, 0, 0.08)";
});

dropZone.addEventListener('drop', (e) => {
    e.preventDefault();
    dropZone.style.boxShadow = "0 25px 50px -12px rgba(0, 0, 0, 0.08)";
    const files = e.dataTransfer.files;
    if (files.length > 0) {
        imageInput.files = files;
        handleFile(files[0]);
    }
});

// Prediction Engine
async function predict() {
    const file = imageInput.files[0];
    if (!file) return alert("Please upload an X-ray first.");

    // UI State: Scanning
    resultBox.classList.add('hidden');
    loader.classList.remove('hidden');
    scanOverlay.classList.remove('hidden');
    analyzeBtn.disabled = true;
    analyzeBtn.innerText = "Processing Neural Nodes...";

    const startTime = performance.now();
    const formData = new FormData();
    formData.append("file", file);

    try {
        const response = await fetch(`${API_URL}/predict`, {
            method: "POST",
            body: formData
        });

        if (!response.ok) throw new Error("API Error");
        const data = await response.json();
        const endTime = performance.now();

        // Update UI
        setTimeout(() => {
            loader.classList.add('hidden');
            scanOverlay.classList.add('hidden');
            resultBox.classList.remove('hidden');
            analyzeBtn.disabled = false;
            analyzeBtn.innerText = "Run Neural Analysis";

            // Populate Results
            predictionText.innerText = data.result;
            confidenceText.innerText = data.confidence;
            confidenceBar.style.width = data.confidence + "%";
            analysisTime.innerText = ((endTime - startTime) / 1000).toFixed(2) + "s";
            
            const isPneumonia = data.result === "PNEUMONIA";
            const themeColor = isPneumonia ? "#ef4444" : "#10b981";
            
            predictionText.style.color = themeColor;
            confidenceBar.style.backgroundColor = themeColor;
            riskLevel.innerText = isPneumonia ? "CRITICAL" : "LOW";
            riskLevel.style.color = themeColor;
            
            recommendationText.innerText = isPneumonia 
                ? `Anomalies detected in ${data.lung}. Correlation with clinical findings and immediate specialist consultation is required.`
                : "No significant abnormalities detected. Continue routine clinical monitoring.";
        }, 800);

    } catch (error) {
        console.error(error);
        loader.classList.add('hidden');
        scanOverlay.classList.add('hidden');
        analyzeBtn.disabled = false;
        analyzeBtn.innerText = "Run Neural Analysis";
        alert("Server connection failed. Please check if the API backend is running.");
    }
}
