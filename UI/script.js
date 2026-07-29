
const fullCanvas = document.getElementById('fullNet');
const fullCtx = fullCanvas.getContext('2d');

const fullLayers = [
    { count: 6, label: "Input (784)" },
    { count: 8, label: "Dense (256)" },
    { count: 7, label: "Dense (128)" },
    { count: 5, label: "Dense (64)" },
    { count: 4, label: "Output (10)" }
];

let fullPositions = [];
let particles = [];

function setupFullNetwork() {
    const w = fullCanvas.width;
    const h = fullCanvas.height;
    const spacing = w / (fullLayers.length + 1);
    
    fullPositions = fullLayers.map((layer, i) => {
        const x = spacing * (i + 1);
        const ySpacing = h / (layer.count + 1);
        return Array.from({ length: layer.count }, (_, j) => ({
            x: x,
            y: ySpacing * (j + 1)
        }));
    });
}

function drawFullNetwork() {
    const ctx = fullCtx;
    ctx.clearRect(0, 0, fullCanvas.width, fullCanvas.height);
    
    ctx.strokeStyle = "#bbb";
    ctx.lineWidth = 1.5;
    for (let i = 0; i < fullPositions.length - 1; i++) {
        for (const n1 of fullPositions[i]) {
            for (const n2 of fullPositions[i + 1]) {
                ctx.beginPath();
                ctx.moveTo(n1.x, n1.y);
                ctx.lineTo(n2.x, n2.y);
                ctx.stroke();
            }
        }
    }

    ctx.fillStyle = "#333";
    ctx.font = "11px sans-serif";
    ctx.textAlign = "center";
    
    fullLayers.forEach((layer, i) => {
        ctx.fillText(layer.label, fullPositions[i][0].x, fullCanvas.height - 10);
        for (const pos of fullPositions[i]) {
            ctx.beginPath();
            ctx.arc(pos.x, pos.y, 4, 0, Math.PI * 2);
            ctx.fillStyle = "#fff";
            ctx.strokeStyle = "#333";
            ctx.lineWidth = 2;
            ctx.fill();
            ctx.stroke();
        }
    });

    if (Math.random() < 0.7) {
        const start = fullPositions[0][Math.floor(Math.random() * fullPositions[0].length)];
        const target = fullPositions[1][Math.floor(Math.random() * fullPositions[1].length)];
        particles.push({
            startX: start.x, startY: start.y,
            endX: target.x, endY: target.y,
            progress: 0,
            speed: 0.02 + Math.random() * 0.02,
            layer: 0
        });
    }

    particles = particles.filter(p => {
        p.progress += p.speed;
        if (p.progress >= 1) {
            p.layer++;
            if (p.layer < fullPositions.length - 1) {
                p.startX = p.endX;
                p.startY = p.endY;
                const next = fullPositions[p.layer + 1][Math.floor(Math.random() * fullPositions[p.layer + 1].length)];
                p.endX = next.x;
                p.endY = next.y;
                p.progress = 0;
            } else {
                return false; 
            }
        }
        
        const x = p.startX + (p.endX - p.startX) * p.progress;
        const y = p.startY + (p.endY - p.startY) * p.progress;
        
        ctx.beginPath();
        ctx.arc(x, y, 2.5, 0, Math.PI * 2);
        ctx.fillStyle = "#0d6efd";
        ctx.fill();
        
        return true;
    });

    requestAnimationFrame(drawFullNetwork);
}


const stepCanvas = document.getElementById('stepNet');
const stepCtx = stepCanvas.getContext('2d');
const stepBtn = document.getElementById('stepBtn');
const statusEl = document.getElementById('status');

const stepLayers = [
    { count: 2, label: "Input" },
    { count: 3, label: "Hidden" },
    { count: 2, label: "Output" }
];

let stepPositions = [];
let currentStep = -1;
let dashOffset = 0;

const steps = [
    { text: "1. Forward Pass: Input to Hidden", color: "#0d6efd", type: "forward", layer: 0 },
    { text: "2. Forward Pass: Hidden to Output", color: "#0d6efd", type: "forward", layer: 1 },
    { text: "3. Loss Calculation: Compare with true labels", color: "#ffc107", type: "none", layer: 2 },
    { text: "4. Backpropagation: Output to Hidden", color: "#dc3545", type: "backward", layer: 1 },
    { text: "5. Backpropagation: Hidden to Input", color: "#dc3545", type: "backward", layer: 0 },
    { text: "6. Optimizer Step: Weights updated", color: "#198754", type: "none", layer: 0 }
];

function setupStepNetwork() {
    const w = stepCanvas.width;
    const h = stepCanvas.height;
    const spacing = w / (stepLayers.length + 1);
    
    stepPositions = stepLayers.map((layer, i) => {
        const x = spacing * (i + 1);
        const ySpacing = h / (layer.count + 1);
        return Array.from({ length: layer.count }, (_, j) => ({
            x: x,
            y: ySpacing * (j + 1)
        }));
    });
}

function drawStepNetwork() {
    const ctx = stepCtx;
    ctx.clearRect(0, 0, stepCanvas.width, stepCanvas.height);
    
    ctx.strokeStyle = "#bbb";
    ctx.lineWidth = 1.5;
    for (let i = 0; i < stepPositions.length - 1; i++) {
        for (const n1 of stepPositions[i]) {
            for (const n2 of stepPositions[i + 1]) {
                ctx.beginPath();
                ctx.moveTo(n1.x, n1.y);
                ctx.lineTo(n2.x, n2.y);
                ctx.stroke();
            }
        }
    }

    if (currentStep >= 0 && currentStep < steps.length) {
        const step = steps[currentStep];
        if (step.type !== "none") {
            let from, to;
            
            if (step.type === "forward") {
                from = stepPositions[step.layer];
                to = stepPositions[step.layer + 1];
            } else { 
                from = stepPositions[step.layer + 1];
                to = stepPositions[step.layer];
            }

            ctx.strokeStyle = step.color;
            ctx.lineWidth = 3;
            ctx.setLineDash([6, 6]);
            ctx.lineDashOffset = -dashOffset;

            for (const n1 of from) {
                for (const n2 of to) {
                    ctx.beginPath();
                    ctx.moveTo(n1.x, n1.y);
                    ctx.lineTo(n2.x, n2.y);
                    ctx.stroke();
                }
            }
            ctx.setLineDash([]);
        }
    }

    ctx.fillStyle = "#333";
    ctx.font = "12px sans-serif";
    ctx.textAlign = "center";
    
    stepLayers.forEach((layer, i) => {
        ctx.fillText(layer.label, stepPositions[i][0].x, stepCanvas.height - 10);
        for (const pos of stepPositions[i]) {
            ctx.beginPath();
            ctx.arc(pos.x, pos.y, 8, 0, Math.PI * 2);
            ctx.fillStyle = "#fff";
            ctx.strokeStyle = "#333";
            ctx.lineWidth = 2;
            ctx.fill();
            ctx.stroke();
            
            if (currentStep >= 0 && currentStep < steps.length) {
                const step = steps[currentStep];
                let isActive = false;
                
                if (step.type === "forward" && i === step.layer + 1) isActive = true;
                if (step.type === "backward" && i === step.layer) isActive = true;
                if (step.type === "none" && step.text.includes("Loss") && i === 2) isActive = true;
                if (step.type === "none" && step.text.includes("Optimizer")) isActive = true;
                
                if (isActive) {
                    ctx.beginPath();
                    ctx.arc(pos.x, pos.y, 12, 0, Math.PI * 2);
                    ctx.strokeStyle = step.color;
                    ctx.lineWidth = 2;
                    ctx.stroke();
                }
            }
        }
    });
}

function animateStepLoop() {
    dashOffset += 0.5;
    drawStepNetwork();
    requestAnimationFrame(animateStepLoop);
}

stepBtn.addEventListener('click', () => {
    currentStep++;
    if (currentStep >= steps.length) {
        currentStep = 0;
    }
    
    const step = steps[currentStep];
    statusEl.textContent = step.text;
    statusEl.style.color = step.color;
});

setupFullNetwork();
setupStepNetwork();
drawFullNetwork();
animateStepLoop();
