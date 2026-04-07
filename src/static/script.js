/* =========================
   GLOBAL STATE
========================= */
let chart;


/* =========================
   INIT ON LOAD
========================= */
window.onload = function () {
    initChart();
    loadHistory();
};


/* =========================
   SEND MESSAGE
========================= */
async function sendMessage() {
    const input = document.getElementById("user-input");
    const chatBox = document.getElementById("chat-box");

    const text = input.value.trim();
    if (!text) return;

    addMessage(text, "user");
    saveChat(text);
    loadHistory();

    input.value = "";

    const typingMsg = addTypingIndicator();

    try {
        const response = await fetch("/chat", {
            method: "POST",
            headers: {
                "Content-Type": "application/json"
            },
            body: JSON.stringify({ text: text })
        });

        const data = await response.json();

        chatBox.removeChild(typingMsg);

        let emotion = "neutral";
        let score = 0;

        if (data.interpreted_emotions && data.interpreted_emotions.length > 0) {
            emotion = data.interpreted_emotions[0][0];
            score = data.interpreted_emotions[0][1];
        }

        let trend = data.trend || "neutral";

        addBotMessage(data.response, emotion, trend);
        updateChart(score);

    } catch (error) {
        chatBox.removeChild(typingMsg);
        addMessage("Error connecting to server.", "bot");
        console.error(error);
    }
}


/* =========================
   USER MESSAGE
========================= */
function addMessage(text, sender) {
    const chatBox = document.getElementById("chat-box");

    const msg = document.createElement("div");
    msg.classList.add("message", sender);
    msg.innerText = text;

    chatBox.appendChild(msg);
    chatBox.scrollTop = chatBox.scrollHeight;

    return msg;
}


/* =========================
   BOT MESSAGE (FIXED)
========================= */
function addBotMessage(text, emotion, trend) {
    const chatBox = document.getElementById("chat-box");

    const msg = document.createElement("div");
    msg.classList.add("message", "bot");

    // 🔥 FIXED LOGIC
    const badge = document.createElement("span");
    badge.classList.add("emotion-badge", emotion);

    if (trend === "declining") {
        badge.innerText = "NEGATIVE";
    } else if (trend === "improving") {
        badge.innerText = "POSITIVE";
    } else {
        badge.innerText = emotion.toUpperCase();
    }

    // Trend indicator
    const trendEl = document.createElement("span");
    trendEl.classList.add("trend");
    trendEl.innerText = getTrendSymbol(trend);

    // Message text
    const content = document.createElement("div");
    content.innerText = text;

    msg.appendChild(badge);
    msg.appendChild(trendEl);
    msg.appendChild(content);

    chatBox.appendChild(msg);
    chatBox.scrollTop = chatBox.scrollHeight;
}


/* =========================
   TYPING ANIMATION
========================= */
function addTypingIndicator() {
    const chatBox = document.getElementById("chat-box");

    const msg = document.createElement("div");
    msg.classList.add("message", "bot");

    const dots = document.createElement("div");
    dots.classList.add("typing");

    dots.innerHTML = `
        <span></span>
        <span></span>
        <span></span>
    `;

    msg.appendChild(dots);
    chatBox.appendChild(msg);
    chatBox.scrollTop = chatBox.scrollHeight;

    return msg;
}


/* =========================
   TREND SYMBOL
========================= */
function getTrendSymbol(trend) {
    if (trend === "declining") return "⚠️ declining";
    if (trend === "improving") return "📈 improving";
    return "— stable";
}


/* =========================
   CHART INIT (FIXED)
========================= */
function initChart() {
    const ctx = document.getElementById("emotionChart").getContext("2d");

    chart = new Chart(ctx, {
        type: "line",
        data: {
            labels: [],
            datasets: [{
                label: "",   // 🔥 FIX: removed label
                data: [],
                borderColor: "rgba(200, 200, 200, 0.8)",
                borderWidth: 2,
                pointRadius: 2,
                tension: 0.3
            }]
        },
        options: {
            plugins: {
                legend: {
                    display: false   // 🔥 FIX: removes blue box
                }
            },
            scales: {
                x: {
                    ticks: { color: "#aaa" },
                    grid: { display: false }
                },
                y: {
                    ticks: { color: "#aaa" },
                    min: 0,
                    max: 1,
                    grid: { color: "rgba(255,255,255,0.05)" }
                }
            }
        }
    });
}


/* =========================
   UPDATE CHART
========================= */
function updateChart(score) {
    chart.data.labels.push(chart.data.labels.length + 1);
    chart.data.datasets[0].data.push(score);
    chart.update();
}


/* =========================
   SAVE CHAT HISTORY
========================= */
function saveChat(text) {
    let history = JSON.parse(localStorage.getItem("chatHistory")) || [];
    history.push(text);
    localStorage.setItem("chatHistory", JSON.stringify(history));
}


/* =========================
   LOAD CHAT HISTORY
========================= */
function loadHistory() {
    let history = JSON.parse(localStorage.getItem("chatHistory")) || [];

    const container = document.getElementById("chat-history");
    if (!container) return;

    container.innerHTML = "";

    history.slice(-10).forEach(msg => {
        const div = document.createElement("div");
        div.classList.add("chat-item");
        div.innerText = msg;
        container.appendChild(div);
    });
}


/* =========================
   ENTER KEY SUPPORT
========================= */
function handleKey(event) {
    if (event.key === "Enter") {
        sendMessage();
    }
}


/* =========================
   TOGGLE PANEL
========================= */
function togglePanel() {
    const panel = document.getElementById("side-panel");
    panel.classList.toggle("show");
}