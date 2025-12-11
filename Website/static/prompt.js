const promptInput = document.getElementById("prompt-input");
const promptBtn = document.getElementById("prompt-btn");
const statusEl = document.getElementById("prompt-status");
const table = document.getElementById("prompt-table");
const tbody = table.querySelector("tbody");

promptBtn.addEventListener("click", async () => {
  const text = promptInput.value.trim();
  if (!text) {
    statusEl.textContent = "Please enter a prompt.";
    return;
  }

  statusEl.textContent = "Searching...";
  table.classList.add("hidden");
  tbody.innerHTML = "";

  try {
    const resp = await fetch("/api/prompt_recommend", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ text }),
    });

    const raw = await resp.text();         
    console.log("prompt response raw:", raw);

    let data;
    try {
      data = JSON.parse(raw);              
    } catch (e) {
      console.error("JSON parse error:", e);
      statusEl.textContent = "Server JSON error.";
      return;
    }

    const modeLabel = {
      "appid": "based on AppID",
      "name": "based on existing game name",
      "semantic": "semantic search from prompt",
    }[data.mode] || data.mode;

    if (data.center) {
      statusEl.textContent =
        `Mode: ${modeLabel}. Center game: ${data.center.name} (appid=${data.center.appid}).`;
    } else {
      statusEl.textContent = `Mode: ${modeLabel}. No exact game found; showing games matching your description.`;
    }

    data.recommendations.forEach((row, idx) => {
      const tr = document.createElement("tr");
      tr.innerHTML = `
        <td>${idx + 1}</td>
        <td>${row.appid}</td>
        <td>${row.name}</td>
        <td>${row.similarity.toFixed(3)}</td>
      `;
      tbody.appendChild(tr);
    });

    table.classList.remove("hidden");
  } catch (err) {
    console.error(err);
    statusEl.textContent = "Network or server error.";
  }
});

