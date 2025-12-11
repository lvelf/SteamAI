const input = document.getElementById("game-input");
const btn = document.getElementById("search-btn");
const statusEl = document.getElementById("status");
const table = document.getElementById("result-table");
const tbody = table.querySelector("tbody");

btn.addEventListener("click", async () => {
  const name = input.value.trim();
  if (!name) {
    statusEl.textContent = "Please enter a game name.";
    return;
  }

  statusEl.textContent = "Loading recommendations...";
  table.classList.add("hidden");
  tbody.innerHTML = "";

  try {
    const resp = await fetch(`/api/recommend?name=${encodeURIComponent(name)}`);
    const data = await resp.json();

    if (!resp.ok) {
      statusEl.textContent = data.error || "Error fetching recommendations.";
      return;
    }

    statusEl.textContent = `Center game: ${data.center.name} (appid=${data.center.appid})`;

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

input.addEventListener("keydown", (e) => {
  if (e.key === "Enter") {
    btn.click();
  }
});
