const input = document.getElementById("summary-input");
const btn = document.getElementById("summary-search-btn");
const statusEl = document.getElementById("summary-status");
const resultSection = document.getElementById("summary-result");
const titleEl = document.getElementById("summary-title");
const metaEl = document.getElementById("summary-meta");
const contentEl = document.getElementById("summary-content");

btn.addEventListener("click", async () => {
  const q = input.value.trim();
  if (!q) {
    statusEl.textContent = "Please enter a game name or appid.";
    resultSection.classList.add("hidden");
    return;
  }

  statusEl.textContent = "Loading summary...";
  resultSection.classList.add("hidden");
  contentEl.textContent = "";

  const params = new URLSearchParams();
  if (/^\d+$/.test(q)) {
    params.set("appid", q);
  } else {
    params.set("name", q);
  }

  try {
    const resp = await fetch(`/api/review_summary?${params.toString()}`);
    const data = await resp.json();

    if (!resp.ok) {
      statusEl.textContent = data.error || "Error fetching summary.";
      return;
    }

    statusEl.textContent = "";
    titleEl.textContent = `${data.name} (appid=${data.appid})`;

    metaEl.textContent = `Summarized from ${data.n_reviews_used} / ${data.n_reviews_total} reviews · Last updated: ${data.last_updated}`;

    contentEl.textContent = data.summary || "No summary text.";
    resultSection.classList.remove("hidden");
  } catch (err) {
    console.error(err);
    statusEl.textContent = "Network or server error.";
    resultSection.classList.add("hidden");
  }
});


input.addEventListener("keydown", (e) => {
  if (e.key === "Enter") {
    btn.click();
  }
});
