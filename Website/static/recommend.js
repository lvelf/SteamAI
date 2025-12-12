const input = document.getElementById("game-input");
const btn = document.getElementById("search-btn");
const statusEl = document.getElementById("status");
const table = document.getElementById("result-table");
const tbody = table.querySelector("tbody");

const minYearInput = document.getElementById("min-year");
const maxYearInput = document.getElementById("max-year");
const genresInput = document.getElementById("genres");
const isFreeInput = document.getElementById("is-free");
const isMultiInput = document.getElementById("multi-area");
const minPosRatioInput = document.getElementById("min-positive-ratio");
const minReviewCountInput = document.getElementById("min-review-count");
const maxPriceInput = document.getElementById("max-price");



btn.addEventListener("click", async () => {
  const name = input.value.trim();
  if (!name) {
    statusEl.textContent = "Please enter a game name.";
    return;
  }

  statusEl.textContent = "Loading recommendations...";
  table.classList.add("hidden");
  tbody.innerHTML = "";

  const params = new URLSearchParams();
  params.set("name", name);

  const minYear = minYearInput?.value.trim();
  const maxYear = maxYearInput?.value.trim();
  const genres = genresInput?.value.trim();
  const isFree = isFreeInput?.checked;
  const isMulti = isMultiInput?.checked;
  const minPosRatio = minPosRatioInput?.value.trim();
  const minReviewCount = minReviewCountInput?.value.trim();
  const maxPrice = maxPriceInput?.value.trim();

  if (minYear) params.set("min_year", minYear);
  if (maxYear) params.set("max_year", maxYear);
  if (genres) params.set("genres", genres);          // "action, rpg"
  if (isFree) params.set("is_free", "true");
  if (isMulti) params.set("is_multi", "true");
  if (minPosRatio) params.set("min_positive_ratio", minPosRatio);
  if (minReviewCount) params.set("min_review_count", minReviewCount);
  if (maxPrice) params.set("max_price", maxPrice);

  try {
    const resp = await fetch(`/api/recommend?${params.toString()}`);
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
