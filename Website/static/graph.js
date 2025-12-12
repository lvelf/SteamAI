// ======================= Graph: inputs & filters =======================
const gInput = document.getElementById("game-input");
const gBtn = document.getElementById("draw-btn");
const gStatus = document.getElementById("status");
const gContainer = document.getElementById("graph");

const minYearInput = document.getElementById("min-year");
const maxYearInput = document.getElementById("max-year");
const genresInput = document.getElementById("genres");
const isFreeInput = document.getElementById("is-free");
const isMultiInput = document.getElementById("multi-area");
const minPosRatioInput = document.getElementById("min-positive-ratio");
const minReviewCountInput = document.getElementById("min-review-count");
const maxPriceInput = document.getElementById("max-price");

// ======================= Tabs: Graph <-> Hot =======================
const tabGraph = document.getElementById("tab-graph");
const tabHot = document.getElementById("tab-hot");
const graphPanel = document.getElementById("graph-panel");
const hotPanel = document.getElementById("hot-panel");

function showGraph() {
  tabGraph?.classList.add("active");
  tabHot?.classList.remove("active");
  graphPanel?.classList.remove("hidden");
  hotPanel?.classList.add("hidden");
}

function showHot() {
  tabHot?.classList.add("active");
  tabGraph?.classList.remove("active");
  hotPanel?.classList.remove("hidden");
  graphPanel?.classList.add("hidden");
}

tabGraph?.addEventListener("click", showGraph);
tabHot?.addEventListener("click", showHot);

// ======================= Hot UI (front-end first: mock) =======================
const hotYearInput = document.getElementById("hot-year");
const hotBtn = document.getElementById("hot-btn");
const hotStatus = document.getElementById("hot-status");
const hotTable = document.getElementById("hot-table");
const hotTbody = hotTable?.querySelector("tbody");

(function initHotDefaultYear() {
  if (!hotYearInput) return;
  // default to last year
  hotYearInput.value = String(new Date().getFullYear() - 1);
})();

function renderHotTable(rows) {
  if (!hotTbody || !hotTable) return;
  hotTbody.innerHTML = "";

  rows.forEach((r, idx) => {
    const pop =
      typeof r.popularity === "number"
        ? r.popularity.toFixed(3)
        : (r.popularity ?? "");

    const pr =
      typeof r.positive_ratio === "number"
        ? r.positive_ratio.toFixed(3)
        : (r.positive_ratio ?? "");

    const tr = document.createElement("tr");
    tr.innerHTML = `
      <td>${idx + 1}</td>
      <td>${r.appid ?? ""}</td>
      <td>${r.name ?? ""}</td>
      <td>${pop}</td>
      <td>${r.review_count ?? ""}</td>
      <td>${pr}</td>
    `;
    hotTbody.appendChild(tr);
  });

  hotTable.classList.remove("hidden");
}

const hotChart = document.getElementById("hot-chart");

function drawHotLineChart(rows) {
  if (!hotChart) return;

  hotChart.innerHTML = "";
  hotChart.classList.remove("hidden");

  // top10
  const data = (rows || []).slice(0, 10).map((r, i) => ({
    rank: i + 1,
    popularity: Number(r.popularity ?? 0),
    name: r.name ?? "",
  }));

  if (!data.length) {
    hotChart.textContent = "No chart data.";
    return;
  }

  const width = 850; 
  const height = 320;
  const margin = { top: 20, right: 80, bottom: 45, left: 60 };
  const [yMinRaw, yMaxRaw] = d3.extent(data, d => d.popularity);
  const span = (yMaxRaw - yMinRaw) || 0;
  const pad = Math.max(span * 0.2, 0.01);

  const svg = d3.select(hotChart)
    .append("svg")
    .attr("width", width)
    .attr("height", height);

  const x = d3.scaleLinear()
    .domain([1, d3.max(data, d => d.rank)])
    .range([margin.left, width - margin.right]);

  const y = d3.scaleLinear()
  .domain([yMinRaw - pad, yMaxRaw + pad])
  .nice()
  .range([height - margin.bottom, margin.top]);

  // axes
  svg.append("g")
    .attr("transform", `translate(0,${height - margin.bottom})`)
    .call(d3.axisBottom(x).ticks(data.length).tickFormat(d3.format("d")));

  svg.append("g")
    .attr("transform", `translate(${margin.left},0)`)
    .call(d3.axisLeft(y));

  // axis labels
  svg.append("text")
    .attr("x", (width) / 2)
    .attr("y", height - 10)
    .attr("text-anchor", "middle")
    .attr("fill", "#9ca3af")
    .text("Rank");

  svg.append("text")
    .attr("transform", "rotate(-90)")
    .attr("x", -height / 2)
    .attr("y", 16)
    .attr("text-anchor", "middle")
    .attr("fill", "#9ca3af")
    .text("Popularity");

  // line
  const line = d3.line()
    .x(d => x(d.rank))
    .y(d => y(d.popularity));

  svg.append("path")
    .datum(data)
    .attr("fill", "none")
    .attr("stroke", "#22c55e")
    .attr("stroke-width", 2.5)
    .attr("d", line);

  // points
  const pts = svg.append("g")
    .selectAll("circle")
    .data(data)
    .enter()
    .append("circle")
    .attr("cx", d => x(d.rank))
    .attr("cy", d => y(d.popularity))
    .attr("r", 4)
    .attr("fill", "#22c55e");

  // rank labels near points
  svg.append("g")
    .selectAll("text.rank-label")
    .data(data)
    .enter()
    .append("text")
    .attr("class", "rank-label")
    .attr("x", d => x(d.rank) + 6)
    .attr("y", d => y(d.popularity) - 6)
    .attr("font-size", 10)
    .attr("fill", "#e5e7eb")
    .text(d => `#${d.rank}`);

  // tooltip (native)
  pts.append("title")
    .text(d => `#${d.rank} ${d.name}\npopularity=${d.popularity.toFixed(3)}`);
}



async function fetchHotTop10(year) {
  const resp = await fetch(`/api/year_hot?year=${encodeURIComponent(year)}`);
  const data = await resp.json();
  if (!resp.ok) throw new Error(data.error || "Error fetching hot");
  return data.rows || [];
}

hotBtn?.addEventListener("click", async () => {
  const year = hotYearInput?.value?.trim();
  if (!year) {
    if (hotStatus) hotStatus.textContent = "Please enter a year.";
    return;
  }

  if (hotStatus) hotStatus.textContent = `Loading Top 10 for ${year}...`;
  hotTable?.classList.add("hidden");

  hotChart?.classList.add("hidden");
  if (hotChart) hotChart.innerHTML = "";

  try {
    const rows = await fetchHotTop10(year);
    if (hotStatus) hotStatus.textContent = `Top 10 games in ${year}`;
    renderHotTable(rows);
    drawHotLineChart(rows);
  } catch (e) {
    console.error(e);
    if (hotStatus) hotStatus.textContent = "Failed to load hot.";
    hotChart?.classList.add("hidden");
  }
});

hotYearInput?.addEventListener("keydown", (e) => {
  if (e.key === "Enter") {
    e.preventDefault();
    hotBtn?.click();
  }
});

// ======================= Graph fetch + render =======================
gBtn?.addEventListener("click", async () => {
  const name = gInput?.value?.trim();
  if (!name) {
    if (gStatus) gStatus.textContent = "Please enter a game name.";
    return;
  }

  if (gStatus) gStatus.textContent = "Loading graph data...";
  if (gContainer) gContainer.innerHTML = "";

  const params = new URLSearchParams();
  params.set("name", name);

  const minYear = minYearInput?.value?.trim();
  const maxYear = maxYearInput?.value?.trim();
  const genres = genresInput?.value?.trim();
  const isFree = isFreeInput?.checked;
  const isMulti = isMultiInput?.checked;
  const minPosRatio = minPosRatioInput?.value?.trim();
  const minReviewCount = minReviewCountInput?.value?.trim();
  const maxPrice = maxPriceInput?.value?.trim();

  if (minYear) params.set("min_year", minYear);
  if (maxYear) params.set("max_year", maxYear);
  if (genres) params.set("genres", genres); // "action, rpg"
  if (isFree) params.set("is_free", "true");
  if (isMulti) params.set("is_multi", "true");
  if (minPosRatio) params.set("min_positive_ratio", minPosRatio);
  if (minReviewCount) params.set("min_review_count", minReviewCount);
  if (maxPrice) params.set("max_price", maxPrice);

  try {
    const resp = await fetch(`/api/graph?${params.toString()}`);
    const data = await resp.json();

    if (!resp.ok) {
      if (gStatus) gStatus.textContent = data.error || "Error fetching graph data.";
      return;
    }

    if (gStatus) gStatus.textContent = `Center game: ${data.center?.name || name}`;
    drawForceGraph(data.nodes, data.links);
  } catch (err) {
    console.error(err);
    if (gStatus) gStatus.textContent = "Network or server error.";
  }
});

function drawForceGraph(nodes, links) {
  const width = 800;
  const height = 650;

  const svg = d3.select("#graph")
    .append("svg")
    .attr("width", width)
    .attr("height", height);

  const color = (d) => (d.group === 0 ? "#f97316" : "#3b82f6");

  const simulation = d3.forceSimulation(nodes)
    .force("link", d3.forceLink(links).id(d => d.id).distance(d => 200 * (1 - d.value)))
    .force("charge", d3.forceManyBody().strength(-450))
    .force("center", d3.forceCenter(width / 2, height / 2));

  const link = svg.append("g")
    .attr("stroke", "#4b5563")
    .attr("stroke-opacity", 0.6)
    .selectAll("line")
    .data(links)
    .enter().append("line")
    .attr("stroke-width", d => 2 + 4 * d.value);

  const node = svg.append("g")
    .attr("stroke", "#020617")
    .attr("stroke-width", 1.5)
    .selectAll("circle")
    .data(nodes)
    .enter().append("circle")
    .attr("r", d => (d.group === 0 ? 10 : 6))
    .attr("fill", color)
    .call(drag(simulation));

  const label = svg.append("g")
    .attr("font-size", 10)
    .attr("fill", "#e5e7eb")
    .selectAll("text")
    .data(nodes)
    .enter().append("text")
    .text(d => (typeof d.label === "string" ? d.label : (d.label?.name ?? String(d.label))))
    .attr("dy", -12);

  simulation.on("tick", () => {
    link
      .attr("x1", d => d.source.x)
      .attr("y1", d => d.source.y)
      .attr("x2", d => d.target.x)
      .attr("y2", d => d.target.y);

    node
      .attr("cx", d => d.x)
      .attr("cy", d => d.y);

    label
      .attr("x", d => d.x)
      .attr("y", d => d.y);
  });

  function drag(sim) {
    function dragstarted(event, d) {
      if (!event.active) sim.alphaTarget(0.3).restart();
      d.fx = d.x;
      d.fy = d.y;
    }
    function dragged(event, d) {
      d.fx = event.x;
      d.fy = event.y;
    }
    function dragended(event, d) {
      if (!event.active) sim.alphaTarget(0);
      d.fx = null;
      d.fy = null;
    }
    return d3.drag()
      .on("start", dragstarted)
      .on("drag", dragged)
      .on("end", dragended);
  }
}

gInput?.addEventListener("keydown", (e) => {
  if (e.key === "Enter") {
    e.preventDefault();
    gBtn?.click();
  }
});
