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


gBtn.addEventListener("click", async () => {
  const name = gInput.value.trim();
  if (!name) {
    gStatus.textContent = "Please enter a game name.";
    return;
  }

  gStatus.textContent = "Loading graph data...";
  gContainer.innerHTML = "";

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
    const resp = await fetch(`/api/graph?${params.toString()}`);
    const data = await resp.json();

    if (!resp.ok) {
      gStatus.textContent = data.error || "Error fetching graph data.";
      return;
    }

    gStatus.textContent = `Center game: ${data.center?.name || name}`;

    drawForceGraph(data.nodes, data.links);
  } catch (err) {
    console.error(err);
    gStatus.textContent = "Network or server error.";
  }
});

function drawForceGraph(nodes, links) {
  const width = 800;
  const height = 650;

  const svg = d3.select("#graph")
    .append("svg")
    .attr("width", width)
    .attr("height", height);

  const color = d => d.group === 0 ? "#f97316" : "#3b82f6";

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
    .attr("r", d => d.group === 0 ? 10 : 6)
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

gInput.addEventListener("keydown", (e) => {
  if (e.key === "Enter") {
    e.preventDefault();
    gBtn.click();
  }
});






