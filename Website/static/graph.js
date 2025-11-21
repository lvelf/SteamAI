const gInput = document.getElementById("game-input");
const gBtn = document.getElementById("draw-btn");
const gStatus = document.getElementById("status");
const gContainer = document.getElementById("graph");

gBtn.addEventListener("click", async () => {
  const name = gInput.value.trim();
  if (!name) {
    gStatus.textContent = "Please enter a game name.";
    return;
  }

  gStatus.textContent = "Loading graph data...";
  gContainer.innerHTML = "";

  try {
    const resp = await fetch(`/api/graph?name=${encodeURIComponent(name)}`);
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
    .text(d => d.label)
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
