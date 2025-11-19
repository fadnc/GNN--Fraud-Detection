// =======================
//  Neon Graph Visualization
// =======================

let svg, g, simulation;
let width, height;
let transform = d3.zoomIdentity;

// Colors (neon theme)
const COLOR_SAFE = "#00eaff";
const COLOR_MED  = "#a020f0";
const COLOR_HIGH = "#ff00ff";

// Suspicious threshold
const SUSPICIOUS_EDGE = 0.50;


// ============================
// INIT SVG CANVAS
// ============================
function initSVG() {
    const container = document.getElementById("graphContainer");
    width = container.clientWidth;
    height = container.clientHeight;

    // Clear old SVG
    d3.select("#graphContainer").selectAll("*").remove();

    svg = d3.select("#graphContainer")
        .append("svg")
        .attr("width", width)
        .attr("height", height);

    // Add gradient definitions for better visuals
    const defs = svg.append("defs");
    
    // Glow filters
    const glowFilter = defs.append("filter")
        .attr("id", "glow")
        .attr("x", "-50%")
        .attr("y", "-50%")
        .attr("width", "200%")
        .attr("height", "200%");
    
    glowFilter.append("feGaussianBlur")
        .attr("stdDeviation", "3")
        .attr("result", "coloredBlur");
    
    const feMerge = glowFilter.append("feMerge");
    feMerge.append("feMergeNode").attr("in", "coloredBlur");
    feMerge.append("feMergeNode").attr("in", "SourceGraphic");

    g = svg.append("g");

    setupZoom(svg, g);
}


// ============================
//  ZOOM + PAN (FIXED)
// ============================
function setupZoom(svg, g) {
    let isPanning = false;
    
    const zoom = d3.zoom()
        .scaleExtent([0.1, 5])
        .on("start", function(event) {
            if (event.sourceEvent && event.sourceEvent.type === "mousedown") {
                isPanning = true;
                svg.style("cursor", "grabbing");
            }
        })
        .on("zoom", function(event) {
            transform = event.transform;
            g.attr("transform", transform);
        })
        .on("end", function(event) {
            isPanning = false;
            svg.style("cursor", "grab");
        });

    svg.call(zoom);
    svg.style("cursor", "grab");
    
    // Disable double-click zoom
    svg.on("dblclick.zoom", null);
    
    // Reset view button
    svg.on("contextmenu", function(event) {
        event.preventDefault();
        svg.transition()
            .duration(750)
            .call(zoom.transform, d3.zoomIdentity);
    });
}


// ============================
//  Fetch Graph
// ============================
async function loadGraph() {
    const n = document.getElementById("nodesInput").value;
    const e = document.getElementById("edgesInput").value;

    showToast("Loading graph...");

    try {
        const res = await fetch(`/graph?nodes=${n}&edges=${e}`);
        
        if (!res.ok) {
            throw new Error(`HTTP error! status: ${res.status}`);
        }
        
        const graph = await res.json();
        hideToast();
        renderGraph(graph);
        
    } catch (error) {
        console.error('Error loading graph:', error);
        showToast("Error loading graph. Please try again.");
        setTimeout(hideToast, 3000);
    }
}


// ============================
//  Render the Graph (IMPROVED)
// ============================
function renderGraph(graph) {
    initSVG();

    if (!graph.nodes || !graph.edges) {
        console.error('Invalid graph data');
        return;
    }

    // Create a map for quick node lookup
    const nodeMap = new Map(graph.nodes.map(d => [d.id, d]));

    // ----- ARROW MARKERS -----
    const defs = svg.select("defs");
    
    // Arrow for edges
    defs.append("marker")
        .attr("id", "arrowhead")
        .attr("viewBox", "0 -5 10 10")
        .attr("refX", 20)
        .attr("refY", 0)
        .attr("markerWidth", 6)
        .attr("markerHeight", 6)
        .attr("orient", "auto")
        .append("path")
        .attr("d", "M0,-5L10,0L0,5")
        .attr("fill", COLOR_SAFE)
        .attr("opacity", 0.6);

    defs.append("marker")
        .attr("id", "arrowhead-suspicious")
        .attr("viewBox", "0 -5 10 10")
        .attr("refX", 20)
        .attr("refY", 0)
        .attr("markerWidth", 6)
        .attr("markerHeight", 6)
        .attr("orient", "auto")
        .append("path")
        .attr("d", "M0,-5L10,0L0,5")
        .attr("fill", COLOR_HIGH)
        .attr("opacity", 0.8);

    // ----- LINKS -----
    const link = g.append("g")
        .attr("class", "links")
        .selectAll("line")
        .data(graph.edges)
        .enter()
        .append("line")
        .attr("class", d => d.is_suspicious ? "link suspicious" : "link safe")
        .style("stroke", d => d.is_suspicious ? COLOR_HIGH : COLOR_SAFE)
        .style("stroke-width", d => d.is_suspicious ? 2.5 : 1.2)
        .style("opacity", d => d.is_suspicious ? 0.7 : 0.4)
        .attr("marker-end", d => d.is_suspicious ? "url(#arrowhead-suspicious)" : "url(#arrowhead)");


    // ----- NODE GROUPS -----
    const nodeGroup = g.append("g")
        .attr("class", "nodes")
        .selectAll("g")
        .data(graph.nodes)
        .enter()
        .append("g")
        .attr("class", "node-group");

    // Node circles
    const node = nodeGroup
        .append("circle")
        .attr("r", d => {
            if (d.is_suspicious) return 8;
            if (d.risk_score > 30) return 6;
            return 5;
        })
        .style("fill", d => {
            if (d.is_suspicious) return COLOR_HIGH;
            if (d.risk_score > 30) return COLOR_MED;
            return COLOR_SAFE;
        })
        .style("stroke", "#ffffff")
        .style("stroke-width", d => d.is_suspicious ? 2 : 1)
        .style("stroke-opacity", d => d.is_suspicious ? 0.9 : 0.5)
        .style("filter", "url(#glow)")
        .style("cursor", "pointer")
        .on("click", (event, d) => {
            event.stopPropagation();
            loadNodeDetails(d.id);
        })
        .on("mouseover", function(event, d) {
            // Highlight node
            d3.select(this)
                .transition()
                .duration(200)
                .attr("r", d => {
                    if (d.is_suspicious) return 12;
                    if (d.risk_score > 30) return 9;
                    return 7;
                })
                .style("stroke-width", 3);
                
            // Show label
            d3.select(this.parentNode).select("text")
                .transition()
                .duration(200)
                .style("opacity", 1)
                .style("font-size", "11px");
                
            // Highlight connected edges
            link.style("opacity", e => {
                if (e.source.id === d.id || e.target.id === d.id) {
                    return e.is_suspicious ? 1 : 0.8;
                }
                return 0.1;
            })
            .style("stroke-width", e => {
                if (e.source.id === d.id || e.target.id === d.id) {
                    return e.is_suspicious ? 4 : 2.5;
                }
                return e.is_suspicious ? 2.5 : 1.2;
            });
            
            // Highlight connected nodes
            nodeGroup.selectAll("circle")
                .style("opacity", nd => {
                    if (nd.id === d.id) return 1;
                    const isConnected = graph.edges.some(e => 
                        (e.source.id === d.id && e.target.id === nd.id) ||
                        (e.target.id === d.id && e.source.id === nd.id)
                    );
                    return isConnected ? 1 : 0.3;
                });
        })
        .on("mouseout", function(event, d) {
            // Reset node
            d3.select(this)
                .transition()
                .duration(200)
                .attr("r", d => {
                    if (d.is_suspicious) return 8;
                    if (d.risk_score > 30) return 6;
                    return 5;
                })
                .style("stroke-width", d => d.is_suspicious ? 2 : 1);
            
            // Hide label
            d3.select(this.parentNode).select("text")
                .transition()
                .duration(200)
                .style("opacity", 0.7)
                .style("font-size", "9px");
                
            // Reset edges
            link.style("opacity", d => d.is_suspicious ? 0.7 : 0.4)
                .style("stroke-width", d => d.is_suspicious ? 2.5 : 1.2);
            
            // Reset nodes
            nodeGroup.selectAll("circle")
                .style("opacity", 1);
        });


    // ----- LABELS -----
    const label = nodeGroup
        .append("text")
        .text(d => d.name || d.id.split('_')[1])
        .attr("font-size", 9)
        .attr("font-family", "monospace")
        .attr("fill", "#ffffff")
        .attr("text-anchor", "middle")
        .attr("dy", -12)
        .style("opacity", 0.7)
        .style("pointer-events", "none")
        .style("user-select", "none")
        .style("text-shadow", "0 0 3px #000, 0 0 5px #000");


    // ============================
    //   Force Simulation (TUNED)
    // ============================
    simulation = d3.forceSimulation(graph.nodes)
        .force("link", d3.forceLink(graph.edges)
            .id(d => d.id)
            .distance(d => {
                // Different distances based on node types
                const source = nodeMap.get(d.source.id || d.source);
                const target = nodeMap.get(d.target.id || d.target);
                if (d.is_suspicious) return 120;
                return 100;
            })
            .strength(0.3)
        )
        .force("charge", d3.forceManyBody()
            .strength(d => {
                // Stronger repulsion for suspicious nodes
                if (d.is_suspicious) return -400;
                if (d.risk_score > 30) return -300;
                return -250;
            })
            .distanceMax(400)
        )
        .force("center", d3.forceCenter(width / 2, height / 2))
        .force("collision", d3.forceCollide()
            .radius(d => {
                if (d.is_suspicious) return 25;
                if (d.risk_score > 30) return 20;
                return 18;
            })
            .strength(0.8)
        )
        .force("x", d3.forceX(width / 2).strength(0.05))
        .force("y", d3.forceY(height / 2).strength(0.05))
        .alphaDecay(0.02)
        .on("tick", ticked);


    // ----- Node Drag (FIXED) -----
    nodeGroup.call(
        d3.drag()
            .on("start", (event, d) => {
                if (!event.active) simulation.alphaTarget(0.3).restart();
                d.fx = d.x;
                d.fy = d.y;
                d3.select(event.sourceEvent.target)
                    .style("cursor", "grabbing");
                // Stop zoom from interfering
                event.sourceEvent.stopPropagation();
            })
            .on("drag", (event, d) => {
                d.fx = event.x;
                d.fy = event.y;
            })
            .on("end", (event, d) => {
                if (!event.active) simulation.alphaTarget(0);
                // Keep node fixed on release, or comment out to let it float
                // d.fx = null;
                // d.fy = null;
                d3.select(event.sourceEvent.target)
                    .style("cursor", "pointer");
            })
    );

    // ============================
    // Tick Update
    // ============================
    function ticked() {
        link
            .attr("x1", d => d.source.x)
            .attr("y1", d => d.source.y)
            .attr("x2", d => d.target.x)
            .attr("y2", d => d.target.y);

        nodeGroup
            .attr("transform", d => `translate(${d.x},${d.y})`);
    }
}


// ============================
//  Node Details Panel
// ============================
async function loadNodeDetails(nodeId) {
    try {
        const res = await fetch(`/node_details?id=${nodeId}`);
        
        if (!res.ok) {
            throw new Error(`HTTP error! status: ${res.status}`);
        }
        
        const data = await res.json();

        const panel = document.getElementById("detailsPanel");
        panel.style.display = "block";

        // Format counterparties nicely
        const topCounterparties = Object.entries(data.top_counterparties || {})
            .sort((a, b) => b[1] - a[1])
            .slice(0, 10);

        const counterpartiesHTML = topCounterparties.length > 0
            ? topCounterparties.map(([id, count]) => 
                `<div style="display: flex; justify-content: space-between; padding: 4px 0; border-bottom: 1px solid #1e3a45;">
                    <span style="color: #00eaff;">${id}</span>
                    <span style="color: #a0d5ff;">${count} txns</span>
                </div>`
              ).join('')
            : '<p style="opacity: 0.6;">No counterparties found</p>';

        panel.innerHTML = `
            <div class="panel-header">
                <span>Node Details</span>
                <button onclick="closeDetails()" class="close-btn">✖</button>
            </div>

            <div class="panel-body">
                <h3>${nodeId}</h3>

                <div style="background: #0b141e; padding: 16px; border-radius: 8px; margin: 16px 0;">
                    <p style="margin: 8px 0;"><strong>Transactions:</strong> ${data.degree}</p>
                    <p style="margin: 8px 0;"><strong>Avg Amount:</strong> <span style="color: #00eaff;">$${data.summary.avg_amount.toFixed(2)}</span></p>
                    <p style="margin: 8px 0;"><strong>Risk Score:</strong> <span style="color: ${data.risk > 0.5 ? '#ff00ff' : data.risk > 0.3 ? '#a020f0' : '#00eaff'};">${(data.risk * 100).toFixed(1)}%</span></p>
                </div>

                <h4>Top Counterparties</h4>
                <div style="max-height: 400px; overflow-y: auto;">
                    ${counterpartiesHTML}
                </div>
            </div>
        `;
    } catch (error) {
        console.error('Error loading node details:', error);
        showToast("Error loading node details");
        setTimeout(hideToast, 3000);
    }
}

function closeDetails() {
    const panel = document.getElementById("detailsPanel");
    panel.style.display = "none";
}


// ============================
//   Toast Popup
// ============================
function showToast(msg) {
    const t = document.getElementById("toast");
    t.innerText = msg;
    t.classList.add('show');
}

function hideToast() {
    const t = document.getElementById("toast");
    t.classList.remove('show');
}


// ============================
// Keyboard shortcuts
// ============================
document.addEventListener('keydown', (e) => {
    if (e.key === 'Escape') {
        closeDetails();
    }
    
    // R to reset view
    if (e.key === 'r' || e.key === 'R') {
        const container = document.getElementById("graphContainer");
        const svg = d3.select(container).select("svg");
        const g = svg.select("g");
        
        svg.transition()
            .duration(750)
            .call(d3.zoom().transform, d3.zoomIdentity);
    }
});


// ============================
// Resize handler
// ============================
let resizeTimer;
window.addEventListener('resize', () => {
    clearTimeout(resizeTimer);
    resizeTimer = setTimeout(() => {
        if (simulation) {
            const container = document.getElementById("graphContainer");
            width = container.clientWidth;
            height = container.clientHeight;
            
            svg.attr("width", width).attr("height", height);
            simulation.force("center", d3.forceCenter(width / 2, height / 2));
            simulation.force("x", d3.forceX(width / 2).strength(0.05));
            simulation.force("y", d3.forceY(height / 2).strength(0.05));
            simulation.alpha(0.3).restart();
        }
    }, 250);
});


// ============================
// Load first graph on startup
// ============================
document.addEventListener("DOMContentLoaded", () => {
    initSVG();
    loadGraph();
});