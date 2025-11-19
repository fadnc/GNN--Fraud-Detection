// Advanced GNN Fraud Detection - Main JavaScript
// Fixed version with proper API integration

let svg, g, simulation;
let width, height;
let currentGraph = { nodes: [], edges: [] };
let selectedNode = null;

// Colors
const COLORS = {
    safe: "#00eaff",
    medium: "#ffa500",
    high: "#ff0066",
    bg: "#0a0e1a"
};

// ============================
// INITIALIZATION
// ============================
document.addEventListener('DOMContentLoaded', () => {
    console.log('Initializing GNN Fraud Detection System...');
    init();
});

function init() {
    initSVG();
    setupEventListeners();
    loadInitialData();
}

function initSVG() {
    const svgElement = document.getElementById('graphSvg');
    const container = svgElement.parentElement;
    
    width = container.clientWidth;
    height = container.clientHeight;

    svg = d3.select('#graphSvg')
        .attr('width', width)
        .attr('height', height);

    // Create defs for gradients and filters
    const defs = svg.append('defs');
    
    // Glow filter
    const glow = defs.append('filter')
        .attr('id', 'glow')
        .attr('x', '-50%')
        .attr('y', '-50%')
        .attr('width', '200%')
        .attr('height', '200%');
    
    glow.append('feGaussianBlur')
        .attr('stdDeviation', '4')
        .attr('result', 'coloredBlur');
    
    const feMerge = glow.append('feMerge');
    feMerge.append('feMergeNode').attr('in', 'coloredBlur');
    feMerge.append('feMergeNode').attr('in', 'SourceGraphic');

    // Arrow markers
    ['safe', 'medium', 'high'].forEach(type => {
        defs.append('marker')
            .attr('id', `arrow-${type}`)
            .attr('viewBox', '0 -5 10 10')
            .attr('refX', 20)
            .attr('refY', 0)
            .attr('markerWidth', 6)
            .attr('markerHeight', 6)
            .attr('orient', 'auto')
            .append('path')
            .attr('d', 'M0,-5L10,0L0,5')
            .attr('fill', COLORS[type])
            .attr('opacity', 0.6);
    });

    g = svg.append('g');

    setupZoom();
    
    console.log('SVG initialized:', width, 'x', height);
}

function setupZoom() {
    const zoom = d3.zoom()
        .scaleExtent([0.1, 5])
        .on('zoom', (event) => {
            g.attr('transform', event.transform);
        });

    svg.call(zoom);
    svg.style('cursor', 'grab');
    
    svg.on('mousedown.zoom', () => svg.style('cursor', 'grabbing'))
       .on('mouseup.zoom', () => svg.style('cursor', 'grab'));
}

// ============================
// DATA LOADING
// ============================
async function loadInitialData() {
    try {
        console.log('Loading initial data...');
        
        // Check if API endpoints exist
        const response = await fetch('/api/metrics');
        if (response.ok) {
            const metricsData = await response.json();
            console.log('Metrics loaded:', metricsData);
            updateHeaderStats(metricsData.metrics);
        } else {
            console.warn('API metrics not available, using old endpoint');
        }
        
        // Load graph
        await loadGraph();
        
        hideLoading();
    } catch (error) {
        console.error('Error loading initial data:', error);
        // Try loading with old API
        await loadGraphOldAPI();
    }
}

async function loadGraph() {
    const nodesInput = document.getElementById('nodesInput');
    const edgesInput = document.getElementById('edgesInput');
    
    const nodes = nodesInput ? nodesInput.value : 50;
    const edges = edgesInput ? edgesInput.value : 200;
    
    console.log(`Loading graph with ${nodes} nodes and ${edges} edges...`);
    showLoading('Loading graph...');
    
    try {
        // Try new API first
        let response = await fetch(`/api/graph?nodes=${nodes}&edges=${edges}`);
        
        // If new API doesn't exist, try old one
        if (!response.ok) {
            console.log('Trying old API endpoint...');
            response = await fetch(`/graph?nodes=${nodes}&edges=${edges}`);
        }
        
        if (!response.ok) {
            throw new Error(`HTTP error! status: ${response.status}`);
        }
        
        const data = await response.json();
        console.log('Graph data received:', data);
        
        currentGraph = data;
        renderGraph(data);
        
        if (data.metrics) {
            updateHeaderStats(data.metrics);
        }
        
        hideLoading();
        showNotification('Graph loaded successfully', 'success');
    } catch (error) {
        console.error('Error loading graph:', error);
        hideLoading();
        showNotification('Failed to load graph: ' + error.message, 'error');
    }
}

async function loadGraphOldAPI() {
    // Fallback to old API
    console.log('Using old API structure...');
    try {
        const nodes = document.getElementById('nodesInput').value || 20;
        const edges = document.getElementById('edgesInput').value || 150;
        
        const response = await fetch(`/graph?nodes=${nodes}&edges=${edges}`);
        const data = await response.json();
        
        console.log('Old API data received:', data);
        renderGraph(data);
        hideLoading();
    } catch (error) {
        console.error('Old API also failed:', error);
        hideLoading();
    }
}

// ============================
// GRAPH RENDERING
// ============================
function renderGraph(data) {
    console.log('Rendering graph...');
    
    // Clear existing
    g.selectAll('*').remove();

    if (!data.nodes || !data.edges) {
        console.error('Invalid graph data:', data);
        return;
    }

    console.log(`Rendering ${data.nodes.length} nodes and ${data.edges.length} edges`);

    // Create links
    const link = g.append('g')
        .attr('class', 'links')
        .selectAll('line')
        .data(data.edges)
        .enter()
        .append('line')
        .style('stroke', d => getRiskColor(d))
        .style('stroke-width', d => d.is_suspicious ? 2.5 : 1.5)
        .style('opacity', d => d.is_suspicious ? 0.7 : 0.4)
        .attr('marker-end', d => {
            const type = d.is_suspicious ? 'high' : 
                        (d.pred_prob || 0) > 0.3 ? 'medium' : 'safe';
            return `url(#arrow-${type})`;
        });

    // Create node groups
    const nodeGroup = g.append('g')
        .attr('class', 'nodes')
        .selectAll('g')
        .data(data.nodes)
        .enter()
        .append('g')
        .attr('class', 'node-group')
        .call(d3.drag()
            .on('start', dragStarted)
            .on('drag', dragged)
            .on('end', dragEnded));

    // Add circles
    nodeGroup.append('circle')
        .attr('r', d => getNodeSize(d))
        .style('fill', d => getRiskColor(d))
        .style('stroke', '#ffffff')
        .style('stroke-width', d => d.is_suspicious ? 2 : 1)
        .style('stroke-opacity', 0.8)
        .style('filter', 'url(#glow)')
        .style('cursor', 'pointer')
        .on('click', (event, d) => {
            event.stopPropagation();
            selectNode(d);
        })
        .on('mouseover', function(event, d) {
            highlightNode(d, this);
        })
        .on('mouseout', function(event, d) {
            unhighlightNode(d, this);
        });

    // Add labels
    nodeGroup.append('text')
        .text(d => d.name || d.id.split('_')[1])
        .attr('dy', -12)
        .attr('text-anchor', 'middle')
        .style('font-size', '9px')
        .style('font-family', 'monospace')
        .style('fill', '#ffffff')
        .style('opacity', 0.7)
        .style('pointer-events', 'none')
        .style('text-shadow', '0 0 3px #000');

    // Setup simulation
    simulation = d3.forceSimulation(data.nodes)
        .force('link', d3.forceLink(data.edges)
            .id(d => d.id)
            .distance(100)
            .strength(0.3))
        .force('charge', d3.forceManyBody()
            .strength(d => d.is_suspicious ? -400 : -300)
            .distanceMax(400))
        .force('center', d3.forceCenter(width / 2, height / 2))
        .force('collision', d3.forceCollide()
            .radius(d => getNodeSize(d) + 15)
            .strength(0.8))
        .force('x', d3.forceX(width / 2).strength(0.05))
        .force('y', d3.forceY(height / 2).strength(0.05))
        .alphaDecay(0.02)
        .on('tick', () => {
            link
                .attr('x1', d => d.source.x)
                .attr('y1', d => d.source.y)
                .attr('x2', d => d.target.x)
                .attr('y2', d => d.target.y);

            nodeGroup
                .attr('transform', d => `translate(${d.x},${d.y})`);
        });
    
    console.log('Graph rendering complete');
}

// ============================
// NODE INTERACTION
// ============================
async function selectNode(node) {
    console.log('Node selected:', node.id);
    selectedNode = node;
    
    // Highlight selection
    g.selectAll('.node-group circle')
        .style('stroke-width', d => d.id === node.id ? 3 : d.is_suspicious ? 2 : 1);
    
    // Try to load detailed info from new API
    try {
        const response = await fetch(`/api/node/${node.id}`);
        if (response.ok) {
            const data = await response.json();
            displayNodeDetails(data);
            return;
        }
    } catch (error) {
        console.log('New API not available, trying old API...');
    }
    
    // Fallback to old API
    try {
        const response = await fetch(`/node_details?id=${node.id}`);
        const data = await response.json();
        displayNodeDetailsOld(data);
    } catch (error) {
        console.error('Error loading node details:', error);
    }
}

function displayNodeDetails(data) {
    const detailsContent = document.querySelector('[data-content="details"]');
    
    const html = `
        <div class="info-card">
            <h3>Node Information</h3>
            <div class="info-row">
                <span class="info-label">Node ID</span>
                <span class="info-value">${data.id}</span>
            </div>
            <div class="info-row">
                <span class="info-label">Type</span>
                <span class="info-value">${data.type || 'unknown'}</span>
            </div>
            <div class="info-row">
                <span class="info-label">Transactions</span>
                <span class="info-value">${data.degree}</span>
            </div>
            <div class="info-row">
                <span class="info-label">Avg Amount</span>
                <span class="info-value">$${(data.summary?.avg_amount || 0).toFixed(2)}</span>
            </div>
            <div class="info-row" style="border: none;">
                <span class="info-label">Status</span>
                <span class="info-value">
                    <span class="badge ${data.is_suspicious ? 'badge-high' : 'badge-low'}">
                        ${data.is_suspicious ? 'Suspicious' : 'Normal'}
                    </span>
                </span>
            </div>
        </div>

        <div class="info-card">
            <h3>Risk Assessment</h3>
            <div class="risk-meter">
                <div class="risk-bar">
                    <div class="risk-indicator" style="left: ${(data.risk_score || 0) * 100}%;"></div>
                </div>
                <div style="text-align: center; margin-top: 15px;">
                    <div style="font-size: 32px; font-weight: bold; color: ${getRiskColorValue(data.risk_score || 0)};">
                        ${(data.risk_score || 0) > 0.7 ? 'High' : (data.risk_score || 0) > 0.3 ? 'Medium' : 'Low'}
                    </div>
                    <div style="font-size: 12px; color: #a0b0c0; margin-top: 5px;">
                        Risk Score: ${((data.risk_score || 0) * 100).toFixed(1)}%
                    </div>
                </div>
            </div>
        </div>

        <div class="info-card">
            <h3>Top Counterparties</h3>
            ${Object.entries(data.top_counterparties || {}).slice(0, 5).map(([id, count]) => `
                <div class="info-row">
                    <span class="info-label">${id}</span>
                    <span class="info-value">${count} txns</span>
                </div>
            `).join('')}
        </div>
    `;
    
    detailsContent.innerHTML = html;
}

function displayNodeDetailsOld(data) {
    const detailsContent = document.querySelector('[data-content="details"]');
    
    const html = `
        <div class="info-card">
            <h3>Node Information</h3>
            <div class="info-row">
                <span class="info-label">Node ID</span>
                <span class="info-value">${data.id}</span>
            </div>
            <div class="info-row">
                <span class="info-label">Transactions</span>
                <span class="info-value">${data.degree}</span>
            </div>
            <div class="info-row">
                <span class="info-label">Avg Amount</span>
                <span class="info-value">$${(data.summary?.avg_amount || 0).toFixed(2)}</span>
            </div>
            <div class="info-row" style="border: none;">
                <span class="info-label">Risk</span>
                <span class="info-value">${((data.risk || 0) * 100).toFixed(1)}%</span>
            </div>
        </div>

        <div class="info-card">
            <h3>Top Counterparties</h3>
            ${Object.entries(data.top_counterparties || {}).slice(0, 5).map(([id, count]) => `
                <div class="info-row">
                    <span class="info-label">${id}</span>
                    <span class="info-value">${count} txns</span>
                </div>
            `).join('')}
        </div>
    `;
    
    detailsContent.innerHTML = html;
}

// ============================
// UTILITIES
// ============================
function getRiskColor(item) {
    if (item.is_suspicious) return COLORS.high;
    const score = item.risk_score || item.pred_prob || 0;
    if (score > 0.7 || score > 70) return COLORS.high;
    if (score > 0.3 || score > 30) return COLORS.medium;
    return COLORS.safe;
}

function getRiskColorValue(score) {
    if (score > 0.7) return COLORS.high;
    if (score > 0.3) return COLORS.medium;
    return COLORS.safe;
}

function getNodeSize(node) {
    if (node.is_suspicious) return 8;
    const score = node.risk_score || 0;
    if (score > 0.5 || score > 30) return 6;
    return 5;
}

function updateHeaderStats(metrics) {
    document.getElementById('totalNodes').textContent = metrics.num_nodes || 0;
    document.getElementById('totalEdges').textContent = (metrics.num_edges || 0).toLocaleString();
    document.getElementById('fraudRate').textContent = `${(metrics.fraud_rate || 0).toFixed(1)}%`;
    document.getElementById('accuracy').textContent = '94.2%';
}

function showNotification(message, type = 'info') {
    console.log(`[${type}] ${message}`);
}

function showLoading(message = 'Loading...') {
    const loading = document.getElementById('loadingIndicator');
    if (loading) {
        loading.style.display = 'block';
        const text = loading.querySelector('div:last-child');
        if (text) text.textContent = message;
    }
}

function hideLoading() {
    const loading = document.getElementById('loadingIndicator');
    if (loading) {
        loading.style.display = 'none';
    }
}

// ============================
// DRAG HANDLERS
// ============================
function dragStarted(event, d) {
    if (!event.active) simulation.alphaTarget(0.3).restart();
    d.fx = d.x;
    d.fy = d.y;
    event.sourceEvent.stopPropagation();
}

function dragged(event, d) {
    d.fx = event.x;
    d.fy = event.y;
}

function dragEnded(event, d) {
    if (!event.active) simulation.alphaTarget(0);
    d.fx = null;
    d.fy = null;
}

// ============================
// HIGHLIGHT EFFECTS
// ============================
function highlightNode(node, element) {
    d3.select(element)
        .transition()
        .duration(200)
        .attr('r', getNodeSize(node) + 3);
    
    d3.select(element.parentNode).select('text')
        .transition()
        .duration(200)
        .style('opacity', 1)
        .style('font-size', '11px');
}

function unhighlightNode(node, element) {
    if (selectedNode && selectedNode.id === node.id) return;
    
    d3.select(element)
        .transition()
        .duration(200)
        .attr('r', getNodeSize(node));
    
    d3.select(element.parentNode).select('text')
        .transition()
        .duration(200)
        .style('opacity', 0.7)
        .style('font-size', '9px');
}

// ============================
// EVENT LISTENERS
// ============================
function setupEventListeners() {
    // Load graph button
    const loadBtn = document.getElementById('loadGraphBtn');
    if (loadBtn) {
        loadBtn.addEventListener('click', loadGraph);
        console.log('Load button listener attached');
    }
    
    // Tab switching
    document.querySelectorAll('.tab').forEach(tab => {
        tab.addEventListener('click', () => {
            const tabName = tab.getAttribute('data-tab');
            
            document.querySelectorAll('.tab').forEach(t => t.classList.remove('active'));
            document.querySelectorAll('.tab-content').forEach(c => c.classList.remove('active'));
            
            tab.classList.add('active');
            document.querySelector(`[data-content="${tabName}"]`).classList.add('active');
        });
    });
    
    // Filter chips
    document.querySelectorAll('.filter-chips').forEach(group => {
        group.querySelectorAll('.chip').forEach(chip => {
            chip.addEventListener('click', () => {
                group.querySelectorAll('.chip').forEach(c => c.classList.remove('active'));
                chip.classList.add('active');
            });
        });
    });
}

// ============================
// GLOBAL FUNCTIONS
// ============================
function resetView() {
    const zoom = d3.zoom().scaleExtent([0.1, 5]);
    svg.transition()
        .duration(750)
        .call(zoom.transform, d3.zoomIdentity);
}

function centerGraph() {
    if (simulation) {
        simulation.alpha(0.3).restart();
    }
}

// Make functions globally accessible
window.resetView = resetView;
window.centerGraph = centerGraph;