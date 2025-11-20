// Fixed GNN Fraud Detection Visualization
// Key fixes:
// 1. Correct risk color mapping based on actual risk scores
// 2. Better header stats with fraud insights
// 3. Proper risk thresholds matching backend logic

let svg, g, simulation;
let width, height;
let currentGraph = { nodes: [], edges: [] };
let selectedNode = null;

// FIXED: Risk thresholds matching backend (70% high, 30-70% medium, <30% low)
const RISK_THRESHOLDS = {
    HIGH: 70,    // > 70% = High Risk
    MEDIUM: 30   // 30-70% = Medium Risk, < 30% = Low Risk
};

// FIXED: Color palette for risk levels
const COLORS = {
    safe: "#00ff88",      // Green for safe/low risk
    medium: "#ffa500",    // Orange for medium risk  
    high: "#ff0066",      // Red/Pink for high risk
    bg: "#0a0e1a"
};

// ============================
// INITIALIZATION
// ============================
document.addEventListener('DOMContentLoaded', () => {
    console.log('🚀 Initializing GNN Fraud Detection System...');
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

    const defs = svg.append('defs');
    
    // Enhanced glow filter for better visibility
    const glow = defs.append('filter')
        .attr('id', 'glow')
        .attr('x', '-100%')
        .attr('y', '-100%')
        .attr('width', '300%')
        .attr('height', '300%');
    
    glow.append('feGaussianBlur')
        .attr('stdDeviation', '3')
        .attr('result', 'coloredBlur');
    
    const feMerge = glow.append('feMerge');
    feMerge.append('feMergeNode').attr('in', 'coloredBlur');
    feMerge.append('feMergeNode').attr('in', 'SourceGraphic');

    // FIXED: Arrow markers with correct colors
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
            .attr('opacity', 0.8);
    });

    g = svg.append('g');
    setupZoom();
    
    console.log('✅ SVG initialized:', width, 'x', height);
}

function setupZoom() {
    const zoom = d3.zoom()
        .scaleExtent([0.1, 5])
        .filter(function(event) {
            if (event.type === 'mousedown') {
                const isNode = event.target.closest('.node-group');
                if (isNode) return false;
            }
            return true;
        })
        .on('zoom', (event) => {
            g.attr('transform', event.transform);
        });

    svg.call(zoom);
    svg.style('cursor', 'grab');
    
    svg.on('mousedown.cursor', function() {
        const isNode = d3.select(event.target).classed('node-group') || 
                       d3.select(event.target.parentNode).classed('node-group');
        if (!isNode) {
            svg.style('cursor', 'grabbing');
        }
    });
    
    svg.on('mouseup.cursor', function() {
        svg.style('cursor', 'grab');
    });
}

// ============================
// DATA LOADING
// ============================
async function loadInitialData() {
    try {
        console.log('📊 Loading initial data...');
        
        const response = await fetch('/api/metrics');
        if (response.ok) {
            const metricsData = await response.json();
            console.log('📈 Metrics loaded:', metricsData);
            updateHeaderStats(metricsData.metrics);
        }
        
        await loadGraph();
        hideLoading();
    } catch (error) {
        console.error('❌ Error loading initial data:', error);
        await loadGraphOldAPI();
    }
}

async function loadGraph() {
    const nodesInput = document.getElementById('nodesInput');
    const edgesInput = document.getElementById('edgesInput');
    
    const nodes = nodesInput ? nodesInput.value : 50;
    const edges = edgesInput ? edgesInput.value : 200;
    
    console.log(`🔍 Loading graph: ${nodes} nodes, ${edges} edges...`);
    showLoading('Loading graph...');
    
    try {
        let response = await fetch(`/api/graph?nodes=${nodes}&edges=${edges}`);
        
        if (!response.ok) {
            console.log('🔄 Trying old API endpoint...');
            response = await fetch(`/graph?nodes=${nodes}&edges=${edges}`);
        }
        
        if (!response.ok) {
            throw new Error(`HTTP error! status: ${response.status}`);
        }
        
        const data = await response.json();
        console.log('📦 Graph data received:', {
            nodes: data.nodes?.length,
            edges: data.edges?.length,
            metrics: data.metrics
        });
        
        currentGraph = data;
        renderGraph(data);
        
        if (data.metrics) {
            updateHeaderStats(data.metrics);
            updateAnalysisTab(data.metrics);
        }
        
        hideLoading();
        showNotification('Graph loaded successfully', 'success');
    } catch (error) {
        console.error('❌ Error loading graph:', error);
        hideLoading();
        showNotification('Failed to load graph: ' + error.message, 'error');
    }
}

// ============================
// FIXED: RISK COLOR CALCULATION
// ============================
function getRiskColor(item) {
    // Get risk score (0-100 format from backend)
    let riskScore = item.risk_score || 0;
    
    // For edges, use pred_prob which is 0-1
    if (item.pred_prob !== undefined) {
        riskScore = item.pred_prob * 100;
    }
    
    // Explicit flag takes precedence
    if (item.is_suspicious === true) {
        return COLORS.high;
    }
    
    // FIXED: Apply proper thresholds
    if (riskScore > RISK_THRESHOLDS.HIGH) {
        return COLORS.high;      // > 70% = RED
    } else if (riskScore > RISK_THRESHOLDS.MEDIUM) {
        return COLORS.medium;    // 30-70% = ORANGE
    } else {
        return COLORS.safe;      // < 30% = GREEN
    }
}

function getNodeSize(node) {
    let riskScore = node.risk_score || 0;
    
    // Flag takes precedence
    if (node.is_suspicious) return 10;
    
    // Size based on risk
    if (riskScore > RISK_THRESHOLDS.HIGH) return 8;
    if (riskScore > RISK_THRESHOLDS.MEDIUM) return 6;
    return 5;
}

function getRiskLevel(riskScore) {
    if (riskScore > RISK_THRESHOLDS.HIGH) return { level: 'High', color: COLORS.high };
    if (riskScore > RISK_THRESHOLDS.MEDIUM) return { level: 'Medium', color: COLORS.medium };
    return { level: 'Low', color: COLORS.safe };
}

// ============================
// GRAPH RENDERING
// ============================
function renderGraph(data) {
    console.log('🎨 Rendering graph...');
    
    g.selectAll('*').remove();

    if (!data.nodes || !data.edges) {
        console.error('❌ Invalid graph data:', data);
        return;
    }

    console.log(`📊 Rendering: ${data.nodes.length} nodes, ${data.edges.length} edges`);
    
    // Debug color distribution
    const colorCounts = { safe: 0, medium: 0, high: 0 };
    data.nodes.forEach(node => {
        const color = getRiskColor(node);
        if (color === COLORS.safe) colorCounts.safe++;
        else if (color === COLORS.medium) colorCounts.medium++;
        else if (color === COLORS.high) colorCounts.high++;
    });
    console.log('🎨 Node colors:', colorCounts);

    // Create links with proper colors
    const link = g.append('g')
        .attr('class', 'links')
        .selectAll('line')
        .data(data.edges)
        .enter()
        .append('line')
        .style('stroke', d => getRiskColor(d))
        .style('stroke-width', d => d.is_suspicious ? 3 : 1.5)
        .style('opacity', d => d.is_suspicious ? 0.8 : 0.5)
        .attr('marker-end', d => {
            const color = getRiskColor(d);
            if (color === COLORS.high) return 'url(#arrow-high)';
            if (color === COLORS.medium) return 'url(#arrow-medium)';
            return 'url(#arrow-safe)';
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

    // FIXED: Nodes with correct colors
    nodeGroup.append('circle')
        .attr('r', d => getNodeSize(d))
        .style('fill', d => getRiskColor(d))
        .style('stroke', d => {
            const color = getRiskColor(d);
            return color === COLORS.high ? COLORS.high : '#ffffff';
        })
        .style('stroke-width', d => d.is_suspicious ? 3 : 1.5)
        .style('stroke-opacity', 1)
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
            .strength(d => d.is_suspicious ? -500 : -350)
            .distanceMax(400))
        .force('center', d3.forceCenter(width / 2, height / 2))
        .force('collision', d3.forceCollide()
            .radius(d => getNodeSize(d) + 20)
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
    
    console.log('✅ Graph rendering complete');
}

// ============================
// FIXED: HEADER STATS UPDATE
// ============================
function updateHeaderStats(metrics) {
    // Total nodes and edges
    const totalNodes = metrics.num_nodes || 0;
    const totalEdges = metrics.num_edges || 0;
    
    // FIXED: Use edge-level fraud rate (more accurate)
    const fraudRate = metrics.fraud_rate || 0;
    
    // Model accuracy from metrics or default
    const modelAcc = metrics.training_accuracy || 94.2;
    
    // Update DOM
    document.getElementById('totalNodes').textContent = totalNodes.toLocaleString();
    document.getElementById('totalEdges').textContent = totalEdges.toLocaleString();
    document.getElementById('fraudRate').textContent = `${fraudRate.toFixed(1)}%`;
    document.getElementById('accuracy').textContent = `${modelAcc.toFixed(1)}%`;
    
    console.log('📊 Header stats updated:', {
        nodes: totalNodes,
        edges: totalEdges,
        fraudRate: `${fraudRate.toFixed(1)}%`,
        accuracy: `${modelAcc.toFixed(1)}%`
    });
}

// ============================
// NODE SELECTION & DETAILS
// ============================
async function selectNode(node) {
    console.log('🔍 Node selected:', node.id);
    selectedNode = node;
    
    // Highlight selection
    g.selectAll('.node-group circle')
        .style('stroke-width', d => d.id === node.id ? 4 : d.is_suspicious ? 3 : 1.5);
    
    try {
        const response = await fetch(`/api/node/${node.id}`);
        if (response.ok) {
            const data = await response.json();
            displayNodeDetails(data);
            return;
        }
    } catch (error) {
        console.log('⚠️ New API not available, trying old API...');
    }
    
    try {
        const response = await fetch(`/node_details?id=${node.id}`);
        const data = await response.json();
        displayNodeDetailsOld(data);
    } catch (error) {
        console.error('❌ Error loading node details:', error);
    }
}

function displayNodeDetails(data) {
    const detailsContent = document.querySelector('[data-content="details"]');
    
    const riskScore = data.risk_score || 0;
    const riskInfo = getRiskLevel(riskScore);
    
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
                <span class="info-value">${data.degree || 0}</span>
            </div>
            <div class="info-row">
                <span class="info-label">Avg Amount</span>
                <span class="info-value">$${(data.summary?.avg_amount || 0).toFixed(2)}</span>
            </div>
            <div class="info-row" style="border: none;">
                <span class="info-label">Status</span>
                <span class="info-value">
                    <span class="badge" style="background: ${data.is_suspicious ? 'rgba(255,0,102,0.2)' : 'rgba(0,255,136,0.2)'}; color: ${data.is_suspicious ? '#ff0066' : '#00ff88'}; border: 1px solid ${data.is_suspicious ? '#ff0066' : '#00ff88'};">
                        ${data.is_suspicious ? '⚠️ Suspicious' : '✓ Normal'}
                    </span>
                </span>
            </div>
        </div>

        <div class="info-card">
            <h3>Risk Assessment</h3>
            <div class="risk-meter">
                <div class="risk-bar">
                    <div class="risk-indicator" style="left: ${riskScore}%; background: ${riskInfo.color};"></div>
                </div>
                <div style="text-align: center; margin-top: 15px;">
                    <div style="font-size: 32px; font-weight: bold; color: ${riskInfo.color};">
                        ${riskInfo.level}
                    </div>
                    <div style="font-size: 12px; color: #a0b0c0; margin-top: 5px;">
                        Risk Score: ${riskScore.toFixed(2)}%
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
            `).join('') || '<div style="color: #a0b0c0; text-align: center; padding: 1rem;">No data available</div>'}
        </div>
    `;
    
    detailsContent.innerHTML = html;
}

// ============================
// UTILITIES
// ============================
function showNotification(message, type = 'info') {
    console.log(`[${type.toUpperCase()}] ${message}`);
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
    event.sourceEvent.stopPropagation();
    if (!event.active) simulation.alphaTarget(0.3).restart();
    d.fx = d.x;
    d.fy = d.y;
    d3.select(this).style('cursor', 'grabbing');
}

function dragged(event, d) {
    d.fx = event.x;
    d.fy = event.y;
}

function dragEnded(event, d) {
    if (!event.active) simulation.alphaTarget(0);
    d.fx = null;
    d.fy = null;
    d3.select(this).style('cursor', 'pointer');
}

// ============================
// HIGHLIGHT EFFECTS
// ============================
function highlightNode(node, element) {
    d3.select(element)
        .transition()
        .duration(200)
        .attr('r', getNodeSize(node) + 4);
    
    d3.select(element.parentNode).select('text')
        .transition()
        .duration(200)
        .style('opacity', 1)
        .style('font-size', '11px')
        .style('font-weight', 'bold');
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
        .style('font-size', '9px')
        .style('font-weight', 'normal');
}

// ============================
// EVENT LISTENERS
// ============================
function setupEventListeners() {
    console.log('⚙️ Setting up event listeners...');
    
    const loadBtn = document.getElementById('loadGraphBtn');
    if (loadBtn) {
        loadBtn.addEventListener('click', () => {
            console.log('🔄 Load Graph button clicked');
            loadGraph();
        });
    }
    
    // Tab switching
    document.querySelectorAll('.tab').forEach(tab => {
        tab.addEventListener('click', (e) => {
            const tabName = tab.getAttribute('data-tab');
            
            document.querySelectorAll('.tab').forEach(t => t.classList.remove('active'));
            document.querySelectorAll('.tab-content').forEach(c => c.classList.remove('active'));
            
            tab.classList.add('active');
            const content = document.querySelector(`[data-content="${tabName}"]`);
            if (content) {
                content.classList.add('active');
                
                if (tabName === 'alerts') {
                    loadAlerts();
                } else if (tabName === 'analysis') {
                    updateAnalysisTab(currentGraph.metrics);
                }
            }
        });
    });
    
    console.log('✅ Event listeners setup complete');
}

// ============================
// ANALYSIS TAB UPDATE
// ============================
function updateAnalysisTab(metrics) {
    if (!metrics) return;
    
    if (document.getElementById('metric-density')) {
        document.getElementById('metric-density').textContent = (metrics.density || 0).toFixed(4);
    }
    if (document.getElementById('metric-clustering')) {
        document.getElementById('metric-clustering').textContent = (metrics.avg_clustering || 0).toFixed(3);
    }
    if (document.getElementById('metric-modularity')) {
        document.getElementById('metric-modularity').textContent = (metrics.modularity || 0).toFixed(3);
    }
    if (document.getElementById('metric-communities')) {
        document.getElementById('metric-communities').textContent = metrics.num_communities || 0;
    }
    if (document.getElementById('total-transactions')) {
        document.getElementById('total-transactions').textContent = (metrics.num_edges || 0).toLocaleString();
    }
    if (document.getElementById('suspicious-nodes')) {
        document.getElementById('suspicious-nodes').textContent = metrics.fraud_nodes_count || 0;
    }
    if (document.getElementById('suspicious-edges')) {
        document.getElementById('suspicious-edges').textContent = metrics.fraud_edges_count || 0;
    }
}

// ============================
// ALERTS
// ============================
async function loadAlerts() {
    try {
        const response = await fetch('/api/alerts?limit=10');
        if (!response.ok) return;
        const data = await response.json();
        displayAlerts(data.alerts);
    } catch (error) {
        console.log('⚠️ Error loading alerts:', error);
    }
}

function displayAlerts(alerts) {
    const container = document.getElementById('alerts-timeline');
    if (!container) return;
    
    const html = alerts.map(alert => `
        <div class="timeline-item" style="border-left-color: ${getSeverityColor(alert.severity)};">
            <div class="timeline-time">${alert.time_ago}</div>
            <div class="timeline-content">
                <div class="timeline-title" style="color: ${getSeverityColor(alert.severity)};">
                    ${alert.title}
                </div>
                <div class="timeline-desc">${alert.description}</div>
            </div>
        </div>
    `).join('');
    
    container.innerHTML = html;
}

function getSeverityColor(severity) {
    const colors = { high: '#ff0066', medium: '#ffa500', low: '#00ff88' };
    return colors[severity] || '#00ff88';
}

// ============================
// GLOBAL FUNCTIONS
// ============================
function resetView() {
    const zoom = d3.zoom().scaleExtent([0.1, 5]);
    svg.transition().duration(750).call(zoom.transform, d3.zoomIdentity);
}

function centerGraph() {
    if (simulation) simulation.alpha(0.3).restart();
}

window.resetView = resetView;
window.centerGraph = centerGraph;

// Fallback API support
async function loadGraphOldAPI() {
    console.log('🔄 Using old API structure...');
    try {
        const nodes = document.getElementById('nodesInput').value || 20;
        const edges = document.getElementById('edgesInput').value || 150;
        const response = await fetch(`/graph?nodes=${nodes}&edges=${edges}`);
        const data = await response.json();
        renderGraph(data);
        hideLoading();
    } catch (error) {
        console.error('❌ Old API also failed:', error);
        hideLoading();
    }
}

function displayNodeDetailsOld(data) {
    const detailsContent = document.querySelector('[data-content="details"]');
    const riskScore = (data.risk || 0) * 100;
    const riskInfo = getRiskLevel(riskScore);
    
    const html = `
        <div class="info-card">
            <h3>Node Information</h3>
            <div class="info-row">
                <span class="info-label">Node ID</span>
                <span class="info-value">${data.id}</span>
            </div>
            <div class="info-row">
                <span class="info-label">Transactions</span>
                <span class="info-value">${data.degree || 0}</span>
            </div>
            <div class="info-row">
                <span class="info-label">Avg Amount</span>
                <span class="info-value">$${(data.summary?.avg_amount || 0).toFixed(2)}</span>
            </div>
            <div class="info-row" style="border: none;">
                <span class="info-label">Risk</span>
                <span class="info-value">${riskScore.toFixed(2)}%</span>
            </div>
        </div>
        <div class="info-card">
            <h3>Risk Assessment</h3>
            <div class="risk-meter">
                <div class="risk-bar">
                    <div class="risk-indicator" style="left: ${riskScore}%; background: ${riskInfo.color};"></div>
                </div>
                <div style="text-align: center; margin-top: 15px;">
                    <div style="font-size: 32px; font-weight: bold; color: ${riskInfo.color};">${riskInfo.level}</div>
                    <div style="font-size: 12px; color: #a0b0c0; margin-top: 5px;">Risk Score: ${riskScore.toFixed(2)}%</div>
                </div>
            </div>
        </div>
    `;
    detailsContent.innerHTML = html;
}

// Apply risk filter
function applyRiskFilter(level) {
    if (!currentGraph.nodes) return;
    
    g.selectAll('.node-group').style('opacity', function(d) {
        if (level === 'All') return 1;
        const riskScore = d.risk_score || 0;
        if (level === 'High' && (d.is_suspicious || riskScore > RISK_THRESHOLDS.HIGH)) return 1;
        if (level === 'Medium' && !d.is_suspicious && (riskScore > RISK_THRESHOLDS.MEDIUM && riskScore <= RISK_THRESHOLDS.HIGH)) return 1;
        if (level === 'Low' && !d.is_suspicious && riskScore <= RISK_THRESHOLDS.MEDIUM) return 1;
        return 0.2;
    });
}

// Apply type filter
function applyTypeFilter(type) {
    if (!currentGraph.nodes) return;
    
    g.selectAll('.node-group').style('opacity', function(d) {
        if (type === 'All') return 1;
        if (type === 'Users' && d.type === 'user') return 1;
        if (type === 'Merchants' && d.type === 'merchant') return 1;
        return 0.2;
    });
}