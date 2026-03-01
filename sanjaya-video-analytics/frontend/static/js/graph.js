window.renderGraph = function(nodes, edges){
  const cy = cytoscape({
    container: document.getElementById('cy'),
    elements: { nodes, edges },
    style: [
      { selector: 'node[type="Human"]', style: { 'background-color': '#00e5ff', 'label': 'data(label)' } },
      { selector: 'node[type="Object"]', style: { 'background-color': '#ff6f00', 'label': 'data(label)' } },
      { selector: 'node', style: { 'background-color': '#888', 'label': 'data(label)', 'font-size': '12px', 'color':'#222' } },
      { selector: 'edge', style: { 'line-color': '#999', 'width': 2, 'curve-style': 'bezier', 'target-arrow-shape':'triangle', 'label':'data(label)', 'font-size':'10px' } }
    ],
    layout: { name: 'cose', fit: true, animate: true }
  });
  cy.on('tap', 'node', (evt)=>{ const d = evt.target.data(); alert(`${d.type}: ${d.label}`); });
};