window.renderGraph = function(nodes, edges){
  const cy = cytoscape({
    container: document.getElementById('cy'),
    elements: { nodes, edges },
    style: [
      { selector: 'node', style: { 'background-color': '#778', 'label': 'data(label)', 'color':'#222', 'font-size':'12px' } },
      { selector: 'node[type="Human"]', style: { 'background-color': '#00e5ff' } },
      { selector: 'node[type="Object"]', style: { 'background-color': '#ff6f00' } },
      { selector: 'edge', style: { 'line-color': '#999', 'width': 2, 'curve-style': 'bezier', 'target-arrow-shape':'triangle', 'label':'data(label)', 'font-size':'10px' } }
    ],
    layout: { name: 'cose', fit: true, animate: true }
  });
  cy.on('tap', 'node', (evt)=>{ const d = evt.target.data(); alert(`${d.type || 'Node'}: ${d.label}`); });
};