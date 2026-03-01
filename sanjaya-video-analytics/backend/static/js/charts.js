// Advanced Charts for Sanjaya Dashboard
// Motion Analysis, Event Distribution, Zone Heatmap, Timeline

let motionChart, eventChart, zoneChart, timelineChart;

function renderCharts(data) {
  console.log('renderCharts called with data:', data);
  
  try {
    const frames = data.salient_frames || [];
    const vlmResults = data.vlm_results || [];
    const insights = data.insights || {};
    const cvEvents = data.cv_events || 0;
    
    // Destroy existing charts
    if (motionChart) motionChart.destroy();
    if (eventChart) eventChart.destroy();
    if (zoneChart) zoneChart.destroy();
    if (timelineChart) timelineChart.destroy();
    
    // 1. MOTION ANALYSIS CHART (Line chart showing person movement over time)
    renderMotionChart(frames);
    
    // 2. EVENT DISTRIBUTION CHART (Pie chart of event types)
    renderEventChart(insights);
    
    // 3. ZONE HEATMAP CHART (Bar chart of zone activity)
    renderZoneChart(frames);
    
    // 4. TIMELINE CHART (Scatter plot of events over time)
    renderTimelineChart(insights);
    
    // 5. UPDATE EVENT STATS
    updateEventStats(insights);
    
    console.log('All charts rendered successfully');
  } catch (error) {
    console.error('Error in renderCharts:', error);
    throw error; // Re-throw so caller knows it failed
  }
}

function renderMotionChart(frames) {
  const ctx = document.getElementById('motionChart');
  if (!ctx) {
    console.warn('motionChart canvas not found');
    return;
  }
  
  try {
    const labels = frames.map((f, i) => `Frame ${i + 1}`);
    const personCounts = frames.map(f => f.persons || 0);
    const objectCounts = frames.map(f => f.objects || 0);
  
  motionChart = new Chart(ctx, {
    type: 'line',
    data: {
      labels: labels,
      datasets: [
        {
          label: 'Persons Detected',
          data: personCounts,
          borderColor: '#667eea',
          backgroundColor: 'rgba(102, 126, 234, 0.1)',
          tension: 0.4,
          fill: true
        },
        {
          label: 'Objects Detected',
          data: objectCounts,
          borderColor: '#f6ad55',
          backgroundColor: 'rgba(246, 173, 85, 0.1)',
          tension: 0.4,
          fill: true
        }
      ]
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      plugins: {
        legend: {
          display: true,
          position: 'bottom'
        },
        title: {
          display: true,
          text: 'Detection Activity Timeline'
        }
      },
      scales: {
        y: {
          beginAtZero: true,
          title: {
            display: true,
            text: 'Count'
          }
        }
      }
    }
  });
  } catch (error) {
    console.error('Error rendering motion chart:', error);
  }
}

function renderEventChart(insights) {
  const ctx = document.getElementById('eventChart');
  if (!ctx) {
    console.warn('eventChart canvas not found');
    return;
  }
  
  try {
    const risks = insights.risks || [];
    const anomalies = insights.anomalies || [];
  
  // Count risk severities
  const highRisk = risks.filter(r => r.severity === 'high').length;
  const mediumRisk = risks.filter(r => r.severity === 'medium').length;
  const lowRisk = risks.filter(r => r.severity === 'low').length;
  const anomalyCount = anomalies.length;
  
  eventChart = new Chart(ctx, {
    type: 'doughnut',
    data: {
      labels: ['High Risk', 'Medium Risk', 'Low Risk', 'Anomalies'],
      datasets: [{
        data: [highRisk, mediumRisk, lowRisk, anomalyCount],
        backgroundColor: [
          '#e53e3e',  // red
          '#ed8936',  // orange
          '#38a169',  // green
          '#9f7aea'   // purple
        ],
        borderWidth: 2,
        borderColor: '#fff'
      }]
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      plugins: {
        legend: {
          display: true,
          position: 'bottom'
        },
        title: {
          display: true,
          text: 'Risk & Anomaly Distribution'
        }
      }
    }
  });
  } catch (error) {
    console.error('Error rendering event chart:', error);
  }
}

function renderZoneChart(frames) {
  const ctx = document.getElementById('zoneChart');
  if (!ctx) {
    console.warn('zoneChart canvas not found');
    return;
  }
  
  try {
  
  // Aggregate zone activity from all frames
  const zoneActivity = {};
  
  frames.forEach(frame => {
    const zones = frame.zones || {};
    Object.keys(zones).forEach(zone => {
      zoneActivity[zone] = (zoneActivity[zone] || 0) + zones[zone];
    });
  });
  
  const zoneLabels = Object.keys(zoneActivity).sort();
  const zoneData = zoneLabels.map(z => zoneActivity[z]);
  
  zoneChart = new Chart(ctx, {
    type: 'bar',
    data: {
      labels: zoneLabels,
      datasets: [{
        label: 'Activity Count',
        data: zoneData,
        backgroundColor: [
          '#667eea',
          '#764ba2',
          '#f093fb',
          '#4facfe',
          '#00f2fe',
          '#43e97b',
          '#f6ad55',
          '#fc8181',
          '#9f7aea'
        ],
        borderColor: '#fff',
        borderWidth: 2
      }]
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      plugins: {
        legend: {
          display: false
        },
        title: {
          display: true,
          text: 'Zone-wise Activity Intensity'
        }
      },
      scales: {
        y: {
          beginAtZero: true,
          title: {
            display: true,
            text: 'Activity Level'
          }
        },
        x: {
          title: {
            display: true,
            text: 'Zones'
          }
        }
      }
    }
  });
  } catch (error) {
    console.error('Error rendering zone chart:', error);
  }
}

function renderTimelineChart(insights) {
  const ctx = document.getElementById('timelineChart');
  if (!ctx) {
    console.warn('timelineChart canvas not found');
    return;
  }
  
  try {
  
  const risks = insights.risks || [];
  const anomalies = insights.anomalies || [];
  
  // Create timeline data points
  const timelineData = [];
  
  // Add risks to timeline
  risks.forEach((risk, i) => {
    const severity = risk.severity === 'high' ? 3 : risk.severity === 'medium' ? 2 : 1;
    timelineData.push({
      x: i,
      y: severity,
      type: 'risk',
      label: risk.type,
      color: severity === 3 ? '#e53e3e' : severity === 2 ? '#ed8936' : '#38a169'
    });
  });
  
  // Add anomalies
  anomalies.forEach((anomaly, i) => {
    timelineData.push({
      x: risks.length + i,
      y: 2.5,
      type: 'anomaly',
      label: anomaly.type,
      color: '#9f7aea'
    });
  });
  
  timelineChart = new Chart(ctx, {
    type: 'scatter',
    data: {
      datasets: [{
        label: 'Events',
        data: timelineData,
        backgroundColor: timelineData.map(d => d.color),
        borderColor: '#fff',
        borderWidth: 2,
        pointRadius: 8,
        pointHoverRadius: 12
      }]
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      plugins: {
        legend: {
          display: false
        },
        tooltip: {
          callbacks: {
            label: function(context) {
              const point = timelineData[context.dataIndex];
              return `${point.type.toUpperCase()}: ${point.label}`;
            }
          }
        }
      },
      scales: {
        y: {
          beginAtZero: true,
          max: 4,
          ticks: {
            callback: function(value) {
              const labels = ['', 'Low', 'Medium', 'High'];
              return labels[value] || '';
            }
          },
          title: {
            display: true,
            text: 'Severity'
          }
        },
        x: {
          title: {
            display: true,
            text: 'Event Sequence'
          }
        }
      }
    }
  });
  } catch (error) {
    console.error('Error rendering timeline chart:', error);
  }
}

function updateEventStats(insights) {
  try {
    // Count event types from insights
    const risks = insights.risks || [];
    
    let stationary = 0, walking = 0, moving = 0, loitering = 0;
    
    risks.forEach(risk => {
      const type = (risk.type || '').toLowerCase();
      if (type.includes('loiter')) loitering++;
      else if (type.includes('crowd') || type.includes('formation')) moving++;
    });
    
    // Update DOM with safety checks
    const statEls = {
      statStationary, 
      statWalking,
      statMoving,
      statLoitering
    };
    
    const el1 = document.getElementById('statStationary');
    const el2 = document.getElementById('statWalking');
    const el3 = document.getElementById('statMoving');
    const el4 = document.getElementById('statLoitering');
    
    if (el1) el1.textContent = stationary;
    if (el2) el2.textContent = walking;
    if (el3) el3.textContent = moving + risks.length;
    if (el4) el4.textContent = loitering;
  } catch (error) {
    console.error('Error updating event stats:', error);
  }
}


// Export function
window.renderCharts = renderCharts;
