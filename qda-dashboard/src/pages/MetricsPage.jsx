import React, { useState } from 'react';
import { Bar, Radar } from 'react-chartjs-2';
import { overallMetrics, perClassMetrics } from '../data/experimentData';
import { defaultChartOptions } from '../utils/chartConfig';

const MetricsPage = () => {
  const [sortField, setSortField] = useState('support');
  const [sortDirection, setSortDirection] = useState('desc');

  const handleSort = (field) => {
    if (sortField === field) {
      setSortDirection(sortDirection === 'asc' ? 'desc' : 'asc');
    } else {
      setSortField(field);
      setSortDirection('desc');
    }
  };

  const sortedMetrics = [...perClassMetrics].sort((a, b) => {
    const factor = sortDirection === 'asc' ? 1 : -1;
    if (typeof a[sortField] === 'string') {
      return a[sortField].localeCompare(b[sortField]) * factor;
    }
    return (a[sortField] - b[sortField]) * factor;
  });

  const comparisonChartData = {
    labels: perClassMetrics.map((m) => m.class),
    datasets: [
      {
        label: 'Precision (%)',
        data: perClassMetrics.map((m) => +(m.precision * 100).toFixed(2)),
        backgroundColor: 'rgba(59, 130, 246, 0.75)',
        borderColor: '#3b82f6',
        borderWidth: 1,
        borderRadius: 4,
      },
      {
        label: 'Recall (%)',
        data: perClassMetrics.map((m) => +(m.recall * 100).toFixed(2)),
        backgroundColor: 'rgba(16, 185, 129, 0.75)',
        borderColor: '#10b981',
        borderWidth: 1,
        borderRadius: 4,
      },
      {
        label: 'F1-Score (%)',
        data: perClassMetrics.map((m) => +(m.f1 * 100).toFixed(2)),
        backgroundColor: 'rgba(139, 92, 246, 0.75)',
        borderColor: '#8b5cf6',
        borderWidth: 1,
        borderRadius: 4,
      },
    ],
  };

  const radarChartData = {
    labels: perClassMetrics.map((m) => m.class),
    datasets: [
      {
        label: 'Precision (%)',
        data: perClassMetrics.map((m) => +(m.precision * 100).toFixed(2)),
        backgroundColor: 'rgba(59, 130, 246, 0.2)',
        borderColor: '#3b82f6',
        pointBackgroundColor: '#3b82f6',
      },
      {
        label: 'Recall (%)',
        data: perClassMetrics.map((m) => +(m.recall * 100).toFixed(2)),
        backgroundColor: 'rgba(16, 185, 129, 0.2)',
        borderColor: '#10b981',
        pointBackgroundColor: '#10b981',
      },
      {
        label: 'F1-Score (%)',
        data: perClassMetrics.map((m) => +(m.f1 * 100).toFixed(2)),
        backgroundColor: 'rgba(139, 92, 246, 0.2)',
        borderColor: '#8b5cf6',
        pointBackgroundColor: '#8b5cf6',
      },
    ],
  };

  return (
    <div className="metrics-page animate-in">
      <div className="page-header">
        <h1>Model Performance & Per-Class Evaluation</h1>
        <p>Comprehensive classification report, discrimination metrics, and generalization analysis across all Android malware families.</p>
      </div>

      {/* Generalization Comparison Cards */}
      <div className="metrics-grid" style={{ marginBottom: '24px' }}>
        <div className="metric-card blue">
          <div className="metric-label">Training Accuracy</div>
          <div className="metric-value">{(overallMetrics.trainAccuracy * 100).toFixed(2)}%</div>
          <div className="metric-change neutral">N = 9,278 samples</div>
        </div>
        <div className="metric-card emerald">
          <div className="metric-label">Test Accuracy</div>
          <div className="metric-value">{(overallMetrics.accuracy * 100).toFixed(2)}%</div>
          <div className="metric-change positive">Δ Generalization Gap: 1.46%</div>
        </div>
        <div className="metric-card purple">
          <div className="metric-label">Training Macro F1</div>
          <div className="metric-value">{(overallMetrics.trainMacroF1 * 100).toFixed(2)}%</div>
          <div className="metric-change neutral">Cross-validated baseline</div>
        </div>
        <div className="metric-card cyan">
          <div className="metric-label">Test Macro F1</div>
          <div className="metric-value">{(overallMetrics.macroF1 * 100).toFixed(2)}%</div>
          <div className="metric-change positive">Δ Generalization Gap: 1.71%</div>
        </div>
      </div>

      {/* Per-Class Table Card */}
      <div className="card" style={{ marginBottom: '28px' }}>
        <div className="card-header">
          <div>
            <div className="card-title">
              <span className="icon">📋</span> Detailed Classification Report
            </div>
            <div className="card-subtitle">Click column headers to sort classes</div>
          </div>
          <div style={{ display: 'flex', gap: '8px' }}>
            <span className="status-badge pass">Test Set: 2,320 instances</span>
          </div>
        </div>

        <div style={{ overflowX: 'auto' }}>
          <table className="data-table">
            <thead>
              <tr>
                <th onClick={() => handleSort('class')} style={{ cursor: 'pointer' }}>
                  Malware Family {sortField === 'class' && (sortDirection === 'asc' ? '▲' : '▼')}
                </th>
                <th onClick={() => handleSort('precision')} style={{ cursor: 'pointer' }}>
                  Precision {sortField === 'precision' && (sortDirection === 'asc' ? '▲' : '▼')}
                </th>
                <th onClick={() => handleSort('recall')} style={{ cursor: 'pointer' }}>
                  Recall {sortField === 'recall' && (sortDirection === 'asc' ? '▲' : '▼')}
                </th>
                <th onClick={() => handleSort('f1')} style={{ cursor: 'pointer' }}>
                  F1-Score {sortField === 'f1' && (sortDirection === 'asc' ? '▲' : '▼')}
                </th>
                <th onClick={() => handleSort('support')} style={{ cursor: 'pointer' }}>
                  Support {sortField === 'support' && (sortDirection === 'asc' ? '▲' : '▼')}
                </th>
                <th style={{ width: '180px' }}>Performance Bar</th>
              </tr>
            </thead>
            <tbody>
              {sortedMetrics.map((item) => (
                <tr key={item.class}>
                  <td style={{ fontWeight: '600', color: 'var(--text-primary)' }}>
                    {item.class}
                  </td>
                  <td className={`table-value ${item.precision >= 0.9 ? 'high' : item.precision >= 0.8 ? 'mid' : 'low'}`}>
                    {(item.precision * 100).toFixed(2)}%
                  </td>
                  <td className={`table-value ${item.recall >= 0.9 ? 'high' : item.recall >= 0.8 ? 'mid' : 'low'}`}>
                    {(item.recall * 100).toFixed(2)}%
                  </td>
                  <td className={`table-value ${item.f1 >= 0.9 ? 'high' : item.f1 >= 0.8 ? 'mid' : 'low'}`}>
                    {(item.f1 * 100).toFixed(2)}%
                  </td>
                  <td style={{ fontFamily: 'var(--font-mono)' }}>
                    {item.support.toLocaleString()}
                  </td>
                  <td>
                    <div className="progress-bar">
                      <div
                        className="progress-fill"
                        style={{
                          width: `${item.f1 * 100}%`,
                          background:
                            item.f1 >= 0.9
                              ? 'linear-gradient(90deg, #10b981, #22d3ee)'
                              : item.f1 >= 0.85
                              ? 'linear-gradient(90deg, #3b82f6, #8b5cf6)'
                              : 'linear-gradient(90deg, #f59e0b, #f43f5e)',
                        }}
                      />
                    </div>
                  </td>
                </tr>
              ))}
            </tbody>
            <tfoot>
              <tr style={{ borderTop: '2px solid rgba(99, 115, 171, 0.2)', background: 'rgba(0, 0, 0, 0.2)' }}>
                <td style={{ fontWeight: '700', color: '#60a5fa' }}>Macro Average</td>
                <td className="table-value">{(overallMetrics.macroPrecision * 100).toFixed(2)}%</td>
                <td className="table-value">{(overallMetrics.macroRecall * 100).toFixed(2)}%</td>
                <td className="table-value" style={{ color: '#22d3ee' }}>{(overallMetrics.macroF1 * 100).toFixed(2)}%</td>
                <td style={{ fontFamily: 'var(--font-mono)' }}>2,320</td>
                <td>—</td>
              </tr>
              <tr style={{ background: 'rgba(0, 0, 0, 0.2)' }}>
                <td style={{ fontWeight: '700', color: '#a78bfa' }}>Weighted Average</td>
                <td className="table-value">{(overallMetrics.weightedPrecision * 100).toFixed(2)}%</td>
                <td className="table-value">{(overallMetrics.weightedRecall * 100).toFixed(2)}%</td>
                <td className="table-value" style={{ color: '#a78bfa' }}>{(overallMetrics.weightedF1 * 100).toFixed(2)}%</td>
                <td style={{ fontFamily: 'var(--font-mono)' }}>2,320</td>
                <td>—</td>
              </tr>
            </tfoot>
          </table>
        </div>
      </div>

      {/* Side-by-side charts */}
      <div className="charts-grid">
        <div className="card">
          <div className="card-header">
            <div>
              <div className="card-title">
                <span className="icon">📈</span> Precision vs Recall vs F1 Comparison
              </div>
              <div className="card-subtitle">Tri-metric breakdown per malware family</div>
            </div>
          </div>
          <div className="chart-container">
            <Bar
              data={comparisonChartData}
              options={{
                ...defaultChartOptions,
                scales: {
                  ...defaultChartOptions.scales,
                  y: {
                    ...defaultChartOptions.scales.y,
                    min: 70,
                    max: 100,
                    ticks: {
                      ...defaultChartOptions.scales.y.ticks,
                      callback: (val) => `${val}%`,
                    },
                  },
                },
              }}
            />
          </div>
        </div>

        <div className="card">
          <div className="card-header">
            <div>
              <div className="card-title">
                <span className="icon">🕸️</span> Class Performance Radar Profile
              </div>
              <div className="card-subtitle">Multi-dimensional trade-offs across classes</div>
            </div>
          </div>
          <div className="chart-container">
            <Radar
              data={radarChartData}
              options={{
                responsive: true,
                maintainAspectRatio: false,
                scales: {
                  r: {
                    angleLines: { color: 'rgba(99, 115, 171, 0.15)' },
                    grid: { color: 'rgba(99, 115, 171, 0.15)' },
                    pointLabels: {
                      color: '#94a3c8',
                      font: { family: "'Inter', sans-serif", size: 11, weight: '600' },
                    },
                    ticks: {
                      backdropColor: 'transparent',
                      color: '#64748b',
                      stepSize: 10,
                      min: 60,
                      max: 100,
                    },
                  },
                },
                plugins: {
                  legend: {
                    position: 'bottom',
                    labels: { color: '#94a3c8', font: { size: 11 } },
                  },
                },
              }}
            />
          </div>
        </div>
      </div>
    </div>
  );
};

export default MetricsPage;
