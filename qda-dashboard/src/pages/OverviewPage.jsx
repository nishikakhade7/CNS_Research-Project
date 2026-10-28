import React from 'react';
import { Bar, Doughnut } from 'react-chartjs-2';
import { Link } from 'react-router-dom';
import {
  overallMetrics,
  perClassMetrics,
  trainDistribution,
  testDistribution,
  experimentMetadata,
} from '../data/experimentData';
import { defaultChartOptions } from '../utils/chartConfig';

const OverviewPage = () => {
  const f1BarData = {
    labels: perClassMetrics.map((m) => m.class),
    datasets: [
      {
        label: 'F1-Score (%)',
        data: perClassMetrics.map((m) => +(m.f1 * 100).toFixed(2)),
        backgroundColor: [
          'rgba(59, 130, 246, 0.75)',
          'rgba(139, 92, 246, 0.75)',
          'rgba(16, 185, 129, 0.75)',
          'rgba(245, 158, 11, 0.75)',
          'rgba(34, 211, 238, 0.75)',
        ],
        borderColor: [
          '#3b82f6',
          '#8b5cf6',
          '#10b981',
          '#f59e0b',
          '#22d3ee',
        ],
        borderWidth: 1.5,
        borderRadius: 8,
      },
    ],
  };

  const distributionDoughnutData = {
    labels: Object.keys(testDistribution),
    datasets: [
      {
        data: Object.values(testDistribution),
        backgroundColor: [
          '#3b82f6',
          '#8b5cf6',
          '#10b981',
          '#f59e0b',
          '#22d3ee',
        ],
        borderColor: '#111827',
        borderWidth: 2,
        hoverOffset: 6,
      },
    ],
  };

  return (
    <div className="overview-page animate-in">
      {/* Hero Section */}
      <div className="hero-section">
        <div className="hero-content">
          <div style={{ display: 'inline-flex', alignItems: 'center', gap: '8px', padding: '4px 12px', borderRadius: '16px', background: 'rgba(59, 130, 246, 0.15)', color: '#60a5fa', fontSize: '12px', fontWeight: '600', marginBottom: '16px' }}>
            <span>IEEE LAYOUT FRAMEWORK</span> • <span>GROUP 1 VERIFIED</span>
          </div>
          <h1 className="hero-title">
            Quadratic Probabilistic <span className="gradient">Mapping Engine (QDA)</span>
          </h1>
          <p className="hero-description">
            Permission-based Android malware profiling framework using Quadratic Discriminant Analysis.
            Captures class-specific covariance structures and non-linear decision boundaries across permission vector sets with optimal regularization (reg_param = 0.1).
          </p>

          <div className="hero-stats">
            <div className="hero-stat">
              <div className="hero-stat-value">90.47%</div>
              <div className="hero-stat-label">Test Accuracy</div>
            </div>
            <div className="hero-stat">
              <div className="hero-stat-value">89.49%</div>
              <div className="hero-stat-label">Macro F1 Score</div>
            </div>
            <div className="hero-stat">
              <div className="hero-stat-value">11,598</div>
              <div className="hero-stat-label">Total Samples</div>
            </div>
            <div className="hero-stat">
              <div className="hero-stat-value">50</div>
              <div className="hero-stat-label">Selected Features</div>
            </div>
            <div className="hero-stat">
              <div className="hero-stat-value">0.10</div>
              <div className="hero-stat-label">Optimal Reg Param</div>
            </div>
          </div>
        </div>
      </div>

      {/* Primary KPI Metrics Grid */}
      <div className="metrics-grid">
        <div className="metric-card blue">
          <div className="metric-label">Overall Accuracy</div>
          <div className="metric-value">
            {(overallMetrics.accuracy * 100).toFixed(2)}<span className="unit">%</span>
          </div>
          <div className="metric-change positive">
            <span>▲</span> 2,099 / 2,320 correctly classified
          </div>
        </div>

        <div className="metric-card emerald">
          <div className="metric-label">Macro F1-Score</div>
          <div className="metric-value">
            {(overallMetrics.macroF1 * 100).toFixed(2)}<span className="unit">%</span>
          </div>
          <div className="metric-change positive">
            <span>▲</span> Unweighted mean across 5 classes
          </div>
        </div>

        <div className="metric-card purple">
          <div className="metric-label">Balanced Accuracy</div>
          <div className="metric-value">
            {(overallMetrics.balancedAccuracy * 100).toFixed(2)}<span className="unit">%</span>
          </div>
          <div className="metric-change positive">
            <span>▲</span> Robust to class imbalance
          </div>
        </div>

        <div className="metric-card cyan">
          <div className="metric-label">Weighted F1-Score</div>
          <div className="metric-value">
            {(overallMetrics.weightedF1 * 100).toFixed(2)}<span className="unit">%</span>
          </div>
          <div className="metric-change positive">
            <span>▲</span> Support-weighted performance
          </div>
        </div>

        <div className="metric-card amber">
          <div className="metric-label">Macro Precision</div>
          <div className="metric-value">
            {(overallMetrics.macroPrecision * 100).toFixed(2)}<span className="unit">%</span>
          </div>
          <div className="metric-change neutral">
            <span>●</span> High purity per alert
          </div>
        </div>

        <div className="metric-card rose">
          <div className="metric-label">Macro Recall</div>
          <div className="metric-value">
            {(overallMetrics.macroRecall * 100).toFixed(2)}<span className="unit">%</span>
          </div>
          <div className="metric-change positive">
            <span>▲</span> Low evasion rate
          </div>
        </div>
      </div>

      {/* Visualizations Grid */}
      <div className="charts-grid">
        <div className="card">
          <div className="card-header">
            <div>
              <div className="card-title">
                <span className="icon">📊</span> Per-Class F1 Performance
              </div>
              <div className="card-subtitle">Harmonic mean of precision and recall per malware family</div>
            </div>
            <Link to="/metrics" style={{ color: 'var(--accent-blue)', fontSize: '12px', textDecoration: 'none', fontWeight: '600' }}>
              View Details →
            </Link>
          </div>
          <div className="chart-container">
            <Bar
              data={f1BarData}
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
                <span className="icon">🍩</span> Test Set Distribution
              </div>
              <div className="card-subtitle">2,320 stratified test instances across 5 categories</div>
            </div>
            <Link to="/confusion-matrix" style={{ color: 'var(--accent-blue)', fontSize: '12px', textDecoration: 'none', fontWeight: '600' }}>
              Matrix →
            </Link>
          </div>
          <div className="chart-container" style={{ display: 'flex', alignItems: 'center', justifyContent: 'center' }}>
            <div style={{ width: '280px', height: '280px' }}>
              <Doughnut
                data={distributionDoughnutData}
                options={{
                  responsive: true,
                  maintainAspectRatio: false,
                  plugins: {
                    legend: {
                      position: 'bottom',
                      labels: {
                        color: '#94a3c8',
                        font: { family: "'Inter', sans-serif", size: 11 },
                        padding: 12,
                      },
                    },
                    tooltip: defaultChartOptions.plugins.tooltip,
                  },
                }}
              />
            </div>
          </div>
        </div>
      </div>

      {/* Research Architectural Highlights */}
      <div className="card" style={{ marginBottom: '28px' }}>
        <div className="card-header">
          <div className="card-title">
            <span className="icon">⚡</span> Key Empirical Discoveries & Architecture
          </div>
          <span className="status-badge best">scikit-learn QDA Engine</span>
        </div>
        <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(300px, 1fr))', gap: '20px' }}>
          <div style={{ padding: '16px', borderRadius: '12px', background: 'rgba(59, 130, 246, 0.05)', border: '1px solid rgba(59, 130, 246, 0.15)' }}>
            <h4 style={{ color: '#60a5fa', marginBottom: '8px', fontSize: '15px' }}>1. Overcoming Covariance Singularity</h4>
            <p style={{ color: '#94a3c8', fontSize: '13px', lineHeight: '1.6' }}>
              Standard QDA with <code style={{ color: '#f43f5e', background: 'rgba(0,0,0,0.3)', padding: '2px 6px', borderRadius: '4px' }}>reg_param = 0.0</code> crashes with <code style={{ color: '#f43f5e' }}>LinAlgError</code> because the <em>SMS Malware</em> permission covariance matrix is not full rank. Applying covariance shrinkage with <code style={{ color: '#10b981' }}>reg_param = 0.10</code> achieves stability with peak 5-fold CV F1 of <strong>0.9076</strong>.
            </p>
          </div>

          <div style={{ padding: '16px', borderRadius: '12px', background: 'rgba(16, 185, 129, 0.05)', border: '1px solid rgba(16, 185, 129, 0.15)' }}>
            <h4 style={{ color: '#34d399', marginBottom: '8px', fontSize: '15px' }}>2. Quadratic Non-Linear Decision Boundaries</h4>
            <p style={{ color: '#94a3c8', fontSize: '13px', lineHeight: '1.6' }}>
              Unlike Linear Discriminant Analysis (LDA), QDA fits an independent covariance matrix $\Sigma_k$ for each of the 5 classes. This accurately captures complex permission correlation structures unique to Banking trojans versus Benign utilities.
            </p>
          </div>

          <div style={{ padding: '16px', borderRadius: '12px', background: 'rgba(139, 92, 246, 0.05)', border: '1px solid rgba(139, 92, 246, 0.15)' }}>
            <h4 style={{ color: '#a78bfa', marginBottom: '8px', fontSize: '15px' }}>3. Exceptional SMS Malware Discrimination</h4>
            <p style={{ color: '#94a3c8', fontSize: '13px', lineHeight: '1.6' }}>
              SMS Malware achieves <strong>98.32% Precision</strong> and <strong>97.44% Recall</strong> (761 / 781 samples identified). The mutual-information selected permission vectors form a distinct high-dimensional cluster for SMS payloads.
            </p>
          </div>
        </div>
      </div>

      {/* Model Spec Quick Info */}
      <div className="card">
        <div className="card-header">
          <div className="card-title">
            <span className="icon">⚙️</span> Experiment Configuration & Environment
          </div>
          <span className="status-badge pass">Verified Contract Compliant</span>
        </div>
        <div className="info-grid">
          <div className="info-item">
            <span className="info-label">Framework Target</span>
            <span className="info-value">Quadratic Probabilistic Mapping</span>
          </div>
          <div className="info-item">
            <span className="info-label">Split Strategy</span>
            <span className="info-value">{experimentMetadata.splitStrategy}</span>
          </div>
          <div className="info-item">
            <span className="info-label">Random Seed</span>
            <span className="info-value">{experimentMetadata.randomSeed}</span>
          </div>
          <div className="info-item">
            <span className="info-label">Selected Features</span>
            <span className="info-value">50 (Mutual Information)</span>
          </div>
          <div className="info-item">
            <span className="info-label">Python Environment</span>
            <span className="info-value">Python {experimentMetadata.packageVersions.python}</span>
          </div>
          <div className="info-item">
            <span className="info-label">Scikit-Learn Version</span>
            <span className="info-value">v{experimentMetadata.packageVersions['scikit-learn']}</span>
          </div>
        </div>
      </div>
    </div>
  );
};

export default OverviewPage;
