import React from 'react';
import { Line } from 'react-chartjs-2';
import { cvResults } from '../data/experimentData';
import { defaultChartOptions } from '../utils/chartConfig';

const TuningPage = () => {
  // Filter successful values for line chart
  const passResults = cvResults.filter((r) => r.status === 'PASS');

  const lineChartData = {
    labels: passResults.map((r) => `α = ${r.regParam.toFixed(2)}`),
    datasets: [
      {
        label: '5-Fold CV Macro F1 (%)',
        data: passResults.map((r) => +(r.meanF1 * 100).toFixed(2)),
        borderColor: '#3b82f6',
        backgroundColor: 'rgba(59, 130, 246, 0.15)',
        fill: true,
        tension: 0.3,
        pointBackgroundColor: passResults.map((r) =>
          r.regParam === 0.1 ? '#10b981' : '#3b82f6'
        ),
        pointBorderColor: '#ffffff',
        pointRadius: passResults.map((r) => (r.regParam === 0.1 ? 8 : 5)),
        pointHoverRadius: 9,
      },
      {
        label: '+1 Std Dev',
        data: passResults.map((r) => +((r.meanF1 + r.stdF1) * 100).toFixed(2)),
        borderColor: 'rgba(148, 163, 184, 0.4)',
        borderDash: [5, 5],
        borderWidth: 1,
        pointRadius: 0,
        fill: false,
      },
      {
        label: '-1 Std Dev',
        data: passResults.map((r) => +((r.meanF1 - r.stdF1) * 100).toFixed(2)),
        borderColor: 'rgba(148, 163, 184, 0.4)',
        borderDash: [5, 5],
        borderWidth: 1,
        pointRadius: 0,
        fill: false,
      },
    ],
  };

  return (
    <div className="tuning-page animate-in">
      <div className="page-header">
        <h1>Hyperparameter Tuning & Covariance Regularization</h1>
        <p>Mathematical resolution of covariance matrix rank deficiency through parameter sweep ($\alpha \in [0.0, 1.0]$).</p>
      </div>

      {/* Singularity Problem Alert Card */}
      <div className="card" style={{ marginBottom: '28px', borderLeft: '4px solid #f43f5e' }}>
        <div className="card-header">
          <div className="card-title">
            <span className="icon">⚠️</span> The Singularity Dilemma: Why Standard QDA Crashed
          </div>
          <span className="status-badge fail">reg_param = 0.0 FAIL</span>
        </div>
        <p style={{ color: '#cbd5e1', fontSize: '14px', lineHeight: '1.7', marginBottom: '14px' }}>
          When fitting standard Quadratic Discriminant Analysis (<code style={{ color: '#f43f5e' }}>reg_param = 0.0</code>), the model computes the class-conditional covariance matrix:
        </p>
        <div style={{ background: 'rgba(0, 0, 0, 0.3)', padding: '12px 18px', borderRadius: '8px', fontFamily: 'var(--font-mono)', fontSize: '13px', color: '#7dd3fc', marginBottom: '14px' }}>
          Σ_k = (1 / (N_k - 1)) * Σ (x - μ_k)(x - μ_k)ᵀ
        </div>
        <p style={{ color: '#94a3c8', fontSize: '13px', lineHeight: '1.6' }}>
          For the <strong style={{ color: '#f87171' }}>SMS Malware</strong> class, linear dependencies among binary permission vectors caused the determinant $|\Sigma_{SMS}| = 0$, leading to <code style={{ color: '#f87171' }}>numpy.linalg.LinAlgError: SMS Malware covariance not full rank</code>.
          To solve this without dropping predictive features, covariance shrinkage was introduced:
        </p>
        <div style={{ background: 'rgba(0, 0, 0, 0.3)', padding: '12px 18px', borderRadius: '8px', fontFamily: 'var(--font-mono)', fontSize: '13px', color: '#34d399', marginTop: '10px' }}>
          Σ_k(α) = (1 - α) * Σ_k + α * I
        </div>
      </div>

      {/* Parameter Search Cards */}
      <div className="reg-param-grid" style={{ marginBottom: '28px' }}>
        {cvResults.map((r) => (
          <div
            key={r.regParam}
            className={`reg-param-card ${r.status === 'FAIL' ? 'fail' : r.regParam === 0.1 ? 'best' : ''}`}
          >
            <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '8px' }}>
              <span className="reg-param-value">α = {r.regParam.toFixed(2)}</span>
              <span className={`status-badge ${r.status === 'FAIL' ? 'fail' : r.regParam === 0.1 ? 'best' : 'pass'}`}>
                {r.status === 'FAIL' ? 'Singular' : r.regParam === 0.1 ? 'Optimal ★' : 'Pass'}
              </span>
            </div>
            <div className="reg-param-score">
              {r.meanF1 !== null ? (
                <>
                  <span style={{ fontSize: '18px', fontWeight: '800', color: r.regParam === 0.1 ? '#34d399' : 'var(--text-primary)' }}>
                    {(r.meanF1 * 100).toFixed(2)}%
                  </span>
                  <span style={{ fontSize: '11px', color: 'var(--text-muted)', display: 'block', marginTop: '2px' }}>
                    ± {(r.stdF1 * 100).toFixed(2)}% (5-fold)
                  </span>
                </>
              ) : (
                <span style={{ color: '#f87171', fontSize: '12px' }}>Covariance not full rank</span>
              )}
            </div>
          </div>
        ))}
      </div>

      {/* CV Curve & Official Plot */}
      <div className="charts-grid">
        <div className="card">
          <div className="card-header">
            <div>
              <div className="card-title">
                <span className="icon">📈</span> Cross-Validation Macro F1 Curve
              </div>
              <div className="card-subtitle">Peak performance at α = 0.10, degrading as α approaches 1.0</div>
            </div>
          </div>
          <div className="chart-container">
            <Line
              data={lineChartData}
              options={{
                ...defaultChartOptions,
                scales: {
                  ...defaultChartOptions.scales,
                  y: {
                    ...defaultChartOptions.scales.y,
                    min: 65,
                    max: 95,
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
                <span className="icon">🖼️</span> Paper Figure: Regularization Search
              </div>
              <div className="card-subtitle">Generated publication figure from QDA pipeline</div>
            </div>
          </div>
          <div style={{ textAlign: 'center' }}>
            <img
              src="/plots/reg_param_search.png"
              alt="Reg Param Search"
              style={{ maxHeight: '310px', maxWidth: '100%', borderRadius: '8px', border: '1px solid var(--border-subtle)' }}
            />
          </div>
        </div>
      </div>

      {/* Fold Breakdown Table */}
      <div className="card">
        <div className="card-header">
          <div className="card-title">
            <span className="icon">📊</span> 5-Fold Stratified Cross-Validation Breakdown
          </div>
          <span className="status-badge best">Fixed Seed = 42</span>
        </div>
        <table className="data-table">
          <thead>
            <tr>
              <th>reg_param (α)</th>
              <th>Status</th>
              <th>Mean F1</th>
              <th>Fold 1</th>
              <th>Fold 2</th>
              <th>Fold 3</th>
              <th>Fold 4</th>
              <th>Fold 5</th>
              <th>Std Dev</th>
            </tr>
          </thead>
          <tbody>
            {cvResults.map((r) => (
              <tr key={r.regParam} style={r.regParam === 0.1 ? { background: 'rgba(59, 130, 246, 0.08)' } : {}}>
                <td style={{ fontFamily: 'var(--font-mono)', fontWeight: '700', color: r.regParam === 0.1 ? '#60a5fa' : 'var(--text-primary)' }}>
                  {r.regParam.toFixed(2)} {r.regParam === 0.1 && '★'}
                </td>
                <td>
                  <span className={`status-badge ${r.status === 'FAIL' ? 'fail' : r.regParam === 0.1 ? 'best' : 'pass'}`}>
                    {r.status}
                  </span>
                </td>
                <td className="table-value" style={{ color: r.regParam === 0.1 ? '#34d399' : 'var(--text-primary)' }}>
                  {r.meanF1 !== null ? `${(r.meanF1 * 100).toFixed(2)}%` : '—'}
                </td>
                {r.folds ? (
                  r.folds.map((fold, idx) => (
                    <td key={idx} style={{ fontFamily: 'var(--font-mono)', fontSize: '12px' }}>
                      {(fold * 100).toFixed(2)}%
                    </td>
                  ))
                ) : (
                  <td colSpan={5} style={{ color: '#f87171', fontSize: '12px' }}>
                    {r.error}
                  </td>
                )}
                <td style={{ fontFamily: 'var(--font-mono)', fontSize: '12px', color: 'var(--text-muted)' }}>
                  {r.stdF1 !== null ? `± ${(r.stdF1 * 100).toFixed(2)}%` : '—'}
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  );
};

export default TuningPage;
