import React, { useState } from 'react';
import { confusionMatrix, classLabels } from '../data/experimentData';

const ConfusionMatrixPage = () => {
  const [mode, setMode] = useState('raw'); // 'raw' | 'normalized'
  const [selectedCell, setSelectedCell] = useState(null);
  const [viewPlotModal, setViewPlotModal] = useState(false);

  // Compute row totals (support per true class)
  const rowTotals = confusionMatrix.map((row) => row.reduce((a, b) => a + b, 0));

  // Compute normalized matrix (row percentages)
  const normalizedMatrix = confusionMatrix.map((row, i) =>
    row.map((val) => (val / rowTotals[i]) * 100)
  );

  // Find max value in raw matrix for opacity scaling
  const maxRaw = Math.max(...confusionMatrix.flat());

  const getCellBackground = (rowIdx, colIdx, val) => {
    const isDiagonal = rowIdx === colIdx;
    if (isDiagonal) {
      const pct = normalizedMatrix[rowIdx][colIdx] / 100;
      return `rgba(16, 185, 129, ${0.15 + pct * 0.75})`;
    } else {
      if (val === 0) return 'rgba(255, 255, 255, 0.02)';
      const ratio = val / rowTotals[rowIdx];
      return `rgba(244, 63, 94, ${0.1 + Math.min(ratio * 1.5, 0.85)})`;
    }
  };

  const getCellTextColor = (rowIdx, colIdx) => {
    const isDiagonal = rowIdx === colIdx;
    return isDiagonal ? '#34d399' : '#f87171';
  };

  return (
    <div className="confusion-matrix-page animate-in">
      <div className="page-header">
        <h1>Confusion Matrix & Error Profiling</h1>
        <p>Interactive quadratic decision mapping matrix across 5 Android malware classes on the 20% holdout test set.</p>
      </div>

      {/* Control Bar */}
      <div className="card" style={{ marginBottom: '24px', padding: '16px 24px' }}>
        <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', flexWrap: 'wrap', gap: '16px' }}>
          <div style={{ display: 'flex', alignItems: 'center', gap: '12px' }}>
            <span style={{ fontSize: '13px', fontWeight: '600', color: 'var(--text-secondary)' }}>Display Mode:</span>
            <div className="toggle-group">
              <button
                className={`toggle-btn ${mode === 'raw' ? 'active' : ''}`}
                onClick={() => setMode('raw')}
              >
                Raw Counts (N)
              </button>
              <button
                className={`toggle-btn ${mode === 'normalized' ? 'active' : ''}`}
                onClick={() => setMode('normalized')}
              >
                Normalized (%)
              </button>
            </div>
          </div>

          <div style={{ display: 'flex', gap: '12px', alignItems: 'center' }}>
            <span style={{ fontSize: '12px', color: 'var(--text-muted)' }}>
              🟢 Diagonal = Correct | 🔴 Off-Diagonal = Misclassified
            </span>
            <button
              onClick={() => setViewPlotModal(!viewPlotModal)}
              style={{
                background: 'rgba(59, 130, 246, 0.15)',
                color: 'var(--accent-blue)',
                border: '1px solid rgba(59, 130, 246, 0.3)',
                borderRadius: '8px',
                padding: '6px 14px',
                fontSize: '12px',
                fontWeight: '600',
                cursor: 'pointer',
              }}
            >
              {viewPlotModal ? 'Hide Publication Plots' : 'View Publication Plots 🖼️'}
            </button>
          </div>
        </div>
      </div>

      {/* Publication Plots Section (Collapsible) */}
      {viewPlotModal && (
        <div className="card" style={{ marginBottom: '24px' }}>
          <div className="card-header">
            <div className="card-title">
              <span className="icon">🖼️</span> High-Resolution IEEE Paper Figures
            </div>
            <span className="status-badge pass">Generated from QDA Run</span>
          </div>
          <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(380px, 1fr))', gap: '20px' }}>
            <div>
              <h4 style={{ color: '#94a3c8', fontSize: '13px', marginBottom: '8px' }}>Figure: Confusion Matrix (Raw Counts)</h4>
              <img
                src="/plots/confusion_matrix.png"
                alt="Confusion Matrix"
                style={{ width: '100%', borderRadius: '10px', border: '1px solid var(--border-subtle)' }}
              />
            </div>
            <div>
              <h4 style={{ color: '#94a3c8', fontSize: '13px', marginBottom: '8px' }}>Figure: Normalized Confusion Matrix</h4>
              <img
                src="/plots/confusion_matrix_normalized.png"
                alt="Normalized Confusion Matrix"
                style={{ width: '100%', borderRadius: '10px', border: '1px solid var(--border-subtle)' }}
              />
            </div>
          </div>
        </div>
      )}

      {/* Interactive Matrix Card */}
      <div className="card" style={{ marginBottom: '28px' }}>
        <div className="card-header">
          <div>
            <div className="card-title">
              <span className="icon">🔲</span> Interactive Quadratic Decision Matrix
            </div>
            <div className="card-subtitle">Click any cell to inspect true vs predicted characteristics</div>
          </div>
        </div>

        <div className="confusion-matrix-wrapper" style={{ padding: '20px 0' }}>
          <div style={{ display: 'flex', flexDirection: 'column', alignItems: 'center' }}>
            <div className="cm-axis-label" style={{ marginBottom: '12px' }}>
              Predicted Class →
            </div>

            <div style={{ display: 'flex' }}>
              <div style={{ writingMode: 'vertical-rl', transform: 'rotate(180deg)', display: 'flex', alignItems: 'center', justifyContent: 'center', marginRight: '8px' }}>
                <span className="cm-axis-label">Actual (True) Class →</span>
              </div>

              <div>
                {/* Column Headers */}
                <div style={{ display: 'flex', marginLeft: '104px', marginBottom: '4px' }}>
                  {classLabels.map((label) => (
                    <div key={label} className="cm-header" style={{ width: '90px' }}>
                      {label}
                    </div>
                  ))}
                  <div className="cm-header" style={{ width: '80px', color: '#a78bfa' }}>
                    Total
                  </div>
                </div>

                {/* Matrix Rows */}
                {confusionMatrix.map((row, rowIdx) => (
                  <div key={rowIdx} style={{ display: 'flex', alignItems: 'center', marginBottom: '4px' }}>
                    {/* Row Header */}
                    <div className="cm-row-label" style={{ width: '100px' }}>
                      {classLabels[rowIdx]}
                    </div>

                    {/* Cells */}
                    {row.map((val, colIdx) => {
                      const isSelected = selectedCell?.row === rowIdx && selectedCell?.col === colIdx;
                      const normVal = normalizedMatrix[rowIdx][colIdx];
                      return (
                        <div
                          key={colIdx}
                          className="cm-cell"
                          onClick={() =>
                            setSelectedCell({
                              row: rowIdx,
                              col: colIdx,
                              trueClass: classLabels[rowIdx],
                              predClass: classLabels[colIdx],
                              count: val,
                              pct: normVal,
                              isCorrect: rowIdx === colIdx,
                            })
                          }
                          style={{
                            width: '90px',
                            marginRight: '4px',
                            background: getCellBackground(rowIdx, colIdx, val),
                            color: getCellTextColor(rowIdx, colIdx),
                            border: isSelected ? '2px solid #22d3ee' : '1px solid rgba(255,255,255,0.05)',
                            cursor: 'pointer',
                          }}
                          title={`True: ${classLabels[rowIdx]} | Pred: ${classLabels[colIdx]} | ${val} (${normVal.toFixed(1)}%)`}
                        >
                          {mode === 'raw' ? val : `${normVal.toFixed(1)}%`}
                        </div>
                      );
                    })}

                    {/* Row Total */}
                    <div
                      style={{
                        width: '80px',
                        display: 'flex',
                        alignItems: 'center',
                        justifyContent: 'center',
                        fontFamily: 'var(--font-mono)',
                        fontSize: '13px',
                        fontWeight: '600',
                        color: 'var(--text-secondary)',
                      }}
                    >
                      {rowTotals[rowIdx]}
                    </div>
                  </div>
                ))}
              </div>
            </div>
          </div>
        </div>

        {/* Selected Cell Inspector */}
        {selectedCell ? (
          <div
            style={{
              marginTop: '16px',
              padding: '16px 20px',
              borderRadius: '12px',
              background: selectedCell.isCorrect ? 'rgba(16, 185, 129, 0.08)' : 'rgba(244, 63, 94, 0.08)',
              border: `1px solid ${selectedCell.isCorrect ? 'rgba(16, 185, 129, 0.3)' : 'rgba(244, 63, 94, 0.3)'}`,
            }}
          >
            <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '8px' }}>
              <span style={{ fontWeight: '700', fontSize: '15px', color: selectedCell.isCorrect ? '#34d399' : '#f87171' }}>
                {selectedCell.isCorrect ? '✅ Correct Classification' : '⚠️ Misclassification Pair'}
              </span>
              <button
                onClick={() => setSelectedCell(null)}
                style={{ background: 'transparent', border: 'none', color: 'var(--text-muted)', cursor: 'pointer', fontSize: '16px' }}
              >
                ✕
              </button>
            </div>
            <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(200px, 1fr))', gap: '16px', fontSize: '13px' }}>
              <div>
                <span style={{ color: 'var(--text-muted)' }}>Actual Class: </span>
                <strong style={{ color: 'var(--text-primary)' }}>{selectedCell.trueClass}</strong>
              </div>
              <div>
                <span style={{ color: 'var(--text-muted)' }}>Predicted Class: </span>
                <strong style={{ color: 'var(--text-primary)' }}>{selectedCell.predClass}</strong>
              </div>
              <div>
                <span style={{ color: 'var(--text-muted)' }}>Instances: </span>
                <strong style={{ fontFamily: 'var(--font-mono)', color: 'var(--text-primary)' }}>
                  {selectedCell.count} samples ({selectedCell.pct.toFixed(2)}% of actual {selectedCell.trueClass})
                </strong>
              </div>
            </div>
          </div>
        ) : (
          <div style={{ textAlign: 'center', padding: '12px', color: 'var(--text-muted)', fontSize: '12px' }}>
            Tip: Click any cell above to inspect the specific classification transfer.
          </div>
        )}
      </div>

      {/* Error Analysis & Insights */}
      <div className="card">
        <div className="card-header">
          <div className="card-title">
            <span className="icon">🔍</span> Critical Misclassification Insights
          </div>
        </div>
        <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(280px, 1fr))', gap: '16px' }}>
          <div style={{ padding: '16px', borderRadius: '10px', background: 'rgba(255,255,255,0.02)', border: '1px solid var(--border-subtle)' }}>
            <h4 style={{ color: '#f59e0b', fontSize: '14px', marginBottom: '6px' }}>Riskware vs Banking (48 errors)</h4>
            <p style={{ color: '#94a3c8', fontSize: '12.5px', lineHeight: '1.5' }}>
              Riskware shares significant overlap in privileged permissions (<code style={{ color: '#f59e0b' }}>SMS</code>, <code style={{ color: '#f59e0b' }}>ContentResolver</code>) with modern Banking trojans, leading to 48 Riskware samples being mapped to Banking.
            </p>
          </div>

          <div style={{ padding: '16px', borderRadius: '10px', background: 'rgba(255,255,255,0.02)', border: '1px solid var(--border-subtle)' }}>
            <h4 style={{ color: '#38bdf8', fontSize: '14px', marginBottom: '6px' }}>Riskware vs Benign (37 errors)</h4>
            <p style={{ color: '#94a3c8', fontSize: '12.5px', lineHeight: '1.5' }}>
              Mild Riskware utilities that primarily request innocuous utility permissions closely resemble benign productivity apps on the quadratic boundary.
            </p>
          </div>

          <div style={{ padding: '16px', borderRadius: '10px', background: 'rgba(255,255,255,0.02)', border: '1px solid var(--border-subtle)' }}>
            <h4 style={{ color: '#10b981', fontSize: '14px', marginBottom: '6px' }}>SMS Malware Isolation (97.4% accuracy)</h4>
            <p style={{ color: '#94a3c8', fontSize: '12.5px', lineHeight: '1.5' }}>
              761 out of 781 SMS Malware samples are cleanly separated. Only 13 leaked into Banking and 5 into Benign, proving that SMS-related APIs form an isolated quadratic cluster.
            </p>
          </div>
        </div>
      </div>
    </div>
  );
};

export default ConfusionMatrixPage;
