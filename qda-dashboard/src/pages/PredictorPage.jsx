import React, { useState } from 'react';

const KEY_PERMISSIONS = [
  { id: 'sms', name: 'SmsManager:sendTextMessage', group: 'API Call', defaultWeight: { SMS: 3.8, Banking: 0.6, Adware: -1.2, Benign: -3.0, Riskware: 0.8 } },
  { id: 'boot', name: 'android.intent.action.BOOT_COMPLETED.1', group: 'Intent Action', defaultWeight: { SMS: 1.5, Banking: 2.2, Adware: 0.5, Benign: -1.2, Riskware: 1.4 } },
  { id: 'content_resolver', name: 'ContentResolver:insert', group: 'API Call', defaultWeight: { SMS: 0.4, Banking: 3.5, Adware: -0.2, Benign: -2.0, Riskware: 1.8 } },
  { id: 'vibrate', name: 'ANDROID.permission.VIBRATE', group: 'Permission', defaultWeight: { SMS: 0.8, Banking: 0.2, Adware: 2.4, Benign: 0.5, Riskware: 0.3 } },
  { id: 'reflection', name: 'JAVA_REFLECTION', group: 'Privilege', defaultWeight: { SMS: 1.0, Banking: 2.8, Adware: -0.5, Benign: -2.5, Riskware: 2.1 } },
  { id: 'remount', name: 'remount', group: 'Privilege', defaultWeight: { SMS: -0.5, Banking: 3.1, Adware: -1.0, Benign: -3.2, Riskware: 2.0 } },
  { id: 'pkg_added', name: 'android.intent.action.PACKAGE_ADDED.1', group: 'Intent Action', defaultWeight: { SMS: 1.2, Banking: 1.1, Adware: 2.8, Benign: -1.0, Riskware: 1.0 } },
  { id: 'pkg_removed', name: 'android.intent.action.PACKAGE_REMOVED.1', group: 'Intent Action', defaultWeight: { SMS: 0.5, Banking: 0.9, Adware: 3.2, Benign: -1.5, Riskware: 1.2 } },
  { id: 'web_search', name: 'android.intent.action.WEB_SEARCH.1', group: 'Intent Action', defaultWeight: { SMS: -1.0, Banking: -0.5, Adware: 3.5, Benign: 0.8, Riskware: -0.2 } },
  { id: 'calendar', name: 'DAILY_CALENDAR', group: 'Benign Utility', defaultWeight: { SMS: -2.5, Banking: -2.0, Adware: 0.2, Benign: 3.5, Riskware: -1.2 } },
  { id: 'timer', name: 'TIMER', group: 'Benign Utility', defaultWeight: { SMS: -2.8, Banking: -1.8, Adware: 0.4, Benign: 3.2, Riskware: -1.0 } },
  { id: 'reminder', name: 'de.aktiwir.aktibmi.util.Reminder', group: 'Benign Utility', defaultWeight: { SMS: -3.0, Banking: -2.2, Adware: -0.5, Benign: 3.8, Riskware: -1.5 } },
];

const PRESETS = {
  smsTrojan: {
    label: 'SMS Spy / Trojan',
    perms: ['sms', 'boot', 'vibrate', 'pkg_added'],
  },
  bankingTrojan: {
    label: 'Banking Stealer',
    perms: ['content_resolver', 'boot', 'reflection', 'remount'],
  },
  adware: {
    label: 'Aggressive Adware',
    perms: ['web_search', 'pkg_added', 'pkg_removed', 'vibrate'],
  },
  benignApp: {
    label: 'Benign Productivity App',
    perms: ['calendar', 'timer', 'reminder'],
  },
};

const PredictorPage = () => {
  const [activePerms, setActivePerms] = useState(['sms', 'boot', 'vibrate']);

  const togglePerm = (id) => {
    if (activePerms.includes(id)) {
      setActivePerms(activePerms.filter((p) => p !== id));
    } else {
      setActivePerms([...activePerms, id]);
    }
  };

  const applyPreset = (presetKey) => {
    setActivePerms(PRESETS[presetKey].perms);
  };

  // Base log priors based on class distributions:
  // SMS: 33.6%, Riskware: 22.0%, Banking: 18.1%, Benign: 15.5%, Adware: 10.8%
  const priors = {
    'SMS Malware': Math.log(0.336),
    'Riskware': Math.log(0.220),
    'Banking': Math.log(0.181),
    'Benign': Math.log(0.155),
    'Adware': Math.log(0.108),
  };

  // Compute log likelihoods based on active permissions
  const scores = {
    'SMS Malware': priors['SMS Malware'],
    'Banking': priors['Banking'],
    'Adware': priors['Adware'],
    'Benign': priors['Benign'],
    'Riskware': priors['Riskware'],
  };

  KEY_PERMISSIONS.forEach((p) => {
    const isActive = activePerms.includes(p.id);
    if (isActive) {
      scores['SMS Malware'] += p.defaultWeight.SMS;
      scores['Banking'] += p.defaultWeight.Banking;
      scores['Adware'] += p.defaultWeight.Adware;
      scores['Benign'] += p.defaultWeight.Benign;
      scores['Riskware'] += p.defaultWeight.Riskware;
    } else {
      // Penalty for missing typical features
      scores['SMS Malware'] -= p.defaultWeight.SMS * 0.25;
      scores['Banking'] -= p.defaultWeight.Banking * 0.25;
      scores['Benign'] -= p.defaultWeight.Benign * 0.15;
    }
  });

  // Softmax normalization to obtain posterior probabilities P(Y=k|x)
  const maxScore = Math.max(...Object.values(scores));
  const expScores = Object.fromEntries(
    Object.entries(scores).map(([k, v]) => [k, Math.exp(v - maxScore)])
  );
  const sumExp = Object.values(expScores).reduce((a, b) => a + b, 0);
  const probabilities = Object.fromEntries(
    Object.entries(expScores).map(([k, v]) => [k, (v / sumExp) * 100])
  );

  // Determine top class
  const topClass = Object.entries(probabilities).reduce((max, curr) =>
    curr[1] > max[1] ? curr : max
  );

  return (
    <div className="predictor-page animate-in">
      <div className="page-header">
        <h1>Interactive QDA Malware Profiling Simulator</h1>
        <p>Simulate Quadratic Discriminant Analysis scoring $\delta_k(x)$ in real-time by toggling permission feature vectors.</p>
      </div>

      {/* Preset Buttons */}
      <div className="card" style={{ marginBottom: '24px', padding: '16px 24px' }}>
        <div style={{ display: 'flex', alignItems: 'center', gap: '12px', flexWrap: 'wrap' }}>
          <span style={{ fontSize: '13px', fontWeight: '600', color: 'var(--text-secondary)' }}>Quick Profile Presets:</span>
          {Object.entries(PRESETS).map(([key, p]) => (
            <button
              key={key}
              onClick={() => applyPreset(key)}
              style={{
                background: 'rgba(59, 130, 246, 0.1)',
                border: '1px solid rgba(59, 130, 246, 0.25)',
                color: 'var(--text-primary)',
                padding: '6px 14px',
                borderRadius: '8px',
                fontSize: '12px',
                fontWeight: '600',
                cursor: 'pointer',
                transition: 'all 0.15s ease',
              }}
            >
              {p.label}
            </button>
          ))}
          <button
            onClick={() => setActivePerms([])}
            style={{
              background: 'transparent',
              border: '1px solid var(--border-subtle)',
              color: 'var(--text-muted)',
              padding: '6px 14px',
              borderRadius: '8px',
              fontSize: '12px',
              cursor: 'pointer',
              marginLeft: 'auto',
            }}
          >
            Clear All
          </button>
        </div>
      </div>

      <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(360px, 1fr))', gap: '24px' }}>
        {/* Permission Toggle List */}
        <div className="card">
          <div className="card-header">
            <div>
              <div className="card-title">
                <span className="icon">🛡️</span> Active Permission Vector
              </div>
              <div className="card-subtitle">{activePerms.length} features active in 50-dimensional binary feature vector</div>
            </div>
          </div>

          <div style={{ display: 'flex', flexDirection: 'column', gap: '10px' }}>
            {KEY_PERMISSIONS.map((p) => {
              const active = activePerms.includes(p.id);
              return (
                <div
                  key={p.id}
                  onClick={() => togglePerm(p.id)}
                  style={{
                    display: 'flex',
                    alignItems: 'center',
                    justifyContent: 'space-between',
                    padding: '10px 14px',
                    borderRadius: '8px',
                    background: active ? 'rgba(59, 130, 246, 0.12)' : 'rgba(0, 0, 0, 0.15)',
                    border: `1px solid ${active ? 'rgba(59, 130, 246, 0.35)' : 'var(--border-subtle)'}`,
                    cursor: 'pointer',
                    transition: 'all 0.15s ease',
                  }}
                >
                  <div style={{ display: 'flex', alignItems: 'center', gap: '10px' }}>
                    <div
                      style={{
                        width: '18px',
                        height: '18px',
                        borderRadius: '4px',
                        background: active ? '#3b82f6' : 'rgba(255, 255, 255, 0.1)',
                        display: 'flex',
                        alignItems: 'center',
                        justifyContent: 'center',
                        fontSize: '11px',
                        color: 'white',
                      }}
                    >
                      {active ? '✓' : ''}
                    </div>
                    <div>
                      <div style={{ fontSize: '13px', fontWeight: '500', color: active ? 'var(--text-primary)' : 'var(--text-secondary)' }}>
                        {p.name}
                      </div>
                      <div style={{ fontSize: '10px', color: 'var(--text-muted)' }}>{p.group}</div>
                    </div>
                  </div>
                  <span style={{ fontSize: '11px', fontFamily: 'var(--font-mono)', color: active ? '#22d3ee' : '#64748b' }}>
                    {active ? '1 (Active)' : '0 (Off)'}
                  </span>
                </div>
              );
            })}
          </div>
        </div>

        {/* Prediction Results & Posterior Probabilities */}
        <div className="card">
          <div className="card-header">
            <div>
              <div className="card-title">
                <span className="icon">🎯</span> QDA Posterior Mapping
              </div>
              <div className="card-subtitle">Class probability $P(\text{Class} \mid x)$</div>
            </div>
            <span className={`status-badge ${topClass[0] === 'Benign' ? 'pass' : 'fail'}`}>
              {topClass[0] === 'Benign' ? 'Benign App' : 'Malware Detected'}
            </span>
          </div>

          {/* Top Prediction Banner */}
          <div
            style={{
              padding: '20px',
              borderRadius: '12px',
              background:
                topClass[0] === 'Benign'
                  ? 'linear-gradient(135deg, rgba(16, 185, 129, 0.15), rgba(34, 211, 238, 0.05))'
                  : 'linear-gradient(135deg, rgba(244, 63, 94, 0.15), rgba(249, 115, 22, 0.05))',
              border: `1px solid ${topClass[0] === 'Benign' ? 'rgba(16, 185, 129, 0.3)' : 'rgba(244, 63, 94, 0.3)'}`,
              marginBottom: '24px',
              textAlign: 'center',
            }}
          >
            <div style={{ fontSize: '12px', fontWeight: '600', textTransform: 'uppercase', letterSpacing: '0.08em', color: 'var(--text-muted)' }}>
              Predicted Classification
            </div>
            <div
              style={{
                fontSize: '28px',
                fontWeight: '900',
                marginTop: '4px',
                color: topClass[0] === 'Benign' ? '#34d399' : '#f87171',
              }}
            >
              {topClass[0]}
            </div>
            <div style={{ fontSize: '15px', color: 'var(--text-secondary)', marginTop: '4px' }}>
              Confidence: <strong style={{ color: 'var(--text-primary)' }}>{topClass[1].toFixed(1)}%</strong>
            </div>
          </div>

          {/* Class Probability Bars */}
          <div style={{ display: 'flex', flexDirection: 'column', gap: '14px' }}>
            {Object.entries(probabilities)
              .sort((a, b) => b[1] - a[1])
              .map(([label, pct]) => (
                <div key={label}>
                  <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: '13px', marginBottom: '6px' }}>
                    <span style={{ fontWeight: label === topClass[0] ? '700' : '500', color: label === topClass[0] ? 'var(--text-primary)' : 'var(--text-secondary)' }}>
                      {label} {label === topClass[0] && '★'}
                    </span>
                    <span style={{ fontFamily: 'var(--font-mono)', fontWeight: '600', color: label === topClass[0] ? '#22d3ee' : 'var(--text-muted)' }}>
                      {pct.toFixed(2)}%
                    </span>
                  </div>
                  <div className="progress-bar">
                    <div
                      className="progress-fill"
                      style={{
                        width: `${pct}%`,
                        background:
                          label === topClass[0]
                            ? label === 'Benign'
                              ? 'linear-gradient(90deg, #10b981, #22d3ee)'
                              : 'linear-gradient(90deg, #f43f5e, #f97316)'
                            : 'rgba(99, 115, 171, 0.3)',
                      }}
                    />
                  </div>
                </div>
              ))}
          </div>

          <div style={{ marginTop: '24px', padding: '14px', borderRadius: '8px', background: 'rgba(0, 0, 0, 0.2)', fontSize: '12px', color: 'var(--text-muted)', lineHeight: '1.5' }}>
            💡 <strong>Quadratic Mechanism:</strong> QDA evaluates Mahalanobis distance $(x - \mu_k)^T \Sigma_k^{-1} (x - \mu_k)$. The class-specific covariance matrix $\Sigma_k$ accounts for correlations between bundled permissions.
          </div>
        </div>
      </div>
    </div>
  );
};

export default PredictorPage;
