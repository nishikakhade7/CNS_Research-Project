import React from 'react';
import { HashRouter as Router, Routes, Route } from 'react-router-dom';
import Sidebar from './components/Sidebar';
import OverviewPage from './pages/OverviewPage';
import MetricsPage from './pages/MetricsPage';
import ConfusionMatrixPage from './pages/ConfusionMatrixPage';
import TuningPage from './pages/TuningPage';
import FeaturesPage from './pages/FeaturesPage';
import PredictorPage from './pages/PredictorPage';
import ResearchPage from './pages/ResearchPage';

function App() {
  return (
    <Router>
      <div className="app-layout">
        <Sidebar />
        <main className="main-content">
          <Routes>
            <Route path="/" element={<OverviewPage />} />
            <Route path="/metrics" element={<MetricsPage />} />
            <Route path="/confusion-matrix" element={<ConfusionMatrixPage />} />
            <Route path="/tuning" element={<TuningPage />} />
            <Route path="/features" element={<FeaturesPage />} />
            <Route path="/predict" element={<PredictorPage />} />
            <Route path="/research" element={<ResearchPage />} />
          </Routes>
        </main>
      </div>
    </Router>
  );
}

export default App;
