import { BrowserRouter, Routes, Route } from 'react-router-dom';
import Layout from './components/Layout';
import Home from './pages/Home';
import PolicyEvaluation from './pages/PolicyEvaluation';
import PolicyIteration from './pages/PolicyIteration';
import ValueIteration from './pages/ValueIteration';
import MonteCarlo from './pages/MonteCarlo';
import TDLearning from './pages/TDLearning';
import SARSA from './pages/SARSA';
import QLearning from './pages/QLearning';
import './App.css';

// Get base path for GitHub Pages deployment
const basename = import.meta.env.BASE_URL || '/';

/**
 * Main App component with routing.
 * Each algorithm has its own dedicated page.
 */
function App() {
  return (
    <BrowserRouter basename={basename}>
      <Routes>
        <Route path="/" element={<Layout />}>
          <Route index element={<Home />} />
          <Route path="policy-evaluation" element={<PolicyEvaluation />} />
          <Route path="policy-iteration" element={<PolicyIteration />} />
          <Route path="value-iteration" element={<ValueIteration />} />
          <Route path="monte-carlo" element={<MonteCarlo />} />
          <Route path="td-learning" element={<TDLearning />} />
          <Route path="sarsa" element={<SARSA />} />
          <Route path="q-learning" element={<QLearning />} />
        </Route>
      </Routes>
    </BrowserRouter>
  );
}

export default App;

