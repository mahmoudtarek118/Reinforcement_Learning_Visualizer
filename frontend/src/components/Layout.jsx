import { Link, Outlet } from 'react-router-dom';
import './Layout.css';

/**
 * Shared Layout component with navigation.
 * Wraps all pages with consistent header and navigation.
 */
function Layout() {
    return (
        <div className="layout">
            <header className="header">
                <div className="header-content">
                    <Link to="/" className="logo">
                        <h1>RL Learning Tool</h1>
                    </Link>
                    <nav className="nav">
                        <div className="nav-group">
                            <span className="nav-label">Dynamic Programming</span>
                            <Link to="/policy-evaluation">Policy Evaluation</Link>
                            <Link to="/policy-iteration">Policy Iteration</Link>
                            <Link to="/value-iteration">Value Iteration</Link>
                        </div>
                        <div className="nav-group">
                            <span className="nav-label">Model-Free</span>
                            <Link to="/monte-carlo">Monte Carlo</Link>
                            <Link to="/td-learning">TD Learning</Link>
                            <Link to="/sarsa">SARSA</Link>
                            <Link to="/q-learning">Q-Learning</Link>
                        </div>
                    </nav>
                </div>
            </header>
            <main className="main">
                <Outlet />
            </main>
            <footer className="footer">
                <p>Educational RL Web Tool</p>
            </footer>
        </div>
    );
}

export default Layout;
