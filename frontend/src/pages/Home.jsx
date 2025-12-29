import { Link } from 'react-router-dom';
import './Home.css';

/**
 * Home page with overview and links to all algorithms.
 */
function Home() {
    const algorithms = [
        {
            category: 'Dynamic Programming',
            description: 'Model-based methods that require complete knowledge of environment dynamics.',
            items: [
                { name: 'Policy Evaluation', path: '/policy-evaluation', desc: 'Evaluate a given policy' },
                { name: 'Policy Iteration', path: '/policy-iteration', desc: 'Iterate between evaluation and improvement' },
                { name: 'Value Iteration', path: '/value-iteration', desc: 'Directly compute optimal values' },
            ]
        },
        {
            category: 'Model-Free Methods',
            description: 'Learn directly from experience without knowing environment dynamics.',
            items: [
                { name: 'Monte Carlo', path: '/monte-carlo', desc: 'Learn from complete episodes' },
                { name: 'TD Learning', path: '/td-learning', desc: 'Bootstrap from current estimates' },
                { name: 'SARSA', path: '/sarsa', desc: 'On-policy TD control' },
                { name: 'Q-Learning', path: '/q-learning', desc: 'Off-policy TD control' },
            ]
        }
    ];

    return (
        <div className="home">
            <section className="hero">
                <h1>Reinforcement Learning Visualizer</h1>
                <p>Explore and understand core RL algorithms through interactive visualizations</p>
            </section>

            <section className="algorithms">
                {algorithms.map(category => (
                    <div key={category.category} className="category">
                        <h2>{category.category}</h2>
                        <p className="category-desc">{category.description}</p>
                        <div className="algorithm-grid">
                            {category.items.map(algo => (
                                <Link key={algo.path} to={algo.path} className="algorithm-card">
                                    <h3>{algo.name}</h3>
                                    <p>{algo.desc}</p>
                                </Link>
                            ))}
                        </div>
                    </div>
                ))}
            </section>
        </div>
    );
}

export default Home;
