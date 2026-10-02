import React, { useState, useEffect, useMemo } from 'react';
import { 
  Package, 
  TrendingUp, 
  TrendingDown, 
  Minus, 
  AlertTriangle, 
  ShieldCheck, 
  CheckCircle2, 
  X, 
  Search, 
  Sparkles, 
  ArrowRightLeft, 
  Info, 
  ThumbsUp, 
  Edit3, 
  Ban,
  LayoutGrid,
  Table as TableIcon,
  ArrowUp,
  ArrowDown,
  Boxes,
  BarChart3
} from 'lucide-react';

export default function App() {
  const [products, setProducts] = useState([]);
  const [summary, setSummary] = useState({ total: 0, increase: 0, maintain: 0, reduce: 0, high_risk: 0 });
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);
  const [healthStatus, setHealthStatus] = useState(null);

  // View Mode: 'table' or 'kanban'
  const [viewMode, setViewMode] = useState('table');

  // Filters & Search
  const [searchQuery, setSearchQuery] = useState('');
  const [decisionFilter, setDecisionFilter] = useState('ALL');
  const [riskFilter, setRiskFilter] = useState('ALL');
  const [categoryFilter, setCategoryFilter] = useState('ALL');

  // Sorting
  const [sortField, setSortField] = useState('product_id');
  const [sortDirection, setSortDirection] = useState('asc'); // 'asc' or 'desc'

  // Selected Product & Detail Modal
  const [selectedProduct, setSelectedProduct] = useState(null);
  const [productDetailLoading, setProductDetailLoading] = useState(false);
  const [productDetailData, setProductDetailData] = useState(null);
  const [activeDetailTab, setActiveDetailTab] = useState('overview'); // 'overview', 'ai_rationale', 'alternatives', 'approval'

  // Human Approval State
  const [customQty, setCustomQty] = useState(0);
  const [approvalNotes, setApprovalNotes] = useState('');
  const [toastMessage, setToastMessage] = useState(null);

  // Load initial data
  useEffect(() => {
    fetchHealth();
    fetchProducts();
  }, []);

  const fetchHealth = async () => {
    try {
      const res = await fetch('/health');
      if (res.ok) {
        const data = await res.json();
        setHealthStatus(data);
      }
    } catch (e) {
      console.warn("Backend health check not reached:", e);
    }
  };

  const fetchProducts = async () => {
    setLoading(true);
    setError(null);
    try {
      const res = await fetch('/products');
      if (!res.ok) throw new Error(`HTTP error ${res.status}`);
      const data = await res.json();
      setProducts(data.products || []);
      setSummary(data.summary || { total: 0, increase: 0, maintain: 0, reduce: 0, high_risk: 0 });
    } catch (err) {
      setError(err.message || 'Failed to connect to decision agent API.');
    } finally {
      setLoading(false);
    }
  };

  // Open Product Details
  const handleOpenDetail = async (product, initialTab = 'overview') => {
    setSelectedProduct(product);
    setActiveDetailTab(initialTab);
    setProductDetailLoading(true);
    setProductDetailData(null);
    setApprovalNotes('');

    try {
      const res = await fetch(`/products/${product.product_id}`);
      if (res.ok) {
        const data = await res.json();
        setProductDetailData(data);
        const recQty = data.decision?.recommended_order_quantity ?? product.recommended_order_quantity;
        setCustomQty(recQty);
      }
    } catch (e) {
      console.error("Failed to load product details:", e);
    } finally {
      setProductDetailLoading(false);
    }
  };

  // Human Approval Action Submission
  const handleApprovalSubmit = async (action, overrideQty = null) => {
    if (!selectedProduct) return;
    const pid = selectedProduct.product_id;
    const currentDecision = productDetailData?.decision?.decision || selectedProduct.decision;
    const qtyToSubmit = overrideQty !== null ? overrideQty : customQty;

    const payload = {
      product_id: pid,
      decision: currentDecision,
      action: action,
      modified_quantity: action === 'MODIFY' ? parseInt(qtyToSubmit, 10) : undefined,
      notes: approvalNotes || undefined
    };

    try {
      const res = await fetch('/approval', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(payload)
      });

      if (res.ok) {
        const record = await res.json();
        showToast(`Action saved: ${action} for ${record.product_name} (${record.final_quantity} units)`);
        
        // Refresh product list and detail
        fetchProducts();
        if (productDetailData) {
          setProductDetailData({
            ...productDetailData,
            approval: record
          });
        }
      } else {
        const err = await res.json();
        alert(`Approval error: ${err.detail || 'Failed to record action'}`);
      }
    } catch (e) {
      alert(`Network error: ${e.message}`);
    }
  };

  // Toast feedback trigger
  const showToast = (msg) => {
    setToastMessage(msg);
    setTimeout(() => {
      setToastMessage(null);
    }, 4500);
  };

  // Table Column Sorting
  const handleSort = (field) => {
    if (sortField === field) {
      setSortDirection(sortDirection === 'asc' ? 'desc' : 'asc');
    } else {
      setSortField(field);
      setSortDirection('asc');
    }
  };

  // Category counts
  const categoryCounts = useMemo(() => {
    const counts = { ALL: products.length };
    products.forEach(p => {
      if (p.category) {
        counts[p.category] = (counts[p.category] || 0) + 1;
      }
    });
    return counts;
  }, [products]);

  const categories = Object.keys(categoryCounts);

  // Filtered & Sorted Products
  const processedProducts = useMemo(() => {
    return products
      .filter(p => {
        const query = searchQuery.trim().toLowerCase();
        const matchesSearch = !query || 
          p.product_name.toLowerCase().includes(query) ||
          p.product_id.toLowerCase().includes(query) ||
          (p.brand && p.brand.toLowerCase().includes(query));

        const matchesDecision = decisionFilter === 'ALL' || p.decision.toUpperCase() === decisionFilter.toUpperCase();
        const matchesRisk = riskFilter === 'ALL' || p.stockout_risk.toUpperCase() === riskFilter.toUpperCase();
        const matchesCategory = categoryFilter === 'ALL' || p.category.toLowerCase() === categoryFilter.toLowerCase();

        return matchesSearch && matchesDecision && matchesRisk && matchesCategory;
      })
      .sort((a, b) => {
        let valA = a[sortField];
        let valB = b[sortField];

        if (typeof valA === 'string') {
          valA = valA.toLowerCase();
          valB = valB.toLowerCase();
        }

        if (valA < valB) return sortDirection === 'asc' ? -1 : 1;
        if (valA > valB) return sortDirection === 'asc' ? 1 : -1;
        return 0;
      });
  }, [products, searchQuery, decisionFilter, riskFilter, categoryFilter, sortField, sortDirection]);

  // Group products for Kanban View
  const kanbanGroups = useMemo(() => {
    return {
      INCREASE: processedProducts.filter(p => p.decision === 'INCREASE'),
      MAINTAIN: processedProducts.filter(p => p.decision === 'MAINTAIN'),
      REDUCE: processedProducts.filter(p => p.decision === 'REDUCE'),
    };
  }, [processedProducts]);

  return (
    <div className="app-container">
      {/* Toast Notification */}
      {toastMessage && (
        <div className="toast-container">
          <div className="toast">
            <CheckCircle2 size={18} color="#10b981" />
            <span>{toastMessage}</span>
          </div>
        </div>
      )}

      {/* Top Header */}
      <header className="app-header">
        <div className="brand-section">
          <div className="brand-logo">
            <Package size={26} />
          </div>
          <div>
            <h1 className="brand-title">Inventory Decision Engine</h1>
            <div className="brand-subtitle">
              <span>Operational Stock Optimization</span>
              <span className="pipeline-badge">
                <span className="status-dot"></span>
                Pipeline Active
              </span>
            </div>
          </div>
        </div>

        <div className="header-controls">
          {/* View Mode Toggle */}
          <div className="view-mode-toggle">
            <button 
              className={`view-mode-btn ${viewMode === 'table' ? 'active' : ''}`}
              onClick={() => setViewMode('table')}
              title="Table View"
            >
              <TableIcon size={15} />
              <span>Table</span>
            </button>
            <button 
              className={`view-mode-btn ${viewMode === 'kanban' ? 'active' : ''}`}
              onClick={() => setViewMode('kanban')}
              title="Decision Matrix"
            >
              <LayoutGrid size={15} />
              <span>Decision Matrix</span>
            </button>
          </div>

          <div className="status-pill">
            <span className="status-dot"></span>
            <span>{healthStatus?.total_products || products.length} Products</span>
          </div>
        </div>
      </header>

      {/* Key Metric KPI Cards (Click to filter) */}
      <section className="metrics-grid">
        <div 
          className={`metric-card total ${decisionFilter === 'ALL' && riskFilter === 'ALL' ? 'active-filter' : ''}`}
          onClick={() => { setDecisionFilter('ALL'); setRiskFilter('ALL'); }}
        >
          <div className="metric-header">
            <span className="metric-label">Total Catalog</span>
            <div className="metric-icon" style={{ background: 'rgba(99, 102, 241, 0.15)', color: '#818cf8' }}>
              <Package size={17} />
            </div>
          </div>
          <div className="metric-value-row">
            <div className="metric-value">{summary.total}</div>
            <span className="metric-percentage">SKUs</span>
          </div>
          <div className="metric-desc">All catalog items</div>
        </div>

        <div 
          className={`metric-card increase ${decisionFilter === 'INCREASE' ? 'active-filter' : ''}`}
          onClick={() => { setDecisionFilter('INCREASE'); setRiskFilter('ALL'); }}
        >
          <div className="metric-header">
            <span className="metric-label">Increase Orders</span>
            <div className="metric-icon" style={{ background: 'var(--color-increase-bg)', color: 'var(--color-increase)' }}>
              <TrendingUp size={17} />
            </div>
          </div>
          <div className="metric-value-row">
            <div className="metric-value">{summary.increase}</div>
            <span className="metric-percentage">
              {summary.total > 0 ? Math.round((summary.increase / summary.total) * 100) : 0}%
            </span>
          </div>
          <div className="metric-desc">Replenishment required</div>
        </div>

        <div 
          className={`metric-card maintain ${decisionFilter === 'MAINTAIN' ? 'active-filter' : ''}`}
          onClick={() => { setDecisionFilter('MAINTAIN'); setRiskFilter('ALL'); }}
        >
          <div className="metric-header">
            <span className="metric-label">Maintain Stock</span>
            <div className="metric-icon" style={{ background: 'var(--color-maintain-bg)', color: 'var(--color-maintain)' }}>
              <ShieldCheck size={17} />
            </div>
          </div>
          <div className="metric-value-row">
            <div className="metric-value">{summary.maintain}</div>
            <span className="metric-percentage">
              {summary.total > 0 ? Math.round((summary.maintain / summary.total) * 100) : 0}%
            </span>
          </div>
          <div className="metric-desc">Balanced stock levels</div>
        </div>

        <div 
          className={`metric-card reduce ${decisionFilter === 'REDUCE' ? 'active-filter' : ''}`}
          onClick={() => { setDecisionFilter('REDUCE'); setRiskFilter('ALL'); }}
        >
          <div className="metric-header">
            <span className="metric-label">Reduce Holdings</span>
            <div className="metric-icon" style={{ background: 'var(--color-reduce-bg)', color: 'var(--color-reduce)' }}>
              <TrendingDown size={17} />
            </div>
          </div>
          <div className="metric-value-row">
            <div className="metric-value">{summary.reduce}</div>
            <span className="metric-percentage">
              {summary.total > 0 ? Math.round((summary.reduce / summary.total) * 100) : 0}%
            </span>
          </div>
          <div className="metric-desc">Surplus stock reduction</div>
        </div>

        <div 
          className={`metric-card high-risk ${riskFilter === 'HIGH' ? 'active-filter' : ''}`}
          onClick={() => { setRiskFilter('HIGH'); setDecisionFilter('ALL'); }}
        >
          <div className="metric-header">
            <span className="metric-label">High Stockout Risk</span>
            <div className="metric-icon" style={{ background: 'var(--color-risk-high-bg)', color: 'var(--color-risk-high)' }}>
              <AlertTriangle size={17} />
            </div>
          </div>
          <div className="metric-value-row">
            <div className="metric-value">{summary.high_risk}</div>
            <span className="metric-percentage">Critical</span>
          </div>
          <div className="metric-desc">Days of stock &lt; lead time</div>
        </div>
      </section>

      {/* Category Pills Filter Bar */}
      <div className="category-pills-bar">
        {categories.map(cat => (
          <button
            key={cat}
            className={`category-pill ${categoryFilter === cat ? 'active' : ''}`}
            onClick={() => setCategoryFilter(cat)}
          >
            <span>{cat === 'ALL' ? 'All Categories' : cat}</span>
            <span className="category-pill-count">{categoryCounts[cat]}</span>
          </button>
        ))}
      </div>

      {/* Search & Filter Controls Bar */}
      <div className="controls-bar">
        <div className="search-input-wrapper">
          <Search className="search-icon" size={16} />
          <input 
            type="text" 
            className="search-input" 
            placeholder="Search products by name, ID, or brand..."
            value={searchQuery}
            onChange={(e) => setSearchQuery(e.target.value)}
          />
          {searchQuery && (
            <button className="search-clear-btn" onClick={() => setSearchQuery('')}>
              <X size={14} />
            </button>
          )}
        </div>

        <div className="filter-dropdowns">
          <select 
            className="filter-select"
            value={decisionFilter}
            onChange={(e) => setDecisionFilter(e.target.value)}
          >
            <option value="ALL">All Decisions</option>
            <option value="INCREASE">Increase</option>
            <option value="MAINTAIN">Maintain</option>
            <option value="REDUCE">Reduce</option>
          </select>

          <select 
            className="filter-select"
            value={riskFilter}
            onChange={(e) => setRiskFilter(e.target.value)}
          >
            <option value="ALL">All Risk Levels</option>
            <option value="HIGH">High Risk</option>
            <option value="MEDIUM">Medium Risk</option>
            <option value="LOW">Low Risk</option>
          </select>
        </div>
      </div>

      {/* Main Content Area: Table vs Kanban */}
      {loading ? (
        <div className="loading-box">
          <div className="spinner"></div>
          <p>Loading inventory decisions...</p>
        </div>
      ) : error ? (
        <div className="loading-box" style={{ color: '#f43f5e' }}>
          <AlertTriangle size={32} />
          <p>{error}</p>
          <button className="demo-scenario-btn" onClick={fetchProducts}>Retry</button>
        </div>
      ) : processedProducts.length === 0 ? (
        <div className="loading-box">
          <Package size={32} />
          <p>No products match the selected criteria.</p>
          <button 
            className="demo-scenario-btn"
            onClick={() => { setSearchQuery(''); setDecisionFilter('ALL'); setRiskFilter('ALL'); setCategoryFilter('ALL'); }}
          >
            Clear Filters
          </button>
        </div>
      ) : viewMode === 'table' ? (
        /* TABLE VIEW */
        <div className="table-card">
          <div className="table-wrapper">
            <table className="product-table">
              <thead>
                <tr>
                  <th className="sortable" onClick={() => handleSort('product_id')}>
                    <span className="th-content">
                      Product {sortField === 'product_id' && (sortDirection === 'asc' ? <ArrowUp size={12}/> : <ArrowDown size={12}/>)}
                    </span>
                  </th>
                  <th className="sortable" onClick={() => handleSort('available_inventory')}>
                    <span className="th-content">
                      Available Stock {sortField === 'available_inventory' && (sortDirection === 'asc' ? <ArrowUp size={12}/> : <ArrowDown size={12}/>)}
                    </span>
                  </th>
                  <th className="sortable" onClick={() => handleSort('forecast_7d')}>
                    <span className="th-content">
                      7d Demand {sortField === 'forecast_7d' && (sortDirection === 'asc' ? <ArrowUp size={12}/> : <ArrowDown size={12}/>)}
                    </span>
                  </th>
                  <th>30d Demand</th>
                  <th>Trend</th>
                  <th className="sortable" onClick={() => handleSort('days_of_stock')}>
                    <span className="th-content">
                      Stock Coverage {sortField === 'days_of_stock' && (sortDirection === 'asc' ? <ArrowUp size={12}/> : <ArrowDown size={12}/>)}
                    </span>
                  </th>
                  <th className="sortable" onClick={() => handleSort('stockout_risk')}>
                    <span className="th-content">
                      Risk {sortField === 'stockout_risk' && (sortDirection === 'asc' ? <ArrowUp size={12}/> : <ArrowDown size={12}/>)}
                    </span>
                  </th>
                  <th>Market Signal</th>
                  <th className="sortable" onClick={() => handleSort('decision')}>
                    <span className="th-content">
                      Decision {sortField === 'decision' && (sortDirection === 'asc' ? <ArrowUp size={12}/> : <ArrowDown size={12}/>)}
                    </span>
                  </th>
                  <th className="sortable" onClick={() => handleSort('recommended_order_quantity')}>
                    <span className="th-content">
                      Recommended PO {sortField === 'recommended_order_quantity' && (sortDirection === 'asc' ? <ArrowUp size={12}/> : <ArrowDown size={12}/>)}
                    </span>
                  </th>
                  <th>Status</th>
                  <th>Action</th>
                </tr>
              </thead>
              <tbody>
                {processedProducts.map(p => {
                  const coverageRatio = p.days_of_stock / Math.max(1, p.supplier_lead_time);
                  const isStockoutRiskHigh = p.days_of_stock < p.supplier_lead_time;
                  
                  return (
                    <tr 
                      key={p.product_id}
                      className={selectedProduct?.product_id === p.product_id ? 'selected-row' : ''}
                      onClick={() => handleOpenDetail(p)}
                    >
                      <td>
                        <div className="product-title-cell">
                          <span className="product-id-tag">{p.product_id}</span>
                          <span className="product-name-text">{p.product_name}</span>
                          <span className="product-cat-text">{p.category} &bull; {p.brand}</span>
                        </div>
                      </td>

                      <td>
                        <div><strong>{p.available_inventory.toLocaleString()}</strong> units</div>
                        <div style={{ fontSize: '11px', color: 'var(--text-faint)' }}>
                          {p.current_inventory.toLocaleString()} on-hand &bull; {p.reserved_inventory} reserved
                        </div>
                      </td>

                      <td>
                        <strong>{p.forecast_7d.toLocaleString()}</strong> units
                      </td>

                      <td>
                        {p.forecast_30d.toLocaleString()} units
                      </td>

                      <td>
                        <span className={`trend-badge ${p.sales_trend}`}>
                          {p.sales_trend === 'increasing' && <TrendingUp size={14} />}
                          {p.sales_trend === 'decreasing' && <TrendingDown size={14} />}
                          {p.sales_trend === 'stable' && <Minus size={14} />}
                          {p.sales_trend}
                        </span>
                      </td>

                      <td>
                        <div className="dos-gauge-container">
                          <div className="dos-value-line">
                            <span><strong>{p.days_of_stock.toFixed(1)}d</strong> stock</span>
                            <span style={{ color: 'var(--text-faint)' }}>Lead: {p.supplier_lead_time}d</span>
                          </div>
                          <div className="dos-bar-track">
                            <div 
                              className={`dos-bar-fill ${isStockoutRiskHigh ? 'danger' : coverageRatio < 2.0 ? 'warning' : 'healthy'}`}
                              style={{ width: `${Math.min(100, Math.max(8, (coverageRatio / 3.0) * 100))}%` }}
                            ></div>
                          </div>
                        </div>
                      </td>

                      <td>
                        <span className={`risk-badge ${p.stockout_risk.toLowerCase()}`}>
                          {p.stockout_risk}
                        </span>
                      </td>

                      <td>
                        <span className={`market-badge ${p.market_signal.toLowerCase()}`}>
                          {p.market_signal}
                        </span>
                      </td>

                      <td>
                        <span className={`decision-badge ${p.decision.toLowerCase()}`}>
                          {p.decision === 'INCREASE' && <TrendingUp size={12} />}
                          {p.decision === 'MAINTAIN' && <ShieldCheck size={12} />}
                          {p.decision === 'REDUCE' && <TrendingDown size={12} />}
                          {p.decision}
                        </span>
                      </td>

                      <td>
                        <strong>{p.recommended_order_quantity.toLocaleString()}</strong> units
                        {p.recommended_order_quantity > 0 && (
                          <div style={{ fontSize: '10.5px', color: 'var(--text-faint)' }}>
                            MOQ: {p.moq}
                          </div>
                        )}
                      </td>

                      <td>
                        <span className={`approval-badge ${p.approval_status.toLowerCase()}`}>
                          {p.approval_status}
                        </span>
                      </td>

                      <td>
                        <button 
                          className="demo-scenario-btn"
                          style={{ padding: '5px 12px', fontSize: '11.5px' }}
                          onClick={(e) => { e.stopPropagation(); handleOpenDetail(p); }}
                        >
                          View Details
                        </button>
                      </td>
                    </tr>
                  );
                })}
              </tbody>
            </table>
          </div>
        </div>
      ) : (
        /* KANBAN / DECISION MATRIX VIEW */
        <div className="kanban-grid">
          {/* INCREASE Column */}
          <div className="kanban-column">
            <div className="kanban-header" style={{ borderTop: '3px solid var(--color-increase)' }}>
              <div className="kanban-title" style={{ color: 'var(--color-increase)' }}>
                <TrendingUp size={16} />
                <span>INCREASE ORDERS</span>
              </div>
              <span className="kanban-count-pill" style={{ background: 'var(--color-increase-bg)', color: 'var(--color-increase)' }}>
                {kanbanGroups.INCREASE.length} SKUs
              </span>
            </div>
            <div className="kanban-cards-list">
              {kanbanGroups.INCREASE.map(p => (
                <div key={p.product_id} className="kanban-card" onClick={() => handleOpenDetail(p)}>
                  <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'flex-start' }}>
                    <div>
                      <span className="product-id-tag">{p.product_id}</span>
                      <div className="product-name-text" style={{ marginTop: '4px' }}>{p.product_name}</div>
                      <div className="product-cat-text">{p.category}</div>
                    </div>
                    <span className="risk-badge high">{p.stockout_risk}</span>
                  </div>

                  <div style={{ background: 'rgba(10,15,29,0.6)', padding: '8px 10px', borderRadius: '6px', fontSize: '12px' }}>
                    <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: '4px' }}>
                      <span style={{ color: 'var(--text-muted)' }}>Coverage:</span>
                      <strong>{p.days_of_stock.toFixed(1)}d (Lead: {p.supplier_lead_time}d)</strong>
                    </div>
                    <div style={{ display: 'flex', justifyContent: 'space-between' }}>
                      <span style={{ color: 'var(--text-muted)' }}>Recommended PO:</span>
                      <strong style={{ color: 'var(--color-increase)' }}>{p.recommended_order_quantity} units</strong>
                    </div>
                  </div>

                  <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', fontSize: '11px', color: 'var(--text-faint)' }}>
                    <span>MOQ: {p.moq} units</span>
                    <span className={`approval-badge ${p.approval_status.toLowerCase()}`}>{p.approval_status}</span>
                  </div>
                </div>
              ))}
            </div>
          </div>

          {/* MAINTAIN Column */}
          <div className="kanban-column">
            <div className="kanban-header" style={{ borderTop: '3px solid var(--color-maintain)' }}>
              <div className="kanban-title" style={{ color: 'var(--color-maintain)' }}>
                <ShieldCheck size={16} />
                <span>MAINTAIN STOCK</span>
              </div>
              <span className="kanban-count-pill" style={{ background: 'var(--color-maintain-bg)', color: 'var(--color-maintain)' }}>
                {kanbanGroups.MAINTAIN.length} SKUs
              </span>
            </div>
            <div className="kanban-cards-list">
              {kanbanGroups.MAINTAIN.map(p => (
                <div key={p.product_id} className="kanban-card" onClick={() => handleOpenDetail(p)}>
                  <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'flex-start' }}>
                    <div>
                      <span className="product-id-tag">{p.product_id}</span>
                      <div className="product-name-text" style={{ marginTop: '4px' }}>{p.product_name}</div>
                      <div className="product-cat-text">{p.category}</div>
                    </div>
                    <span className={`risk-badge ${p.stockout_risk.toLowerCase()}`}>{p.stockout_risk}</span>
                  </div>

                  <div style={{ background: 'rgba(10,15,29,0.6)', padding: '8px 10px', borderRadius: '6px', fontSize: '12px' }}>
                    <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: '4px' }}>
                      <span style={{ color: 'var(--text-muted)' }}>Days of Stock:</span>
                      <strong>{p.days_of_stock.toFixed(1)} days</strong>
                    </div>
                    <div style={{ display: 'flex', justifyContent: 'space-between' }}>
                      <span style={{ color: 'var(--text-muted)' }}>Status:</span>
                      <span style={{ color: 'var(--color-maintain)' }}>Balanced Stock</span>
                    </div>
                  </div>

                  <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', fontSize: '11px', color: 'var(--text-faint)' }}>
                    <span>Lead time: {p.supplier_lead_time}d</span>
                    <span className={`approval-badge ${p.approval_status.toLowerCase()}`}>{p.approval_status}</span>
                  </div>
                </div>
              ))}
            </div>
          </div>

          {/* REDUCE Column */}
          <div className="kanban-column">
            <div className="kanban-header" style={{ borderTop: '3px solid var(--color-reduce)' }}>
              <div className="kanban-title" style={{ color: 'var(--color-reduce)' }}>
                <TrendingDown size={16} />
                <span>REDUCE HOLDINGS</span>
              </div>
              <span className="kanban-count-pill" style={{ background: 'var(--color-reduce-bg)', color: 'var(--color-reduce)' }}>
                {kanbanGroups.REDUCE.length} SKUs
              </span>
            </div>
            <div className="kanban-cards-list">
              {kanbanGroups.REDUCE.map(p => (
                <div key={p.product_id} className="kanban-card" onClick={() => handleOpenDetail(p)}>
                  <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'flex-start' }}>
                    <div>
                      <span className="product-id-tag">{p.product_id}</span>
                      <div className="product-name-text" style={{ marginTop: '4px' }}>{p.product_name}</div>
                      <div className="product-cat-text">{p.category}</div>
                    </div>
                    <span className="risk-badge low">Overstock</span>
                  </div>

                  <div style={{ background: 'rgba(10,15,29,0.6)', padding: '8px 10px', borderRadius: '6px', fontSize: '12px' }}>
                    <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: '4px' }}>
                      <span style={{ color: 'var(--text-muted)' }}>Stock Coverage:</span>
                      <strong>{p.days_of_stock.toFixed(0)} days</strong>
                    </div>
                    <div style={{ display: 'flex', justifyContent: 'space-between' }}>
                      <span style={{ color: 'var(--text-muted)' }}>Action:</span>
                      <span style={{ color: 'var(--color-reduce)' }}>Surplus Stock</span>
                    </div>
                  </div>

                  <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', fontSize: '11px', color: 'var(--text-faint)' }}>
                    <span>Trend: {p.sales_trend}</span>
                    <span className={`approval-badge ${p.approval_status.toLowerCase()}`}>{p.approval_status}</span>
                  </div>
                </div>
              ))}
            </div>
          </div>
        </div>
      )}

      {/* Product Details Modal */}
      {selectedProduct && (
        <div className="modal-overlay" onClick={() => setSelectedProduct(null)}>
          <div className="modal-card" onClick={(e) => e.stopPropagation()}>
            {/* Modal Header */}
            <div className="modal-header">
              <div>
                <span className="product-id-tag">{selectedProduct.product_id}</span>
                <h2 style={{ fontFamily: 'var(--font-heading)', fontSize: '22px', marginTop: '4px', color: 'white' }}>
                  {selectedProduct.product_name}
                </h2>
                <p style={{ fontSize: '12.5px', color: 'var(--text-muted)' }}>
                  {selectedProduct.category} &bull; Brand: {selectedProduct.brand}
                </p>
              </div>
              <button className="modal-close-btn" onClick={() => setSelectedProduct(null)}>
                <X size={20} />
              </button>
            </div>

            {/* Modal Tabs Bar */}
            <div className="modal-tabs-bar">
              <button 
                className={`modal-tab-btn ${activeDetailTab === 'overview' ? 'active' : ''}`}
                onClick={() => setActiveDetailTab('overview')}
              >
                <Boxes size={15} />
                <span>Overview</span>
              </button>
              <button 
                className={`modal-tab-btn ${activeDetailTab === 'ai_rationale' ? 'active' : ''}`}
                onClick={() => setActiveDetailTab('ai_rationale')}
              >
                <Sparkles size={15} />
                <span>AI Rationale</span>
              </button>
              <button 
                className={`modal-tab-btn ${activeDetailTab === 'alternatives' ? 'active' : ''}`}
                onClick={() => setActiveDetailTab('alternatives')}
              >
                <ArrowRightLeft size={15} />
                <span>Alternatives</span>
              </button>
              <button 
                className={`modal-tab-btn ${activeDetailTab === 'approval' ? 'active' : ''}`}
                onClick={() => setActiveDetailTab('approval')}
              >
                <ShieldCheck size={15} />
                <span>Approval</span>
              </button>
            </div>

            {/* Modal Content */}
            <div className="modal-body">
              {productDetailLoading ? (
                <div className="loading-box">
                  <div className="spinner"></div>
                  <p>Loading product details...</p>
                </div>
              ) : (
                <>
                  {/* TAB 1: OVERVIEW & COVERAGE */}
                  {activeDetailTab === 'overview' && (
                    <>
                      {/* Decision Hero Banner */}
                      <div className={`decision-hero ${(productDetailData?.decision?.decision || selectedProduct.decision).toLowerCase()}`}>
                        <div className="decision-hero-left">
                          <span className="hero-recommendation-title">Recommendation</span>
                          <div className="hero-decision-val">
                            {productDetailData?.decision?.decision || selectedProduct.decision}
                          </div>
                          <div style={{ display: 'flex', gap: '10px', alignItems: 'center', marginTop: '4px' }}>
                            <span className={`risk-badge ${(productDetailData?.decision?.stockout_risk || selectedProduct.stockout_risk).toLowerCase()}`}>
                              Risk: {productDetailData?.decision?.stockout_risk || selectedProduct.stockout_risk}
                            </span>
                            <span style={{ fontSize: '13px', color: 'var(--text-muted)' }}>
                              Confidence: {Math.round((productDetailData?.decision?.confidence || selectedProduct.confidence || 0.8) * 100)}%
                            </span>
                          </div>
                        </div>

                        <div className="hero-order-box">
                          <div className="hero-order-label">Recommended Order</div>
                          <div className="hero-order-num">
                            {(productDetailData?.decision?.recommended_order_quantity ?? selectedProduct.recommended_order_quantity).toLocaleString()} units
                          </div>
                          <div style={{ fontSize: '11px', color: 'var(--text-faint)', marginTop: '4px' }}>
                            MOQ: {selectedProduct.moq} units
                          </div>
                        </div>
                      </div>

                      {/* Visual Coverage Gauge vs Supplier Lead Time */}
                      <div className="coverage-comparison-card">
                        <div className="coverage-header">
                          <div>
                            <strong style={{ fontSize: '14px', color: 'white' }}>Stock Coverage vs Supplier Lead Time</strong>
                            <div style={{ fontSize: '12px', color: 'var(--text-muted)' }}>
                              {selectedProduct.days_of_stock.toFixed(1)} days of stock available vs {selectedProduct.supplier_lead_time} days supplier lead time
                            </div>
                          </div>
                          <div style={{ textAlign: 'right' }}>
                            <span style={{ fontSize: '13px', fontWeight: '700', color: selectedProduct.days_of_stock < selectedProduct.supplier_lead_time ? '#f43f5e' : '#34d399' }}>
                              {(selectedProduct.days_of_stock / Math.max(1, selectedProduct.supplier_lead_time)).toFixed(2)}x Coverage
                            </span>
                          </div>
                        </div>
                        <div className="coverage-bar-track">
                          <div 
                            className={`coverage-bar-fill ${selectedProduct.days_of_stock < selectedProduct.supplier_lead_time ? 'danger' : 'healthy'}`}
                            style={{ 
                              background: selectedProduct.days_of_stock < selectedProduct.supplier_lead_time ? '#f43f5e' : '#10b981',
                              width: `${Math.min(100, Math.max(10, (selectedProduct.days_of_stock / (selectedProduct.supplier_lead_time * 2.5)) * 100))}%` 
                            }}
                          ></div>
                        </div>
                      </div>

                      {/* Supply Chain Metrics Grid */}
                      <div className="detail-grid">
                        <div className="detail-cell">
                          <div className="detail-cell-label">Available Inventory</div>
                          <div className="detail-cell-val">
                            {selectedProduct.available_inventory.toLocaleString()} units
                          </div>
                          <div style={{ fontSize: '11px', color: 'var(--text-faint)', marginTop: '3px' }}>
                            {selectedProduct.current_inventory} on-hand &bull; {selectedProduct.reserved_inventory} reserved
                          </div>
                        </div>

                        <div className="detail-cell">
                          <div className="detail-cell-label">Daily Demand Velocity</div>
                          <div className="detail-cell-val">
                            {(productDetailData?.decision?.metrics?.average_daily_demand || (selectedProduct.forecast_7d / 7)).toFixed(1)} units/day
                          </div>
                          <div style={{ fontSize: '11px', color: 'var(--text-faint)', marginTop: '3px' }}>
                            7d: {selectedProduct.forecast_7d} &bull; 30d: {selectedProduct.forecast_30d}
                          </div>
                        </div>

                        <div className="detail-cell">
                          <div className="detail-cell-label">Supplier Specifications</div>
                          <div className="detail-cell-val">
                            {selectedProduct.supplier_lead_time} days lead time
                          </div>
                          <div style={{ fontSize: '11px', color: 'var(--text-faint)', marginTop: '3px' }}>
                            MOQ: {selectedProduct.moq} units
                          </div>
                        </div>

                        <div className="detail-cell">
                          <div className="detail-cell-label">Safety Stock & Reorder Point</div>
                          <div className="detail-cell-val">
                            {productDetailData?.decision?.metrics?.safety_stock ?? 'Calculated'} / {selectedProduct.reorder_point}
                          </div>
                          <div style={{ fontSize: '11px', color: 'var(--text-faint)', marginTop: '3px' }}>
                            {productDetailData?.decision?.metrics?.is_reorder_point_fallback ? 'Calculated ROP' : 'Supplier ROP'}
                          </div>
                        </div>
                      </div>

                      {/* Step-by-Step MOQ Calculation Evidence */}
                      {productDetailData?.decision?.metrics?.order_calculation_evidence && (
                        <div className="reason-section">
                          <h3 className="section-title">
                            <BarChart3 size={17} color="#38bdf8" />
                            Order Quantity Breakdown
                          </h3>
                          <div className="moq-stepper-grid">
                            <div className="moq-step-card">
                              <div className="moq-step-title">30-Day Demand</div>
                              <div className="moq-step-val">
                                {productDetailData.decision.metrics.order_calculation_evidence.forecast_demand_horizon} units
                              </div>
                            </div>
                            <div className="moq-step-card">
                              <div className="moq-step-title">+ Safety Stock</div>
                              <div className="moq-step-val">
                                +{productDetailData.decision.metrics.order_calculation_evidence.safety_stock} units
                              </div>
                            </div>
                            <div className="moq-step-card">
                              <div className="moq-step-title">- Available Stock</div>
                              <div className="moq-step-val">
                                -{productDetailData.decision.metrics.order_calculation_evidence.available_inventory} units
                              </div>
                            </div>
                            <div className="moq-step-card">
                              <div className="moq-step-title">= Net Need</div>
                              <div className="moq-step-val" style={{ color: productDetailData.decision.metrics.order_calculation_evidence.net_requirement > 0 ? '#10b981' : '#94a3b8' }}>
                                {productDetailData.decision.metrics.order_calculation_evidence.net_requirement} units
                              </div>
                            </div>
                            <div className="moq-step-card" style={{ border: '1px solid rgba(16, 185, 129, 0.3)' }}>
                              <div className="moq-step-title">Recommended Order (MOQ: {selectedProduct.moq})</div>
                              <div className="moq-step-val" style={{ color: '#10b981' }}>
                                {productDetailData.decision.recommended_order_quantity} units
                              </div>
                            </div>
                          </div>
                        </div>
                      )}
                    </>
                  )}

                  {/* TAB 2: AI RATIONALE */}
                  {activeDetailTab === 'ai_rationale' && (
                    <>
                      {/* Key Reasons Checklist */}
                      <div className="reason-section">
                        <h3 className="section-title">
                          <CheckCircle2 size={18} color="#10b981" />
                          Key Decision Factors
                        </h3>
                        <ul className="reasons-list">
                          {(productDetailData?.decision?.reasons || []).map((reason, idx) => (
                            <li key={idx} className="reason-item">
                              <CheckCircle2 size={16} className="reason-check" />
                              <span>{reason}</span>
                            </li>
                          ))}
                        </ul>
                      </div>

                      {/* AI Executive Summary */}
                      {productDetailData?.ai_explanation && (
                        <div className="reason-section">
                          <h3 className="section-title">
                            <Sparkles size={18} color="#818cf8" />
                            AI Executive Summary
                          </h3>
                          <div className="ai-narrative-box">
                            {productDetailData.ai_explanation}
                          </div>
                        </div>
                      )}

                      {/* Market Context */}
                      {selectedProduct.market_intelligence && (
                        <div className="reason-section">
                          <h3 className="section-title">
                            <Info size={17} color="#38bdf8" />
                            Market Context
                          </h3>
                          <div style={{ fontSize: '13px', color: '#cbd5e1', marginBottom: '10px' }}>
                            Sentiment: <strong style={{ textTransform: 'capitalize' }}>{selectedProduct.market_intelligence.market_signal}</strong> &bull; Confidence: {Math.round((selectedProduct.market_intelligence.market_confidence || 0) * 100)}%
                          </div>
                          {selectedProduct.market_intelligence.market_reason && (
                            <p style={{ fontSize: '12.5px', color: 'var(--text-muted)', lineHeight: '1.5' }}>
                              {selectedProduct.market_intelligence.market_reason}
                            </p>
                          )}
                        </div>
                      )}
                    </>
                  )}

                  {/* TAB 3: ALTERNATIVES */}
                  {activeDetailTab === 'alternatives' && (
                    <div className="reason-section">
                      <h3 className="section-title">
                        <ArrowRightLeft size={18} color="#38bdf8" />
                        Alternative Product Recommendations
                      </h3>
                      <p style={{ fontSize: '13px', color: 'var(--text-muted)', marginBottom: '16px' }}>
                        High-velocity catalog alternatives for potential inventory rebalancing:
                      </p>

                      {productDetailData?.alternative_products && productDetailData.alternative_products.length > 0 ? (
                        <div className="alternatives-grid">
                          {productDetailData.alternative_products.map((alt) => (
                            <div key={alt.product_id} className="alternative-card">
                              <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                                <span className="alt-name">{alt.product_name}</span>
                                <span className="product-id-tag">{alt.product_id}</span>
                              </div>
                              <div style={{ fontSize: '11.5px', color: 'var(--text-faint)' }}>
                                Category: {alt.category} &bull; Brand: {alt.brand}
                              </div>
                              <div style={{ display: 'flex', gap: '8px', fontSize: '12px', margin: '4px 0' }}>
                                <span className={`trend-badge ${alt.sales_trend}`}>
                                  {alt.sales_trend === 'increasing' ? <TrendingUp size={13}/> : <Minus size={13}/>}
                                  {alt.sales_trend}
                                </span>
                                <span style={{ color: 'var(--text-muted)' }}>
                                  7d: {alt.forecast_7d} units
                                </span>
                              </div>
                              <div className="alt-reason">{alt.reason}</div>
                            </div>
                          ))}
                        </div>
                      ) : (
                        <p style={{ color: 'var(--text-faint)', fontSize: '13px' }}>
                          No alternative products required. Inventory is performing at optimal velocity.
                        </p>
                      )}
                    </div>
                  )}

                  {/* TAB 4: APPROVAL */}
                  {activeDetailTab === 'approval' && (
                    <div style={{ display: 'flex', flexDirection: 'column', gap: '18px' }}>
                      <div className="approval-action-bar">
                        <div>
                          <div style={{ fontSize: '14px', fontWeight: '700', color: 'white' }}>
                            Order Approval
                          </div>
                          <div style={{ fontSize: '12.5px', color: 'var(--text-muted)', marginTop: '2px' }}>
                            Current Status: <strong>{productDetailData?.approval?.action || selectedProduct.approval_status}</strong>
                            {productDetailData?.approval && (
                              <span> (Recorded PO: {productDetailData.approval.final_quantity} units)</span>
                            )}
                          </div>
                        </div>

                        <div className="approval-btn-group">
                          <button 
                            className="btn-approve"
                            onClick={() => handleApprovalSubmit('APPROVE', productDetailData?.decision?.recommended_order_quantity ?? selectedProduct.recommended_order_quantity)}
                          >
                            <ThumbsUp size={15} />
                            <span>Approve ({(productDetailData?.decision?.recommended_order_quantity ?? selectedProduct.recommended_order_quantity).toLocaleString()} units)</span>
                          </button>

                          <button 
                            className="btn-reject"
                            onClick={() => handleApprovalSubmit('REJECT', 0)}
                          >
                            <Ban size={15} />
                            <span>Reject (0 units)</span>
                          </button>
                        </div>
                      </div>

                      {/* Modify Quantity Panel */}
                      <div style={{ background: 'rgba(15, 23, 42, 0.75)', border: '1px solid var(--border-subtle)', borderRadius: '12px', padding: '20px', display: 'flex', flexDirection: 'column', gap: '14px' }}>
                        <div style={{ display: 'flex', alignItems: 'center', gap: '8px', color: 'white', fontWeight: '600' }}>
                          <Edit3 size={16} color="#38bdf8" />
                          <span>Adjust Order Quantity</span>
                        </div>

                        <div style={{ display: 'flex', gap: '16px', alignItems: 'flex-end', flexWrap: 'wrap' }}>
                          <div style={{ minWidth: '220px' }}>
                            <label style={{ fontSize: '12px', color: 'var(--text-muted)' }}>
                              Adjusted Quantity (MOQ: {selectedProduct.moq}):
                            </label>
                            <div style={{ display: 'flex', gap: '8px', marginTop: '6px' }}>
                              <button 
                                className="demo-scenario-btn" 
                                style={{ padding: '8px 12px' }}
                                onClick={() => setCustomQty(Math.max(0, customQty - selectedProduct.moq))}
                              >
                                -{selectedProduct.moq}
                              </button>
                              <input 
                                type="number" 
                                className="search-input" 
                                value={customQty}
                                onChange={(e) => setCustomQty(Math.max(0, parseInt(e.target.value, 10) || 0))}
                                min="0"
                                step={selectedProduct.moq}
                                style={{ textAlign: 'center', fontWeight: '700' }}
                              />
                              <button 
                                className="demo-scenario-btn" 
                                style={{ padding: '8px 12px' }}
                                onClick={() => setCustomQty(customQty + selectedProduct.moq)}
                              >
                                +{selectedProduct.moq}
                              </button>
                            </div>
                          </div>

                          <div style={{ flex: 1, minWidth: '240px' }}>
                            <label style={{ fontSize: '12px', color: 'var(--text-muted)' }}>
                              Approval Notes / Comments:
                            </label>
                            <input 
                              type="text" 
                              className="search-input" 
                              style={{ marginTop: '6px' }}
                              placeholder="e.g. Adjusted order for upcoming marketing promotion..."
                              value={approvalNotes}
                              onChange={(e) => setApprovalNotes(e.target.value)}
                            />
                          </div>

                          <button 
                            className="btn-modify"
                            style={{ height: '38px', whiteSpace: 'nowrap' }}
                            onClick={() => handleApprovalSubmit('MODIFY', customQty)}
                          >
                            Save Adjusted Order
                          </button>
                        </div>
                      </div>
                    </div>
                  )}
                </>
              )}
            </div>
          </div>
        </div>
      )}
    </div>
  );
}
